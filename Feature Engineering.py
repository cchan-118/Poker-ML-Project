import re
import csv
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from tqdm import tqdm
except:
    def tqdm(x, **kwargs):
        return x

RANK_MAP = {r: i + 2 for i, r in enumerate("23456789TJQKA")}
STRAIGHT_SETS = [{14, 2, 3, 4, 5}] + [set(range(s, s + 5)) for s in range(2, 11)]
FEATURES_DIR = Path("/Users/colinchan/Desktop/Trading 2025-26/Poker Algo/Linus Features")
ERRORS_FILE = FEATURES_DIR / "errors.csv"
ERROR_LOCK = None


# Helpers
def split_hands(text):
    starts = [m.start() for m in re.finditer(r"PokerStars Hand #", text)]
    return [text[s:e].splitlines() for s, e in zip(starts, starts[1:] + [None])]


def hands_from_file(file_path):
    text = Path(file_path).read_bytes().decode("utf-8", "ignore")
    return split_hands(text)


def set_error_lock(lock):
    global ERROR_LOCK
    ERROR_LOCK = lock


def log_error(file_path, hand_number, error_text):
    if ERROR_LOCK:
        ERROR_LOCK.acquire()
    new_file = not ERRORS_FILE.exists()
    with open(ERRORS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["file_path", "hand_number", "error"])
        w.writerow([str(file_path), str(hand_number), str(error_text)])
    if ERROR_LOCK:
        ERROR_LOCK.release()


def process_one_file(args):
    folder_path, file_path = args
    try:
        file_hands = hands_from_file(file_path)
        frames = []
        for hand in file_hands:
            try:
                frames.append(process_hand(hand, file_hands))
            except Exception as e:
                hdr = hand[0] if hand else ""
                hand_num = re.search(r"Hand #(\d+)", hdr).group(1) if re.search(r"Hand #(\d+)", hdr) else ""
                log_error(file_path, hand_num, f"{type(e).__name__}: {e}")
        if frames:
            out = pd.concat(frames, ignore_index=True).sort_values(["hand_number", "player_name"]).reset_index(drop=True)
            rel = Path(file_path).relative_to(folder_path)
            out_dir = FEATURES_DIR / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_dir / (rel.stem + ".csv"), index=False)
        return file_path
    except Exception as e:
        log_error(file_path, "", f"{type(e).__name__}: {e}")
        return file_path


def amt(pattern, line):
    return float(re.search(pattern, line).group(1))


def has_consecutive(ranks, length):
    if len(ranks) < length:
        return 0
    diffs = np.diff(ranks)
    need = length - 1
    if len(diffs) < need:
        return 0
    run = np.convolve((diffs == 1).astype(int), np.ones(need, dtype=int), "valid")
    return int((run == need).any())


def has_gutshot(ranks):
    if len(ranks) < 4:
        return 0
    return int(((ranks[3:] - ranks[:-3]) == 4).any())


def idx_star(lines, kw):
    return next((i for i, l in enumerate(lines) if kw in l and l.strip().startswith("*")), -1)


def idx(lines, kw):
    return next((i for i, l in enumerate(lines) if kw in l), len(lines))


def sum_pot(action_lines):
    invested, pot = {}, 0
    for line in action_lines:
        if line.startswith("Uncalled bet"):
            pot -= amt(r"Uncalled bet \(\$(\d+\.?\d*)\)", line)
            continue
        p = line.split(":")[0]
        if "posts small blind" in line:
            a = amt(r"posts small blind \$(\d+\.?\d*)", line)
        elif "posts big blind" in line:
            a = amt(r"posts big blind \$(\d+\.?\d*)", line)
        elif ": bets $" in line:
            a = amt(r"bets \$(\d+\.?\d*)", line)
        elif ": calls $" in line:
            a = amt(r"calls \$(\d+\.?\d*)", line)
        elif ": raises $" in line:
            raise_to = amt(r"to \$(\d+\.?\d*)", line)
            pot += raise_to - invested.get(p, 0)
            invested[p] = raise_to
            continue
        else:
            continue
        invested[p] = invested.get(p, 0) + a
        pot += a
    return pot


def eff_stack(lines, stop):
    hole = idx(lines, "HOLE CARDS")
    seats = [l for l in lines[:hole] if l.startswith("Seat ") and "is sitting out" not in l]
    stacks = {re.search(r": (.+?) \(", l).group(1): amt(r"\$(\d+\.?\d*)", l) for l in seats}
    contrib, street, folded = {p: 0 for p in stacks}, {p: 0 for p in stacks}, set()
    for line in lines[:stop]:
        if any(s in line for s in ["FLOP", "TURN", "RIVER"]) and line.strip().startswith("*"):
            street = {p: 0 for p in stacks}
        p = line.split(":")[0]
        if p not in stacks:
            continue
        if ": folds" in line:
            folded.add(p)
        for pat in [r"posts small blind \$(\d+\.?\d*)", r"posts big blind \$(\d+\.?\d*)",
                    r"bets \$(\d+\.?\d*)", r"calls \$(\d+\.?\d*)"]:
            if re.search(pat, line):
                contrib[p] += amt(pat, line)
                street[p] += amt(pat, line)
        if ": raises $" in line:
            raise_to = amt(r"to \$(\d+\.?\d*)", line)
            contrib[p] += raise_to - street[p]
            street[p] = raise_to
    remaining = [stacks[p] - contrib[p] for p in stacks if p not in folded]
    return min(remaining) if remaining else 0


def board_stats(line):
    cards = " ".join(re.findall(r"\[([^\]]+)\]", line)).split()
    ranks = np.array([RANK_MAP[c[0]] for c in cards], dtype=int)
    suits = np.array([c[-1] for c in cards])
    uniq = np.unique(ranks)
    uniq_low = np.unique(np.where(ranks == 14, 1, ranks)) if 14 in ranks else uniq
    diffs = np.diff(uniq)
    conn_mask = diffs == 1
    _, suit_counts = np.unique(suits, return_counts=True)
    _, rank_counts = np.unique(ranks, return_counts=True)
    max_s = int(suit_counts.max()) if len(suit_counts) else 0
    max_r = int(rank_counts.max()) if len(rank_counts) else 0
    # Straight Chance
    str_ch_1 = ((uniq[2:] - uniq[:-2]) <= 4).any() if len(uniq) >= 3 else 0
    str_ch_2 = ((uniq_low[2:] - uniq_low[:-2]) <= 4).any() if len(uniq_low) >= 3 else 0
    str_ch = int(len(uniq) >= 3 and (str_ch_1 or str_ch_2))
    paired = int(len(uniq) < len(ranks))
    trips = int(max_r >= 3)
    connected = int(conn_mask.any())
    top_conn = int(uniq[1:][conn_mask].max()) if connected else 0
    rainbow = int(len(np.unique(suits)) == len(suits))
    return (ranks.tolist(), suits.tolist(), int(ranks.max()), paired, trips, connected, top_conn, rainbow,
            int(max_s == 2), int(max_s >= 3), str_ch, int(max_s >= 3))


def parse_cards(lines, st_idx, show_line):
    board = " ".join(re.findall(r"\[([^\]]+)\]", lines[st_idx])).split()
    hole = re.search(r"\[([^\]]+)\]", show_line).group(1).split()
    return [RANK_MAP[c[0]] for c in board], [c[-1] for c in board], [RANK_MAP[c[0]] for c in hole], [c[-1] for c in hole], hole


def player_actions(lines, player, start, end):
    acts = lines[start + 1:end]
    return ([l for l in acts if l.startswith(player + ": bets $")], [l for l in acts if l.startswith(player + ": raises $")],
            [l for l in acts if l.startswith(player + ": calls $")], [l for l in acts if l.startswith(player + ": folds")])


def action_feats(bets, raises, calls, folds, bb, pot):
    # Bet Size
    bet_amt = sum(amt(r"bets \$(\d+\.?\d*)", l) for l in bets) if bets else 0
    bet_sz = bet_amt / (pot * bb) if bets and pot > 0 else 0
    # Raise Size
    raise_amt = sum(amt(r"to \$(\d+\.?\d*)", l) for l in raises) if raises else 0
    raise_sz = raise_amt / (pot * bb) if raises and pot > 0 else 0
    return bet_sz, raise_sz, int(bool(bets)), int(bool(raises)), int(bool(calls)), int(bool(folds))


def draw_count(all_ranks, all_suits):
    ranks = np.array(all_ranks, dtype=int)
    suits = np.array(all_suits)
    uniq = np.unique(ranks)
    uniq_low = np.unique(np.where(ranks == 14, 1, ranks)) if 14 in ranks else uniq
    _, suit_counts = np.unique(suits, return_counts=True)
    max_s = int(suit_counts.max()) if len(suit_counts) else 0
    fd = int(max_s >= 4)
    oe = int(has_consecutive(uniq, 4) or has_consecutive(uniq_low, 4))
    gs = int((not oe) and (has_gutshot(uniq) or has_gutshot(uniq_low)))
    draw = 4 * gs + 8 * oe + 9 * fd - fd * gs - 2 * fd * oe
    return draw, max_s, uniq.tolist(), uniq_low.tolist()


def made_straight(rl):
    ranks = np.array(rl, dtype=int)
    return bool(has_consecutive(ranks, 5))


def made_hands(all_ranks, max_s, uniq, uniq_low):
    counts = np.bincount(np.array(all_ranks, dtype=int), minlength=15)
    sorted_counts = np.sort(counts)[::-1]
    top, second = int(sorted_counts[0]), int(sorted_counts[1])
    return (int(made_straight(uniq) or made_straight(uniq_low)), int(max_s >= 5),
            int(top >= 3), int(top >= 3 and second >= 2), int(top >= 4))


def blocker_feats(df, b_ranks, b_suits, h_ranks, h_suits, h_cards, street):
    paired = int(df[f"paired_{street}"].iloc[0])
    str_fl = int(df[f"straight_chance_{street}"].iloc[0])
    fl_fl = int(df[f"flush_chance_{street}"].iloc[0]) if street != "flop" else (int(df["two_tone_flop"].iloc[0]) or int(df["monotone_flop"].iloc[0]))
    fl_suit = max(set(b_suits), key=b_suits.count) if fl_fl else ""
    # Nut Blocker
    ace_bl = int(any(c[0] == "A" and c[-1] == fl_suit for c in h_cards)) if fl_suit else 0
    semi_help = 0
    if str_fl:
        b_low = [1 if r == 14 else r for r in b_ranks]
        use_low = (14 in b_ranks) and (max(b_low) - min(b_low) < max(b_ranks) - min(b_ranks))
        b_use = b_low if use_low else b_ranks
        h_use = [1 if use_low and r == 14 else r for r in h_ranks] if use_low else h_ranks
        semi_help = int(any(max(b_use) - 4 <= r <= max(b_use) for r in h_use))
    pocket_set = int(paired and h_ranks[0] == h_ranks[1] and h_ranks[0] in b_ranks)
    nut_bl = int(ace_bl or semi_help or pocket_set)
    # Unblocker
    unbl = int(h_ranks[0] == h_ranks[1] == min(b_ranks))
    return nut_bl, unbl, fl_fl, str_fl


def value_flags(b_ranks, h_ranks, all_r, all_s, uniq, uniq_low):
    b = np.array(b_ranks, dtype=int)
    h = np.array(h_ranks, dtype=int)
    counts = np.bincount(np.array(all_r, dtype=int), minlength=15)
    _, suit_counts = np.unique(np.array(all_s), return_counts=True)
    max_s = int(suit_counts.max()) if len(suit_counts) else 0
    b_paired = int(len(np.unique(b)) < len(b))
    top_pr = int((not b_paired) and (int(b.max()) in h_ranks))
    pocket = int(h_ranks[0] == h_ranks[1])
    overpair = int(pocket and h_ranks[0] > int(b.max()))
    hole_set = set(h_ranks)
    two_pair = int(len([r for r in hole_set if counts[r] >= 2]) >= 2)
    trips = int(any(counts[r] >= 3 for r in hole_set))
    quads = int(any(counts[r] >= 4 for r in hole_set))
    fh = int(int(counts.max()) >= 3 and any(counts[r] >= 2 for r in hole_set))
    val_core = int(quads or fh or trips or two_pair or made_straight(uniq) or made_straight(uniq_low) or max_s >= 5)
    val_fl = int(val_core or overpair or top_pr)
    any_pr = int(pocket or bool(hole_set & set(b_ranks)))
    no_pr = int(not any_pr)
    return val_fl, top_pr, any_pr, no_pr, overpair


def hand_feats(h_ranks):
    # Top Card Hand
    top = max(h_ranks)
    # Bottom Card Hand
    bot = min(h_ranks)
    # Paired Hand
    paired = int(h_ranks[0] == h_ranks[1])
    # Connected Hand
    conn = int(abs(h_ranks[0] - h_ranks[1]) == 1)
    return top, bot, paired, conn


def player_contrib(lines, player):
    total, street = 0, 0
    for line in lines:
        if any(s in line for s in ["FLOP", "TURN", "RIVER"]) and line.strip().startswith("*"):
            street = 0
        if line.startswith("Uncalled bet") and ("returned to " + player) in line:
            a = amt(r"Uncalled bet \(\$(\d+\.?\d*)\)", line)
            total -= a
            street -= a
        if not line.startswith(player + ":"):
            continue
        for pat in [r"posts small blind \$(\d+\.?\d*)", r"posts big blind \$(\d+\.?\d*)",
                    r"bets \$(\d+\.?\d*)", r"calls \$(\d+\.?\d*)"]:
            if re.search(pat, line):
                total += amt(pat, line)
                street += amt(pat, line)
        if ": raises $" in line:
            raise_to = amt(r"to \$(\d+\.?\d*)", line)
            total += raise_to - street
            street = raise_to
    return total


def compute_pnl(lines, player, all_players, bb):
    summary = idx(lines, "SUMMARY")
    acts = lines[:summary]
    winners = {l.split(" collected $")[0] for l in acts if " collected $" in l}
    n_win = len(winners) if winners else 1
    p_con = player_contrib(acts, player)
    o_con = sum(player_contrib(acts, p) for p in all_players if p != player)
    won = player in winners
    # PnL
    pnl = (o_con - p_con) / n_win if (won and n_win > 1) else (o_con if won else -p_con)
    return pnl / bb


def compute_nuts(str_ch, fl_ch, has_str, has_fl, has_trips, has_fh, has_quads):
    return int((str_ch == 1 and fl_ch == 0 and has_str) or (fl_ch == 1 and has_fl) or
               (str_ch == 0 and fl_ch == 0 and has_trips) or has_fh or has_quads)


# 3) Basic features
def extract_basic(df, lines):
    hdr = lines[0]
    # Hand Number
    hand_num = int(re.search(r"Hand #(\d+)", hdr).group(1))
    # Hand Datetime
    dt = pd.to_datetime(re.search(r"- (\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2})", hdr).group(1))
    # Small Blind Size
    sb = float(re.search(r"\(\$(\d+\.?\d*)/\$(\d+\.?\d*)", hdr).group(1))
    # Big Blind Size
    bb = float(re.search(r"\(\$(\d+\.?\d*)/\$(\d+\.?\d*)", hdr).group(2))
    hole = idx(lines, "HOLE CARDS")
    seats = [l for l in lines[:hole] if l.startswith("Seat ")]
    # Player Count
    n_players = len(seats) - sum("is sitting out" in s for s in seats)
    return df.assign(hand_number=hand_num, datetime=dt, players_count=n_players, big_blind=bb, small_blind=sb)


# 4) Preflop
def extract_preflop(df, lines):
    bb, n_players = df["big_blind"].iloc[0], df["players_count"].iloc[0]
    hole, flop = idx(lines, "HOLE CARDS"), idx_star(lines, "FLOP")
    flop = flop if flop != -1 else len(lines)
    pf_lines = lines[hole + 1:flop]
    calls = [amt(r"calls \$(\d+\.?\d*)", l) for l in pf_lines if ": calls $" in l]
    # Limp Count
    limps = sum(a == bb for a in calls)
    # Raise Count
    raises = sum(1 for l in pf_lines if ": raises $" in l)
    # Pot Preflop
    pot_pf = sum_pot(lines[:flop]) / bb
    # Postflop Players
    pf_players = n_players - sum(": folds" in l for l in pf_lines)
    return df.assign(limps=limps, raises=raises, postflop_players=pf_players, pot_preflop=pot_pf)


# 5) Flop
def extract_flop(df, lines):
    bb, pot_pf = df["big_blind"].iloc[0], df["pot_preflop"].iloc[0]
    flop = idx_star(lines, "FLOP")
    if flop == -1:
        return df.assign(SPR_flop=-1, pot_flop=-1, top_card_flop=-1, paired_flop=-1, connected_flop=-1,
                         top_connected_flop=-1, rainbow_flop=-1, two_tone_flop=-1, monotone_flop=-1,
                         straight_chance_flop=-1, flush_chance_flop=-1)
    turn = idx_star(lines, "TURN")
    turn = turn if turn != -1 else len(lines)
    # SPR Flop
    spr = eff_stack(lines, flop) / (pot_pf * bb)
    # Pot Flop
    pot = pot_pf + sum_pot(lines[flop + 1:turn]) / bb
    _, _, top, paired, _, conn, top_conn, rainbow, two_tone, mono, str_ch, fl_ch = board_stats(lines[flop])
    return df.assign(SPR_flop=spr, pot_flop=pot, top_card_flop=top, paired_flop=paired, connected_flop=conn,
                     top_connected_flop=top_conn, rainbow_flop=rainbow, two_tone_flop=two_tone,
                     monotone_flop=mono, straight_chance_flop=str_ch, flush_chance_flop=fl_ch)


# 6) Turn
def extract_turn(df, lines):
    bb, pot_flop = df["big_blind"].iloc[0], df["pot_flop"].iloc[0]
    turn = idx_star(lines, "TURN")
    if turn == -1:
        return df.assign(SPR_turn=-1, pot_turn=-1, top_card_turn=-1, paired_turn=-1, trips_turn=-1,
                         connected_turn=-1, top_connected_turn=-1, rainbow_turn=-1, two_tone_turn=-1,
                         monotone_turn=-1, straight_chance_turn=-1, flush_chance_turn=-1, nut_changing_turn=-1)
    river = idx_star(lines, "RIVER")
    river = river if river != -1 else len(lines)
    # SPR Turn
    spr = eff_stack(lines, turn) / (pot_flop * bb)
    # Pot Turn
    pot = pot_flop + sum_pot(lines[turn + 1:river]) / bb
    _, _, top, paired, trips, conn, top_conn, rainbow, two_tone, mono, str_ch, fl_ch = board_stats(lines[turn])
    # Nut Changing Turn
    nut_ch = int((df["paired_flop"].iloc[0] == 0 and paired) or (top > df["top_card_flop"].iloc[0]) or
                 (df["straight_chance_flop"].iloc[0] == 0 and str_ch) or (df["flush_chance_flop"].iloc[0] == 0 and fl_ch))
    return df.assign(SPR_turn=spr, pot_turn=pot, top_card_turn=top, paired_turn=paired, trips_turn=trips,
                     connected_turn=conn, top_connected_turn=top_conn, rainbow_turn=rainbow, two_tone_turn=two_tone,
                     monotone_turn=mono, straight_chance_turn=str_ch, flush_chance_turn=fl_ch, nut_changing_turn=nut_ch)


# 7) River
def extract_river(df, lines):
    bb, pot_turn = df["big_blind"].iloc[0], df["pot_turn"].iloc[0]
    river = idx_star(lines, "RIVER")
    if river == -1:
        return df.assign(SPR_river=-1, pot_river=-1, top_card_river=-1, paired_river=-1, trips_river=-1,
                         connected_river=-1, top_connected_river=-1, rainbow_river=-1, two_tone_river=-1,
                         monotone_river=-1, straight_chance_river=-1, flush_chance_river=-1, nut_changing_river=-1)
    showdown = idx(lines, "SHOW DOWN")
    # SPR River
    spr = eff_stack(lines, river) / (pot_turn * bb)
    # Pot River
    pot = pot_turn + sum_pot(lines[river + 1:showdown]) / bb
    _, _, top, paired, trips, conn, top_conn, rainbow, two_tone, mono, str_ch, fl_ch = board_stats(lines[river])
    # Nut Changing River
    nut_ch = int((df["paired_turn"].iloc[0] == 0 and paired) or (top > df["top_card_turn"].iloc[0]) or
                 (df["straight_chance_turn"].iloc[0] == 0 and str_ch) or (df["flush_chance_turn"].iloc[0] == 0 and fl_ch))
    return df.assign(SPR_river=spr, pot_river=pot, top_card_river=top, paired_river=paired, trips_river=trips,
                     connected_river=conn, top_connected_river=top_conn, rainbow_river=rainbow, two_tone_river=two_tone,
                     monotone_river=mono, straight_chance_river=str_ch, flush_chance_river=fl_ch, nut_changing_river=nut_ch)


# 8) Player flop features
def player_flop(df, lines, player):
    act_names = ["bet_size_flop", "raise_size_flop", "bettor_flop", "raiser_flop", "caller_flop", "folder_flop"]
    card_names = ["top_card_hand", "bottom_card_hand", "paired_hand", "connected_hand", "draw_count_flop",
                  "backdoor_flush_flop", "backdoor_straight_flop", "nut_blocker_flop", "unblocker_flop",
                  "trash_flop", "bluff_flop", "semibluff_flop", "bluffcatch_flop", "value_flop", "nuts_flop"]
    bb, pot_pf = df["big_blind"].iloc[0], df["pot_preflop"].iloc[0]
    flop = idx_star(lines, "FLOP")
    if flop == -1:
        return {n: -1 for n in act_names + card_names}
    turn, sd, summ = idx_star(lines, "TURN"), idx(lines, "SHOW DOWN"), idx(lines, "SUMMARY")
    end = min(i for i in [turn, sd, summ, len(lines)] if i > flop)
    bets, raises, calls, folds = player_actions(lines, player, flop, end)
    bet_sz, raise_sz, bettor, raiser, caller, folder = action_feats(bets, raises, calls, folds, bb, pot_pf)
    act = {"bet_size_flop": bet_sz, "raise_size_flop": raise_sz, "bettor_flop": bettor,
           "raiser_flop": raiser, "caller_flop": caller, "folder_flop": folder}
    sd_lines = lines[sd:summ] if sd < len(lines) else []
    show = next((l for l in sd_lines if l.startswith(player + ": shows [")), "")
    if not show:
        return {**act, **{n: -1 for n in card_names}}
    f_ranks, f_suits, h_ranks, h_suits, h_cards = parse_cards(lines, flop, show)
    top_h, bot_h, paired_h, conn_h = hand_feats(h_ranks)
    all_r, all_s = h_ranks + f_ranks, h_suits + f_suits
    # Draw Count Flop
    dc, max_s, uniq, uniq_low = draw_count(all_r, all_s)
    # Backdoor Straight Flop
    bd_str = int(any(len(set(uniq) & ss) >= 3 or len(set(uniq_low) & ss) >= 3 for ss in STRAIGHT_SETS))
    # Backdoor Flush Flop
    bd_fl = int(any(f_suits.count(s) == 2 and h_suits.count(s) == 1 for s in set(all_s)))
    nut_bl, unbl, fl_fl, _ = blocker_feats(df, f_ranks, f_suits, h_ranks, h_suits, h_cards, "flop")
    val_fl, top_pr, any_pr, no_pr, _ = value_flags(f_ranks, h_ranks, all_r, all_s, uniq, uniq_low)
    str_ch, fl_ch = int(df["straight_chance_flop"].iloc[0]), int(df["flush_chance_flop"].iloc[0])
    has_str, has_fl, has_trips, has_fh, has_quads = made_hands(all_r, max_s, uniq, uniq_low)
    # Nuts Flop
    nuts = compute_nuts(str_ch, fl_ch, has_str, has_fl, has_trips, has_fh, has_quads)
    # Bluffcatch Flop
    bc = int(any_pr and not top_pr and not val_fl)
    # Semibluff Flop
    sb = int(no_pr and dc > 0 and not val_fl)
    # Bluff Flop
    bl = int(no_pr and nut_bl and not sb and not val_fl)
    # Trash Flop
    tr = int(no_pr and not val_fl and not sb and not bl)
    # Value Flop
    val = int(val_fl and not nuts)
    return {**act, "top_card_hand": top_h, "bottom_card_hand": bot_h, "paired_hand": paired_h, "connected_hand": conn_h,
            "draw_count_flop": dc, "backdoor_flush_flop": bd_fl, "backdoor_straight_flop": bd_str,
            "nut_blocker_flop": nut_bl, "unblocker_flop": unbl, "trash_flop": tr, "bluff_flop": bl,
            "semibluff_flop": sb, "bluffcatch_flop": bc, "value_flop": val, "nuts_flop": nuts}


# 9) Player turn features
def player_turn(df, lines, player):
    act_names = ["bet_size_turn", "raise_size_turn", "bettor_turn", "raiser_turn", "caller_turn", "folder_turn"]
    card_names = ["draw_count_turn", "nut_blocker_turn", "unblocker_turn", "trash_turn", "bluff_turn",
                  "semibluff_turn", "bluffcatch_turn", "value_turn", "nuts_turn", "improved_hand_turn"]
    bb, pot_flop = df["big_blind"].iloc[0], df["pot_flop"].iloc[0]
    turn = idx_star(lines, "TURN")
    if turn == -1:
        return {n: -1 for n in act_names + card_names}
    river, sd, summ = idx_star(lines, "RIVER"), idx(lines, "SHOW DOWN"), idx(lines, "SUMMARY")
    end = min(i for i in [river, sd, summ, len(lines)] if i > turn)
    bets, raises, calls, folds = player_actions(lines, player, turn, end)
    bet_sz, raise_sz, bettor, raiser, caller, folder = action_feats(bets, raises, calls, folds, bb, pot_flop)
    act = {"bet_size_turn": bet_sz, "raise_size_turn": raise_sz, "bettor_turn": bettor,
           "raiser_turn": raiser, "caller_turn": caller, "folder_turn": folder}
    sd_lines = lines[sd:summ] if sd < len(lines) else []
    show = next((l for l in sd_lines if l.startswith(player + ": shows [")), "")
    if not show:
        return {**act, **{n: -1 for n in card_names}}
    b_ranks, b_suits, h_ranks, h_suits, h_cards = parse_cards(lines, turn, show)
    all_r, all_s = h_ranks + b_ranks, h_suits + b_suits
    # Draw Count Turn
    dc, max_s, uniq, uniq_low = draw_count(all_r, all_s)
    nut_bl, unbl, fl_fl, _ = blocker_feats(df, b_ranks, b_suits, h_ranks, h_suits, h_cards, "turn")
    val_fl, top_pr, any_pr, no_pr, _ = value_flags(b_ranks, h_ranks, all_r, all_s, uniq, uniq_low)
    str_ch, fl_ch = int(df["straight_chance_turn"].iloc[0]), int(df["flush_chance_turn"].iloc[0])
    has_str, has_fl, has_trips, has_fh, has_quads = made_hands(all_r, max_s, uniq, uniq_low)
    # Nuts Turn
    nuts = compute_nuts(str_ch, fl_ch, has_str, has_fl, has_trips, has_fh, has_quads)
    # Bluffcatch Turn
    bc = int(any_pr and not val_fl and ((top_pr and fl_fl) or not top_pr))
    # Semibluff Turn
    sb = int(no_pr and dc > 0 and not val_fl)
    # Bluff Turn
    bl = int(no_pr and nut_bl and not sb and not val_fl)
    # Trash Turn
    tr = int(no_pr and not val_fl and not sb and not bl)
    # Value Turn
    val = int(val_fl and not nuts)
    flop_f = player_flop(df, lines, player)
    # Improved Hand Turn
    imp = int((flop_f["trash_flop"] == 1 and tr == 0) or (flop_f["bluff_flop"] == 1 and bl == 0) or
              (flop_f["semibluff_flop"] == 1 and sb == 0) or (flop_f["bluffcatch_flop"] == 1 and bc == 0) or
              (flop_f["value_flop"] == 1 and val == 0 and bc == 0))
    return {**act, "draw_count_turn": dc, "nut_blocker_turn": nut_bl, "unblocker_turn": unbl, "trash_turn": tr,
            "bluff_turn": bl, "semibluff_turn": sb, "bluffcatch_turn": bc, "value_turn": val,
            "nuts_turn": nuts, "improved_hand_turn": imp}


# 10) Player river features
def player_river(df, lines, player):
    act_names = ["bet_size_river", "raise_size_river", "bettor_river", "raiser_river", "caller_river", "folder_river"]
    card_names = ["nut_blocker_river", "unblocker_river", "trash_river", "bluff_river", "bluffcatch_river",
                  "value_river", "nuts_river", "improved_hand_river"]
    bb, pot_turn = df["big_blind"].iloc[0], df["pot_turn"].iloc[0]
    river = idx_star(lines, "RIVER")
    sd, summ = idx(lines, "SHOW DOWN"), idx(lines, "SUMMARY")
    if river == -1:
        return {n: -1 for n in act_names + card_names}
    end = min(i for i in [sd, summ, len(lines)] if i > river)
    bets, raises, calls, folds = player_actions(lines, player, river, end)
    bet_sz, raise_sz, bettor, raiser, caller, folder = action_feats(bets, raises, calls, folds, bb, pot_turn)
    act = {"bet_size_river": bet_sz, "raise_size_river": raise_sz, "bettor_river": bettor,
           "raiser_river": raiser, "caller_river": caller, "folder_river": folder}
    sd_lines = lines[sd:summ] if sd < len(lines) else []
    show = next((l for l in sd_lines if l.startswith(player + ": shows [")), "")
    if not show:
        return {**act, **{n: -1 for n in card_names}}
    b_ranks, b_suits, h_ranks, h_suits, h_cards = parse_cards(lines, river, show)
    all_r, all_s = h_ranks + b_ranks, h_suits + b_suits
    dc, max_s, uniq, uniq_low = draw_count(all_r, all_s)
    nut_bl, unbl, fl_fl, str_fl = blocker_feats(df, b_ranks, b_suits, h_ranks, h_suits, h_cards, "river")
    val_fl, top_pr, any_pr, no_pr, _ = value_flags(b_ranks, h_ranks, all_r, all_s, uniq, uniq_low)
    str_ch, fl_ch = int(df["straight_chance_river"].iloc[0]), int(df["flush_chance_river"].iloc[0])
    has_str, has_fl, has_trips, has_fh, has_quads = made_hands(all_r, max_s, uniq, uniq_low)
    # Nuts River
    nuts = compute_nuts(str_ch, fl_ch, has_str, has_fl, has_trips, has_fh, has_quads)
    sb_riv = int(no_pr and dc > 0 and not val_fl)
    # Bluffcatch River
    bc = int(any_pr and not val_fl and ((top_pr and (str_fl or fl_fl)) or not top_pr))
    # Bluff River
    bl = int(no_pr and nut_bl and not val_fl)
    # Trash River
    tr = int(no_pr and not val_fl and not bl)
    # Value River
    val = int(val_fl and not nuts)
    turn_f = player_turn(df, lines, player)
    # Improved Hand River
    imp = int((turn_f["trash_turn"] == 1 and tr == 0) or (turn_f["bluff_turn"] == 1 and bl == 0) or
              (turn_f["semibluff_turn"] == 1 and sb_riv == 0) or (turn_f["bluffcatch_turn"] == 1 and bc == 0) or
              (turn_f["value_turn"] == 1 and val == 0 and bc == 0))
    return {**act, "nut_blocker_river": nut_bl, "unblocker_river": unbl, "trash_river": tr, "bluff_river": bl,
            "bluffcatch_river": bc, "value_river": val, "nuts_river": nuts, "improved_hand_river": imp}


# 11) Player info
def extract_players(df, lines, all_hands_list):
    dt = df["datetime"].iloc[0]
    hole, flop = idx(lines, "HOLE CARDS"), idx_star(lines, "FLOP")
    flop = flop if flop != -1 else len(lines)
    pf_lines = lines[hole + 1:flop]
    sd, summ = idx(lines, "SHOW DOWN"), idx(lines, "SUMMARY")
    sd_lines = lines[sd:summ] if sd < len(lines) else []
    seats = [l for l in lines[:hole] if l.startswith("Seat ") and "is sitting out" not in l]
    seat_map = {int(re.search(r"Seat (\d+):", l).group(1)): re.search(r": (.+?) \(", l).group(1) for l in seats}
    name_seat = {n: s for s, n in seat_map.items()}
    folded = {l.split(":")[0] for l in pf_lines if ": folds" in l}
    active = [p for p in seat_map.values() if p not in folded]
    all_p = list(seat_map.values())
    agg_map = {l.split(":")[0]: 0 if ": calls $" in l else 1 for l in pf_lines if ": calls $" in l or ": raises $" in l}
    btn = int(re.search(r"Seat #(\d+) is the button", next(l for l in lines if "is the button" in l)).group(1))
    order = list(range(btn + 1, max(seat_map.keys()) + 1)) + list(range(1, btn + 1))
    rank = {s: i for i, s in enumerate(order)}
    ip = max(active, key=lambda p: rank.get(name_seat[p], -1))
    past = []
    for h in all_hands_list:
        h_dt = pd.to_datetime(re.search(r"- (\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2})", h[0]).group(1))
        if h_dt > dt:
            continue
        h_hole = idx(h, "HOLE CARDS")
        h_flop = next((i for i, l in enumerate(h) if "FLOP" in l and l.strip().startswith("*")), len(h))
        h_pf = h[h_hole + 1:h_flop]
        seated = {re.search(r": (.+?) \(", l).group(1) for l in h[:h_hole] if l.startswith("Seat ") and "is sitting out" not in l}
        vpip = {l.split(":")[0] for l in h_pf if ": calls $" in l or ": raises $" in l}
        three_bet, raises = set(), set()
        for l in h_pf:
            if ": raises $" in l:
                r = l.split(":")[0]
                if raises - {r}:
                    three_bet.add(r)
                raises.add(r)
        past.append((seated, vpip, three_bet))
    rows = []
    base = df.iloc[0].to_dict()
    for p in active:
        # Reached Showdown
        reached_sd = int(any(l.startswith(p + ": shows [") for l in sd_lines))
        hands = sum(p in s for s, _, _ in past)
        flop_f = player_flop(df, lines, p)
        turn_f = player_turn(df, lines, p)
        river_f = player_river(df, lines, p)
        # Table VPIP
        t_vpip = sum(p in v for _, v, _ in past) / hands if hands else 0
        # 3bet Frequency
        t_3bet = sum(p in t for _, _, t in past) / hands if hands else 0
        # PnL
        pnl = compute_pnl(lines, p, all_p, df["big_blind"].iloc[0])
        rows.append({**base, "player_name": p, "reached_showdown": reached_sd, "preflop_raiser": agg_map.get(p, 0),
                     "table_vpip": t_vpip, "position": int(p == ip), "3bet_freq": t_3bet, "PnL": pnl,
                     **flop_f, **turn_f, **river_f})
    return pd.DataFrame(rows)


# Batch all hands
def process_hand(hand, all_hands_list):
    df = pd.DataFrame(index=[0])
    df = extract_basic(df, hand)
    df = extract_preflop(df, hand)
    df = extract_flop(df, hand)
    df = extract_turn(df, hand)
    df = extract_river(df, hand)
    return extract_players(df, hand, all_hands_list)


def loop_hands(all_hands_list):
    frames = [process_hand(h, all_hands_list) for h in all_hands_list]
    result = pd.concat(frames, ignore_index=True).sort_values(["hand_number", "player_name"]).reset_index(drop=True)
    result.to_csv("player_features.csv", index=False)
    print(result)
    return result


def loop_folder(folder_path):
    folder = Path(folder_path)
    txt_files = sorted(folder.rglob("*.txt"))
    FEATURES_DIR.mkdir(exist_ok=True)
    lock = mp.Lock()
    args = [(folder, f) for f in txt_files]
    with mp.Pool(mp.cpu_count(), initializer=set_error_lock, initargs=(lock,)) as pool:
        for _ in tqdm(pool.imap_unordered(process_one_file, args), total=len(args), desc="Processing txt"):
            pass
    return "Features"


if __name__ == "__main__":
    folder_path = "/Users/colinchan/Desktop/Trading 2025-26/Poker Algo/50:100 Linus Cash Game"
    player_features = loop_folder(folder_path)