"""What a live run is worth: gate sweep, and the cost arithmetic.

Accuracy is not money. This takes a ledger from `btcbook-live`, prices every
call against the move that actually followed, and compares the gross edge with
the round-trip cost of capturing it.

Run:  btcbook-eval --ledger live_ledger.csv --bars book_1s_live.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

BP = 1e4
HORIZON = 10


def gate_sweep(led: pd.DataFrame) -> None:
    led = led[led["call"].notna() & led["outcome"].notna()].copy()
    led["correct"] = ((led["call"] == "UP") & (led["outcome"] == 1)) | \
                     ((led["call"] == "DOWN") & (led["outcome"] == 0))
    print(f"resolved calls: {len(led):,}\n")
    print(f"  {'gate':<8}{'calls':>9}{'right':>8}{'wrong':>8}"
          f"{'accuracy':>10}{'95% CI':>18}")
    print("  " + "-" * 60)
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        m = led["lean"] >= thr
        if m.sum() < 30:
            continue
        acc = float(led.loc[m, "correct"].mean())
        se = np.sqrt(acc * (1 - acc) / m.sum())
        c = int(led.loc[m, "correct"].sum())
        print(f"  {thr:<8.2f}{int(m.sum()):>9,}{c:>8,}{int(m.sum())-c:>8,}"
              f"{100*acc:>9.1f}%   [{100*(acc-1.96*se):5.1f}, "
              f"{100*(acc+1.96*se):5.1f}]")


def economics(led: pd.DataFrame, bars_path: str) -> None:
    bars = pd.read_parquet(bars_path)[["sec", "mid"]].drop_duplicates("sec")
    mid_by_sec = dict(zip(bars["sec"].astype(int), bars["mid"].astype(float)))

    rows = []
    for r in led.itertuples():
        if not isinstance(r.call, str):
            continue
        p0 = mid_by_sec.get(int(r.sec))
        p1 = mid_by_sec.get(int(r.sec) + HORIZON)
        if not p0 or not p1 or p0 <= 0:
            continue
        side = 1.0 if r.call == "UP" else -1.0
        move = (p1 - p0) / p0
        rows.append({"lean": r.lean, "abs_bp": BP * abs(move),
                     "signed_bp": BP * side * move,
                     "correct": side * move > 0})
    t = pd.DataFrame(rows)
    if not len(t):
        print("\n  could not price any calls")
        return

    print(f"\n  priced {len(t):,} calls\n")
    print(f"  {'gate':<8}{'n':>8}{'hit':>8}{'med |move|':>13}"
          f"{'gross/trade':>14}")
    print("  " + "-" * 52)
    for thr in (0.10, 0.25, 0.35):
        b = t[t["lean"] >= thr]
        if len(b) < 30:
            continue
        print(f"  {thr:<8.2f}{len(b):>8,}{100*b['correct'].mean():>7.1f}%"
              f"{b['abs_bp'].median():>11.2f}bp{b['signed_bp'].mean():>+12.3f}bp")

    g = t[t["lean"] >= 0.25]
    if len(g) < 30:
        return
    gross = float(g["signed_bp"].mean())
    print(f"\n  {'route':<26}{'cost':>9}{'net':>12}")
    print("  " + "-" * 48)
    for name, cost in (("taker in + taker out", 10.0),
                       ("maker in + taker out", 7.0),
                       ("maker in + maker out", 4.0)):
        print(f"  {name:<26}{cost:>8.1f}bp{gross-cost:>+11.2f}bp")
    print(f"\n  gross edge per trade {gross:+.3f}bp against the cheapest")
    print(f"  realistic round trip of 4.0bp.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="live_ledger.csv")
    ap.add_argument("--bars", default=None)
    a = ap.parse_args()
    led = pd.read_csv(a.ledger)
    gate_sweep(led)
    if a.bars:
        economics(led, a.bars)


if __name__ == "__main__":
    main()
