"""Live harness: predict, wait, check what actually happened.

Every row in the ledger is a real prediction made at a real time and scored
against the real outcome. The accuracy printed at the end is the accuracy of
those rows, including the ones that were wrong.

Run:  btcbook-live --seconds 3600 --gate 0.25
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import time
from pathlib import Path

import numpy as np

from .model import ObiPredictor

FUTURES_WS = "wss://fstream.binance.com/ws/btcusdt@bookTicker"


async def run(seconds: int, gate: float, ledger: str, bars_out: str | None,
              verbose: bool = False) -> None:
    import websockets

    pred = ObiPredictor(gate=gate)
    print(f"model: {len(pred.features)} features, holdout AUC {pred.auc}")
    print(f"gate {gate*100:.0f}%  warmup {pred.bars.__class__.__name__}  "
          f"running {seconds}s\n")

    pending: list[dict] = []
    rows: list[dict] = []
    started = None
    last_sec = None
    stop = False

    def handler(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    async with websockets.connect(FUTURES_WS, ping_interval=20) as ws:
        while not stop:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            m = json.loads(raw)
            try:
                pred.on_tick(float(m["b"]), float(m["a"]),
                             float(m["B"]), float(m["A"]),
                             int(m.get("E") or time.time() * 1000))
            except (KeyError, TypeError, ValueError):
                continue

            sec = pred.bars.sec
            if sec is None or sec == last_sec:
                continue
            last_sec = sec
            if started is None:
                started = sec
            if sec - started > seconds:
                break

            # settle anything whose horizon has elapsed
            keep = []
            for r in pending:
                if sec >= r["resolve_sec"]:
                    now_mid = pred.bars.cur["mid"] if pred.bars.cur else r["mid"]
                    move = now_mid - r["mid"]
                    r["outcome"] = None if move == 0 else int(move > 0)
                    rows.append(r)
                else:
                    keep.append(r)
            pending = keep

            if not pred.ready():
                continue
            out = pred.predict()
            if out is None:
                continue
            pending.append({"sec": sec, "resolve_sec": sec + pred.horizon,
                            "mid": out["mid"], "p_up": out["p_up"],
                            "lean": out["lean"], "call": out["call"],
                            "outcome": None})
            if verbose:
                print(f"  {time.strftime('%H:%M:%S')}  p_up {out['p_up']:.3f}  "
                      f"lean {out['lean']*100:4.1f}%  {out['call'] or '--'}")

    # ---- persist
    if bars_out and len(pred.bars):
        pred.bars.frame().to_parquet(bars_out, index=False)

    done = [r for r in rows if r["outcome"] is not None]
    if done:
        with open(ledger, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(done[0]))
            w.writeheader()
            w.writerows(done)

    print("\n" + "=" * 70)
    print("LIVE RESULTS")
    print("=" * 70)
    if not done:
        print("  no resolved predictions -- run longer or lower the gate")
        return

    correct = [r for r in done if r["call"]]
    right = sum(1 for r in correct
                if (r["call"] == "UP") == (r["outcome"] == 1))
    print(f"  predictions made     {len(rows):,}")
    print(f"  resolved (moved)     {len(done):,}")
    print(f"  windows that went up {100*np.mean([r['outcome'] for r in done]):.1f}%")
    if correct:
        acc = right / len(correct)
        print(f"\n  gated calls          {len(correct):,} "
              f"({100*len(correct)/len(done):.1f}% of resolved)")
        print(f"  right / wrong        {right:,} / {len(correct)-right:,}")
        print(f"  accuracy             {100*acc:.1f}%")
        se = np.sqrt(acc * (1 - acc) / len(correct))
        print(f"  95% CI               [{100*(acc-1.96*se):.1f}, "
              f"{100*(acc+1.96*se):.1f}]")

    # non-overlapping: consecutive predictions share 9/10 of their label window
    stride = done[::pred.horizon]
    s = [r for r in stride if r["call"]]
    if len(s) >= 20:
        sr = sum(1 for r in s if (r["call"] == "UP") == (r["outcome"] == 1))
        print(f"\n  non-overlapping      {sr:,}/{len(s):,} = "
              f"{100*sr/len(s):.1f}%")

    print(f"\n  {'confidence':<14}{'n':>8}{'accuracy':>10}")
    for lo, hi in ((0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 1.0)):
        blk = [r for r in done if r["call"] and lo <= r["lean"] < hi]
        if len(blk) < 20:
            continue
        b = sum(1 for r in blk if (r["call"] == "UP") == (r["outcome"] == 1))
        print(f"  {lo:.2f}-{hi:.2f}{'':<6}{len(blk):>8,}{100*b/len(blk):>9.1f}%")

    if done:
        print(f"\n  ledger -> {ledger}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--gate", type=float, default=0.25)
    ap.add_argument("--ledger", default="live_ledger.csv")
    ap.add_argument("--bars-out", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.seconds, a.gate, a.ledger, a.bars_out, a.verbose))


if __name__ == "__main__":
    main()
