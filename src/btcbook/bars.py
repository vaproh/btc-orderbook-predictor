"""Aggregate bookTicker updates into one-second bars.

The model was trained on Binance's archived `bookTicker` files, where each row is
one best-bid/ask change with sizes. Live, the same fields arrive as a websocket
stream. This class turns the stream into the same shape:

  sec      the second the bar covers
  bid/ask  the last quote of that second
  bidq/askq its sizes
  mid      (bid+ask)/2
  micro    size-weighted mid; the deviation from mid is a signal
  ofi      Cont-Kukanov-Stoikov order flow imbalance, summed over the second
  n        updates in the second -- quote arrival intensity

Gaps are filled by carrying the previous bar forward, so the grid stays regular
and rolling windows count seconds rather than observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BookBars:
    def __init__(self, max_bars: int = 4000):
        self.bars: list[dict] = []
        self.sec: int | None = None
        self.cur: dict | None = None
        self.prev: tuple | None = None
        self.max_bars = max_bars

    def on_tick(self, bid: float, ask: float, bid_qty: float, ask_qty: float,
                ts_ms: int) -> None:
        """Fold one quote update into the current second."""
        if not (bid > 0 and ask > 0 and ask >= bid):
            return

        sec = ts_ms // 1000
        mid = (bid + ask) / 2.0
        denom = bid_qty + ask_qty
        micro = ((bid * ask_qty + ask * bid_qty) / denom) if denom > 0 else mid

        ofi = 0.0
        if self.prev is not None:
            pbid, pbidq, pask, paskq = self.prev
            # A bid that grows, or an ask that shrinks, is positive pressure.
            # A bid that shrinks, or an ask that grows, is negative.
            e_bid = (bid_qty if bid >= pbid else 0.0) - (pbidq if bid <= pbid else 0.0)
            e_ask = (ask_qty if ask <= pask else 0.0) - (paskq if ask >= pask else 0.0)
            ofi = e_bid - e_ask
        self.prev = (bid, bid_qty, ask, ask_qty)

        if self.sec is None:
            self.sec = sec
        if sec != self.sec:
            while self.sec < sec:
                self.bars.append(dict(self.cur) if self.cur else
                                 (dict(self.bars[-1]) if self.bars else None))
                if len(self.bars) > self.max_bars:
                    self.bars = self.bars[-self.max_bars:]
                self.sec += 1
            self.cur = None

        if self.cur is None:
            self.cur = {"sec": sec, "bid": bid, "ask": ask, "bidq": bid_qty,
                        "askq": ask_qty, "mid": mid, "micro": micro,
                        "spread": ask - bid, "ofi": ofi, "n": 1}
        else:
            self.cur.update({"bid": bid, "ask": ask, "bidq": bid_qty,
                             "askq": ask_qty, "mid": mid, "micro": micro,
                             "spread": ask - bid})
            self.cur["ofi"] += ofi
            self.cur["n"] += 1

    def __len__(self) -> int:
        return sum(1 for b in self.bars if b)

    def frame(self) -> pd.DataFrame:
        """The bars as a DataFrame with the derived columns the model needs."""
        rows = [b for b in self.bars if b]
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows).dropna(subset=["mid"])
        if not len(d):
            return pd.DataFrame()
        tot = (d["bidq"] + d["askq"]).replace(0, np.nan)
        d["obi"] = ((d["bidq"] - d["askq"]) / tot).fillna(0.0)
        d["micro_dev"] = (d["micro"] - d["mid"]) / d["mid"]
        d["rel_spread"] = d["spread"] / d["mid"]
        d["ofi_norm"] = (d["ofi"] / tot).fillna(0.0)
        return d.reset_index(drop=True)
