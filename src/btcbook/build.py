"""Download Binance bookTicker archives and build one-second book bars.

The archive lives at
    https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/
    BTCUSDT-bookTicker-YYYY-MM-DD.zip

Each zip holds one day of best-bid/ask updates with sizes (~50-150 MB per day,
about 120 updates per second). Timestamps are in **milliseconds**.

Note: this archive stops in April 2024. Anything trained from it is trained on
old data and must be validated against live behaviour before its numbers are
quoted. To build bars from current data, run the live collector and reuse
`--from-parquet`.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE = ("https://data.binance.vision/data/futures/um/daily/bookTicker/"
        "BTCUSDT/BTCUSDT-bookTicker-{date}.zip")

COLS = ["update_id", "best_bid_price", "best_bid_qty",
        "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]


def fetch_day(date: str, dest: str | None = None) -> str:
    """Download one archive; returns the path."""
    path = dest or f"bt-{date}.zip"
    urllib.request.urlretrieve(BASE.format(date=date), path)
    return path


def _read_day(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    # some archives ship without a header row
    head = raw.split(b"\n", 1)[0]
    has_header = b"best_bid_price" in head or b"update_id" in head
    d = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                    names=None if has_header else COLS)
    d.columns = [c.strip().lower() for c in d.columns]
    return d


def build_bars(paths: list[str], out: str = "book_1s.parquet") -> pd.DataFrame:
    """Zip files -> one row per second of book state."""
    frames = []
    for p in paths:
        d = _read_day(p)
        bid = d["best_bid_price"].to_numpy(float)
        ask = d["best_ask_price"].to_numpy(float)
        bq = d["best_bid_qty"].to_numpy(float)
        aq = d["best_ask_qty"].to_numpy(float)
        # milliseconds, not nanoseconds
        ts = d["transaction_time"].to_numpy(np.int64)
        if ts[0] < 10**14:          # already ms
            sec = ts // 1000
        else:                        # microseconds
            sec = ts // 1_000_000

        mid = (bid + ask) / 2.0
        denom = bq + aq
        micro = np.where(denom > 0, (bid * aq + ask * bq) / np.maximum(denom, 1e-12), mid)

        # OFI increment between consecutive updates
        up_b = bid >= np.roll(bid, 1)
        dn_b = bid <= np.roll(bid, 1)
        up_a = ask <= np.roll(ask, 1)
        dn_a = ask >= np.roll(ask, 1)
        e_bid = np.where(up_b, bq, 0.0) - np.where(dn_b, np.roll(bq, 1), 0.0)
        e_ask = np.where(up_a, aq, 0.0) - np.where(dn_a, np.roll(aq, 1), 0.0)
        ofi = e_bid - e_ask
        ofi[0] = 0.0

        frames.append(pd.DataFrame({"sec": sec, "bid": bid, "ask": ask,
                                    "bidq": bq, "askq": aq, "mid": mid,
                                    "micro": micro, "spread": ask - bid,
                                    "ofi": ofi, "n": 1}))

    d = pd.concat(frames, ignore_index=True)
    g = d.groupby("sec", sort=True)
    bars = pd.DataFrame({
        "bid": g["bid"].last(), "ask": g["ask"].last(),
        "bidq": g["bidq"].last(), "askq": g["askq"].last(),
        "mid": g["mid"].last(), "micro": g["micro"].last(),
        "spread": g["spread"].last(),
        "ofi": g["ofi"].sum(), "n": g["n"].sum(),
    }).reset_index()

    tot = (bars["bidq"] + bars["askq"]).replace(0, np.nan)
    bars["obi"] = ((bars["bidq"] - bars["askq"]) / tot).fillna(0.0)
    bars["micro_dev"] = (bars["micro"] - bars["mid"]) / bars["mid"]
    bars["rel_spread"] = bars["spread"] / bars["mid"]
    bars["ofi_norm"] = (bars["ofi"] / tot).fillna(0.0)

    bars.to_parquet(out, index=False)
    return bars
