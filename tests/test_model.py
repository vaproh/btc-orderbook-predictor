"""The checks that matter: features match, labels exclude flats, no lookahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcbook.bars import BookBars
from btcbook.features import build_features, book_columns
from btcbook.model import ObiPredictor, _phi, _phi_inv, extend_to


def synthetic_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A random walk with a book whose imbalance leads the next move."""
    rng = np.random.default_rng(seed)
    mid = 80000.0 * np.exp(np.cumsum(rng.normal(0, 2e-5, n)))
    future = np.r_[np.diff(mid), 0.0]
    # imbalance leans with the NEXT move, as a real book would
    obi = np.tanh(future / 1e-5) * 0.5 + rng.normal(0, 0.2, n)
    obi = np.clip(obi, -1, 1)
    bidq = 1.0 + obi
    askq = 1.0 - obi
    return pd.DataFrame({
        "sec": np.arange(n) + 1_700_000_000,
        "bid": mid - 0.5, "ask": mid + 0.5,
        "bidq": bidq, "askq": askq,
        "mid": mid, "micro": mid + (bidq - askq) * 0.1,
        "spread": 1.0,
        "obi": obi, "micro_dev": ((bidq - askq) * 0.1) / mid,
        "rel_spread": 1.0 / mid,
        "ofi": rng.normal(0, 0.3, n),
        "ofi_norm": rng.normal(0, 0.3, n),
        "n": rng.integers(50, 200, n),
    })


def test_book_columns_exclude_price_features():
    d = synthetic_bars()
    X = build_features(d)
    cols = book_columns(X.columns)
    assert len(X.columns) == 81
    assert len(cols) == 58
    assert "vol900" not in cols
    assert "r233" not in cols
    assert "obi" in cols
    assert "ofi_sum300" in cols


def test_features_are_causal():
    """Changing a future bar must not change an earlier row's features."""
    d = synthetic_bars(200)
    cols = book_columns(build_features(d).columns)
    a = build_features(d)[cols].iloc[150].to_numpy(float)

    d2 = d.copy()
    d2.loc[160:, ["mid", "obi", "ofi", "bidq", "askq"]] *= 1.5
    b = build_features(d2)[cols].iloc[150].to_numpy(float)

    np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True)


def test_bars_aggregate_into_seconds():
    b = BookBars()
    for i in range(250):                      # 250 ticks across 5 seconds
        b.on_tick(80000.0 + i * 0.1, 80000.5 + i * 0.1,
                  1.0, 1.0, ts_ms=1_700_000_000_000 + i * 20)
    assert 4 <= len(b) <= 6
    d = b.frame()
    assert list(d.columns[-4:]) == ["obi", "micro_dev", "rel_spread", "ofi_norm"]
    assert (d["n"] > 0).all()


def test_bars_reject_crossed_and_zero():
    b = BookBars()
    b.on_tick(0, 0, 0, 0, ts_ms=1_700_000_000_000)
    b.on_tick(80001.0, 80000.0, 1.0, 1.0, ts_ms=1_700_000_000_500)
    assert len(b) == 0


def test_flat_windows_are_excluded_from_labels():
    """A zero move is neither up nor down and must not be labelled DOWN."""
    mid = np.full(200, 80000.0)
    fwd = np.log(np.r_[mid[10:], np.full(10, np.nan)] / mid)
    moved = np.isfinite(fwd) & (np.nan_to_num(fwd, nan=0.0) != 0)
    assert not moved[:190].any()


def test_extend_to_is_identity_at_horizon():
    for p in (0.55, 0.62, 0.75, 0.9):
        assert extend_to(p, 10) == pytest.approx(p, abs=1e-6)


def test_extend_to_shrinks_toward_half_as_time_grows():
    """Skill decays with horizon, so confidence must move toward 0.5."""
    p = 0.75
    prev = p
    for t in (20, 60, 300, 900):
        cur = extend_to(p, t)
        assert cur < prev + 1e-12         # never more confident than before
        assert cur >= 0.5                 # and never past the midpoint
        prev = cur
    assert extend_to(0.75, 900) < 0.52
    # symmetric for a DOWN lean
    assert extend_to(0.25, 900) > 0.48


def test_phi_inverse_roundtrips():
    for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        assert _phi(_phi_inv(p)) == pytest.approx(p, abs=1e-8)


def test_predictor_returns_none_while_warming():
    p = ObiPredictor()
    p.on_tick(80000.0, 80001.0, 1.0, 1.0)
    assert p.predict() is None


def test_meta_matches_model():
    p = ObiPredictor()
    assert p.horizon == 10
    assert len(p.features) == 58
    assert p.auc is not None and 0.7 < p.auc < 0.85
