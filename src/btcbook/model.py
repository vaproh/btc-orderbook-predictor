"""Load the model, predict, and convert a ten-second probability to a longer one."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .bars import BookBars
from .features import build_features, book_columns

HERE = Path(__file__).parent
DEFAULT_MODEL = HERE / "models" / "obi_book.txt"
DEFAULT_META = HERE / "models" / "obi_book_meta.json"

HORIZON = 10
WARMUP_BARS = 320
GATE_DEFAULT = 0.25


class ObiPredictor:
    """Loads the booster once; `predict` is cheap enough to run every second."""

    def __init__(self, model_path=None, meta_path=None, gate: float = GATE_DEFAULT):
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(model_path or DEFAULT_MODEL))
        meta = json.loads(Path(meta_path or DEFAULT_META).read_text())
        self.features: list[str] = meta["features"]
        self.meta = meta
        self.horizon = int(meta.get("horizon", HORIZON))
        self.auc = meta.get("auc")
        self.gates = meta.get("gates", {})
        self.gate = gate
        self.bars = BookBars()

    def on_tick(self, bid, ask, bid_qty, ask_qty, ts_ms=None) -> None:
        self.bars.on_tick(bid, ask, bid_qty, ask_qty,
                          ts_ms if ts_ms is not None else int(time.time() * 1000))

    def ready(self) -> bool:
        return len(self.bars) >= WARMUP_BARS

    def predict(self) -> dict | None:
        """{'p_up', 'lean', 'call', 'mid'} — None while warming up."""
        if not self.ready():
            return None
        d = self.bars.frame()
        if not len(d):
            return None
        X = build_features(d)
        if not all(c in X.columns for c in self.features):
            return None
        row = X[self.features].iloc[[-1]]
        arr = row.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            return None
        p = float(self.model.predict(row.to_numpy(dtype="float32"))[0])
        lean = abs(p - 0.5)
        return {"p_up": p, "lean": lean,
                "call": ("UP" if p >= 0.5 else "DOWN") if lean >= self.gate else None,
                "mid": float(d["mid"].iloc[-1])}

    def feature_frame(self):
        """Exposed for tests and for the parity check against training."""
        d = self.bars.frame()
        if not len(d):
            return None
        return build_features(d)[self.features]


# --------------------------------------------------------------------------
# turning a ten-second probability into a probability for a longer horizon
# --------------------------------------------------------------------------
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Acklam's rational approximation; accurate to ~1e-9 for p in (0, 1)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def extend_to(p10: float, seconds: float, horizon: float = HORIZON,
              half_life: float = 30.0) -> float:
    """P(up) over `seconds`, from P(up) over `horizon`, shrunk toward 0.5.

    Skill decays with horizon, and the decay is measured, not assumed: AUC runs
    0.77 at ten seconds, 0.60 at thirty, 0.55 at sixty, and 0.52 by two to three
    minutes. So confidence is halved roughly every `half_life` seconds beyond the
    training horizon, which is why this returns something *closer to 0.5* as
    `seconds` grows.

    It is deliberately not a drift extrapolation. Scaling an implied drift by
    time would make a directional call *more* confident at four minutes, and the
    measurement says the opposite happens.
    """
    if seconds <= horizon:
        return p10
    decay = 0.5 ** ((float(seconds) - horizon) / max(half_life, 1e-9))
    return 0.5 + (p10 - 0.5) * decay


def twap_probability(p10: float, now_price: float, strike: float,
                     secs_left: float, sigma_per_sec: float,
                     twap_observed: float | None = None,
                     twap_observed_seconds: float = 0.0) -> float | None:
    """P(closing 60-second TWAP ends above the strike).

    Resolution compares a 60-second average at the close against the window's
    opening price. With T seconds left, the part of that average already printed
    is fixed and only the remaining T seconds can move it.

    Untested against resolved outcomes. Treat it as an approximation until it has
    its own Brier score from real windows.
    """
    if now_price <= 0 or strike <= 0 or secs_left <= 0:
        return None

    T = min(float(secs_left), 60.0)
    obs = min(max(float(twap_observed_seconds), 0.0), T)
    known = max(T - obs, 0.0)

    sig10 = max(sigma_per_sec * math.sqrt(HORIZON), 1e-12)
    mu = sig10 * _phi_inv(p10) / HORIZON

    fut_mean = now_price * math.exp(mu * known / 2.0)
    if twap_observed is None or obs <= 0:
        expected = fut_mean
    else:
        expected = (obs * twap_observed + known * fut_mean) / (obs + known)

    # a Brownian average over `known` seconds has standard deviation
    # sigma * sqrt(T/3)
    sd = now_price * sigma_per_sec * math.sqrt(max(known, 0.0) / 3.0)
    if sd <= 0:
        return 1.0 if expected > strike else 0.0
    return _phi((expected - strike) / sd)
