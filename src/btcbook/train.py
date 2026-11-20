"""Train, validate, and export the model.

The validation here is the point of the repo. A model that scores well on one
chronological split has shown one observation; a model that survives four
sequential blocks, each trained only on data before it, has shown four.

Run:  btcbook-train --parquet book_1s.parquet --out models/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, book_columns

PARAMS = dict(num_leaves=63, min_child_samples=1000, n_estimators=500,
              learning_rate=0.03, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2,
              random_state=42, verbose=-1)

HORIZON = 10
FOLDS = 4


def prepare(parquet: str, horizon: int = HORIZON):
    """Bars -> (feature frame, feature matrix, labels, valid mask, n).

    Flat windows are excluded from the labels. A zero move is neither up nor
    down, and labelling it `return > 0` silently files it as DOWN -- on this
    data that is 20.7% of ten-second windows, which manufactures a fake 63/37
    drift and inflates any model that learns to detect quiet tape.
    """
    d = pd.read_parquet(parquet).sort_values("sec").reset_index(drop=True)
    mid = d["mid"].astype(float).to_numpy(float)
    X = build_features(d)
    fwd = np.log(np.r_[mid[horizon:], np.full(horizon, np.nan)] / mid)
    y = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)
    moved = np.isfinite(fwd) & (np.nan_to_num(fwd, nan=0.0) != 0)
    return d, X, y, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="book_1s.parquet")
    ap.add_argument("--out", default="models")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    a = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    d, X, y, moved = prepare(a.parquet, a.horizon)
    cols = book_columns(X.columns)
    Xb = X[cols].to_numpy(dtype="float32")
    n = len(d)
    ok = np.isfinite(Xb).all(axis=1) & np.isfinite(y) & moved

    split = int(n * 0.8)
    tr = np.flatnonzero((np.arange(n) < split - a.horizon) & ok)
    te = np.flatnonzero((np.arange(n) >= split) & ok)
    ytr, yt = y[tr].astype("int8"), y[te].astype("int8")
    print(f"bars {n:,}  features {len(cols)}  train {len(tr):,}  test {len(te):,}")

    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xb[tr], ytr)
    p = m.predict_proba(Xb[te])[:, 1]
    auc = float(roc_auc_score(yt, p))
    acc = float(((p >= 0.5) == (yt > 0.5)).mean())
    print(f"holdout  AUC {auc:.4f}  accuracy {100*acc:.1f}%")

    gates = {}
    print(f"\n  {'gate':<8}{'calls':>9}{'share':>8}{'accuracy':>10}")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        msk = np.abs(p - 0.5) >= thr
        if msk.sum() < 50:
            continue
        hit = float(((p[msk] >= 0.5) == (yt[msk] > 0.5)).mean())
        gates[str(round(thr, 2))] = {"accuracy": round(hit, 4),
                                     "share": round(float(msk.mean()), 4),
                                     "calls": int(msk.sum())}
        print(f"  {thr:<8}{int(msk.sum()):>9,}{100*msk.mean():>7.1f}%{100*hit:>9.1f}%")

    print(f"\n  walk-forward ({FOLDS} sequential blocks):")
    width = (n - split) // FOLDS
    aucs, g25 = [], []
    for f in range(FOLDS):
        s0, s1 = split + f * width, split + (f + 1) * width
        trf = np.flatnonzero((np.arange(n) < s0 - a.horizon) & ok)
        tef = np.flatnonzero((np.arange(n) >= s0) & (np.arange(n) < s1) & ok)
        if len(trf) < 20_000 or len(tef) < 200:
            continue
        mf = lgb.LGBMClassifier(**PARAMS)
        mf.fit(Xb[trf], y[trf].astype("int8"))
        pf = mf.predict_proba(Xb[tef])[:, 1]
        yf = y[tef].astype("int8")
        aa = float(roc_auc_score(yf, pf))
        ac = float(((pf >= 0.5) == (yf > 0.5)).mean())
        ms = np.abs(pf - 0.5) >= 0.25
        h = float(((pf[ms] >= 0.5) == (yf[ms] > 0.5)).mean()) if ms.sum() >= 50 else float("nan")
        aucs.append(aa)
        g25.append(h)
        print(f"    fold {f+1}  test {len(tef):>7,}  AUC {aa:.4f}  "
              f"acc {100*ac:.1f}%  gate25 {100*h:.1f}%")
    print(f"    mean AUC {np.mean(aucs):.4f}  min {np.min(aucs):.4f}  "
          f"folds>0.5 {sum(1 for x in aucs if x > 0.5)}/{len(aucs)}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(str(out / "obi_book.txt"))
    meta = {
        "features": cols, "horizon": a.horizon,
        "auc": round(auc, 4), "accuracy": round(acc, 4), "gates": gates,
        "walkforward_aucs": [round(x, 4) for x in aucs],
        "walkforward_gate25": [round(x, 4) for x in g25],
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_window": [int(d.sec.iloc[0]), int(d.sec.iloc[-1])],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "obi_book_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  wrote {out}/obi_book.txt and {out}/obi_book_meta.json")


if __name__ == "__main__":
    main()
