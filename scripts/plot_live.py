"""Regenerate docs/live_chart.png from a live run's ledger and bars.

Usage: uv run python scripts/plot_live.py <ledger.csv> [bars.parquet]
Reads the btcbook-live ledger (and optionally the bars parquet for a full
price line) and writes a dark two-panel chart: mid price on top, P(up) with
gated calls marked right/wrong below.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def style(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor("#0d1117")
    ax.set_xlabel(xlabel, color="#e6edf3", fontsize=10)
    ax.set_ylabel(ylabel, color="#e6edf3", fontsize=10)
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold",
                 pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#8b949e")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ledger_path = Path(sys.argv[1])
    bars_path = Path(sys.argv[2]) if len(sys.argv) > 3 else None
    out = ROOT / "docs" / "live_chart.png"

    rows = [r for r in csv.DictReader(open(ledger_path))
            if r.get("p_up") not in (None, "")]
    secs = [int(float(r["sec"])) for r in rows]
    t0 = secs[0]
    t = [(s - t0) / 60.0 for s in secs]
    p = [float(r["p_up"]) for r in rows]
    calls = [((float(r["sec"]) - t0) / 60.0, r.get("call") or "",
              r.get("outcome")) for r in rows]

    price_t, price = t, [float(r["mid"]) for r in rows]
    if bars_path is not None:
        import pandas as pd
        d = pd.read_parquet(bars_path)
        b0 = int(d["sec"].iloc[0])
        off = (secs[0] - b0) / 60.0
        kept = [(int(s) - b0) / 60.0 - off for s in d["sec"]]
        price_t = [x for x in kept if x >= -0.05]
        price = [float(x) for x, kt in zip(d["mid"], kept) if kt >= -0.05]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor("#0d1117")
    style(ax1, ylabel="BTC mid (USDT)",
          title="Live: mid price vs model P(up), one prediction per second")
    ax1.plot(price_t, price, color="white", lw=0.9)
    style(ax2, xlabel="minutes into run", ylabel="P(up)")
    ax2.plot(t, p, color="#58a6ff", lw=0.9)
    ax2.axhline(0.5, color="white", lw=0.8, alpha=0.6)
    ax2.axhspan(0.75, 1.0, color="#3fb950", alpha=0.12)
    ax2.axhspan(0.0, 0.25, color="#f85149", alpha=0.12)
    for ct, call, outcome in calls:
        if not call or outcome in (None, ""):
            continue
        ok = (call == "UP") == (str(outcome) == "1")
        ax2.scatter([ct], [0.97 if call == "UP" else 0.03],
                    c="#3fb950" if ok else "#f85149",
                    marker="^" if call == "UP" else "v", s=36, zorder=5)
    ax2.set_ylim(-0.03, 1.03)

    n_call = sum(1 for _, c, o in calls if c and o not in (None, ""))
    n_ok = sum(1 for _, c, o in calls
               if c and o not in (None, "")
               and (c == "UP") == (str(o) == "1"))
    if n_call:
        footer = f"gated calls: {n_ok}/{n_call} right"
    else:
        footer = "no gated calls this run (quiet tape, model stayed silent)"
    fig.text(0.99, 0.01, footer, color="#8b949e", ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
