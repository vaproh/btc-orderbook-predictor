"""Regenerate docs/gate_curve.png from models/obi_book_meta.json.

Usage: uv run python scripts/plot_gate_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def style(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor("#0d1117")
    ax.set_xlabel(xlabel, color="#e6edf3", fontsize=10)
    ax.set_ylabel(ylabel, color="#e6edf3", fontsize=10)
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#8b949e")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((ROOT / "src" / "btcbook" / "models" /
                       "obi_book_meta.json").read_text())
    gates = sorted(((float(k), v["share"] * 100, v["accuracy"] * 100)
                    for k, v in meta["gates"].items()))
    xs = [g for g, _, _ in gates]
    share = [s for _, s, _ in gates]
    acc = [a for _, _, a in gates]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.patch.set_facecolor("#0d1117")
    style(ax, xlabel="gate (min |p - 0.5|)",
          title="Confidence gate: accuracy vs coverage")
    ax.plot(xs, acc, "o-", color="#3fb950", lw=2.2, ms=7,
            label="accuracy %")
    ax.plot(xs, share, "s--", color="#58a6ff", lw=1.6, ms=6,
            label="share of moments %")
    for x, a, s in zip(xs, acc, share):
        ax.annotate(f"{a:.1f}%", (x, a), color="#3fb950", fontsize=8,
                    textcoords="offset points", xytext=(0, 9), ha="center")
        ax.annotate(f"{s:.1f}%", (x, s), color="#58a6ff", fontsize=8,
                    textcoords="offset points", xytext=(0, -14), ha="center")
    leg = ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#e6edf3")
    fig.text(0.01, 0.01,
             f"held-out n=37,509  |  AUC {meta.get('auc')}  |  "
             f"{len(meta.get('walkforward_aucs', []))}-fold walk-forward",
             color="#8b949e", fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "gate_curve.png"
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
