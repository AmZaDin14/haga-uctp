#!/usr/bin/env python3
"""Render the Demšar Nemenyi critical-difference diagram for the HAGA suite.

Consumes ``results/final-21instances/friedman_nemenyi.json`` (written by
``scripts/aggregate_batches.py``) and writes both a vector PDF (for the IEEE
manuscript) and a PNG preview into ``drafts/figures``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "Simulated_Annealing": "#2ca02c",
    "HAGA": "#1f77b4",
    "Canonical_GA": "#ff7f0e",
    "Discrete_PSO": "#d62728",
}
LABELS = {
    "Simulated_Annealing": "Simulated Annealing",
    "HAGA": "HAGA (Proposed)",
    "Canonical_GA": "Canonical GA",
    "Discrete_PSO": "Discrete PSO",
}
# vertical offsets so neighbouring labels never collide
OFFSETS = {
    "Simulated_Annealing": 0.10,
    "HAGA": 0.22,
    "Canonical_GA": 0.13,
    "Discrete_PSO": 0.10,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "results" / "final-21instances" / "friedman_nemenyi.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "drafts" / "figures")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(args.stats.read_text())
    suite = payload["full_suite"]
    ranks: dict[str, float] = suite["mean_ranks"]
    cd: float = suite["nemenyi_cd"]
    n_instances: int = suite["n_instances"]
    k_algorithms: int = suite["k_algorithms"]

    ordered = sorted(ranks.items(), key=lambda item: item[1])
    lo = min(ranks.values()) - 0.35
    hi = max(ranks.values()) + 0.35

    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, 0.85)
    ax.axis("off")

    ax.set_title(
        f"Nemenyi Critical Difference (CD) Diagram "
        f"($k = {k_algorithms}$, $N = {n_instances}$, $\\alpha = {args.alpha}$)",
        fontsize=15,
        fontweight="bold",
        pad=6,
    )

    axis_y = 0.0
    ax.add_line(Line2D([lo + 0.1, hi - 0.1], [axis_y, axis_y], color="black", lw=2.0, zorder=1))

    # CD scale bar
    bar_y = 0.62
    ax.add_line(Line2D([lo + 0.15, lo + 0.15 + cd], [bar_y, bar_y], color="black", lw=2.0))
    for x in (lo + 0.15, lo + 0.15 + cd):
        ax.add_line(Line2D([x, x], [bar_y - 0.035, bar_y + 0.035], color="black", lw=2.0))
    ax.text(
        lo + 0.15 + cd / 2,
        bar_y + 0.06,
        f"CD = {cd:.4f} ($\\alpha = {args.alpha}$)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    for name, rank in ordered:
        color = COLORS[name]
        ax.plot(rank, axis_y, "o", ms=11, color=color, zorder=3)
        ax.text(rank, axis_y - 0.13, f"{rank:.1f}", ha="center", va="top",
                fontsize=12, fontweight="bold")
        label_y = axis_y + OFFSETS[name]
        ax.text(rank, label_y + 0.04, LABELS[name], ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=color)
        ax.add_line(
            Line2D([rank, rank], [label_y - 0.02, axis_y + 0.02], color=color,
                   lw=1.2, ls=(0, (3, 3)), zorder=2)
        )

    # contiguous rank group that is not significantly different from the best
    best_rank = ordered[0][1]
    tie_high = best_rank
    for _, rank in ordered[1:]:
        if rank - best_rank <= cd:
            tie_high = rank
    underline_y = axis_y - 0.36
    ax.add_line(
        Line2D([best_rank - 0.12, tie_high + 0.12], [underline_y, underline_y],
               color="#1f2d8f", lw=4.0, solid_capstyle="butt")
    )
    ax.text(
        (best_rank + tie_high) / 2,
        underline_y - 0.06,
        f"No significant difference ($p > {args.alpha}$)",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#1f2d8f",
    )

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = args.output_dir / f"nemenyi_cd_plot.{suffix}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(f"wrote {target}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
