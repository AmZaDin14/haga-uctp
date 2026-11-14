#!/usr/bin/env python3
"""Summarise a pilot run and apply the soft-ELS gate.

Gate: on each pilot instance, HAGA's median SCV among feasible runs must be
within 10% of SA's (or better). Falls back to median HCV when either algorithm
has too few feasible runs to compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "pilot-softels-101"
    frames = []
    for path in sorted(run_dir.glob("comp*/experiment_summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no summaries under {run_dir}")
    data = pd.concat(frames, ignore_index=True)

    rows = []
    for instance, group in data.groupby("instance"):
        for algorithm, arm in group.groupby("algorithm"):
            feasible = arm[arm["feasible"]]
            rows.append({
                "instance": instance,
                "algorithm": algorithm,
                "runs": len(arm),
                "feasible": int(arm["feasible"].sum()),
                "hcv_median": float(arm["best_hcv"].median()),
                "scv_median_feasible": float(feasible["best_scv"].median()) if len(feasible) else float("nan"),
                "scv_median_all": float(arm["best_scv"].median()),
                "runtime_median": float(arm["runtime_seconds"].median()),
            })
    summary = pd.DataFrame(rows).sort_values(["instance", "algorithm"])
    print(summary.to_string(index=False))

    print("\nGate (HAGA vs SA, median SCV among feasible runs):")
    passed = True
    for instance in sorted(data["instance"].unique()):
        block = summary[summary["instance"] == instance].set_index("algorithm")
        if {"HAGA", "Simulated_Annealing"} - set(block.index):
            continue
        haga = block.loc["HAGA", "scv_median_feasible"]
        sa = block.loc["Simulated_Annealing", "scv_median_feasible"]
        if pd.isna(haga) or pd.isna(sa):
            print(f"  {instance}: not comparable (feasible medians missing)")
            passed = False
            continue
        ratio = haga / sa
        ok = ratio <= 1.10
        passed &= ok
        print(f"  {instance}: HAGA {haga:.0f} vs SA {sa:.0f} -> ratio {ratio:.3f} {'PASS' if ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
