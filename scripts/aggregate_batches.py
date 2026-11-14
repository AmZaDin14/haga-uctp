#!/usr/bin/env python3
"""Aggregate every completed comparative batch into one 21-instance dataset.

Reads the per-batch ``analysis.json`` files produced by ``src.runner`` and
rebuilds the publication-level artefacts in ``results/final-21instances``:

* ``per_instance_algorithm_summary.csv`` - descriptive medians per instance
  and algorithm (the source of the manuscript's comparative table);
* ``friedman_nemenyi.json`` - Friedman omnibus test, mean ranks, and the
  Nemenyi critical difference for the full 21-instance suite (the manuscript
  previously reported the 15-instance subset).

The script is read-only with respect to the batches and refuses to write a
partial aggregate: every instance must be present with all four algorithms
and ``observed_runs == expected_runs``.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats import compute_nemenyi_cd, friedman_test  # noqa: E402

ALGORITHMS = ["Canonical_GA", "Discrete_PSO", "Simulated_Annealing", "HAGA"]
PAPER_SUBSET_15 = [
    "comp01", "comp02", "comp03", "comp04", "comp05", "comp06", "comp07",
    "comp08", "comp09", "comp11", "comp13", "comp14", "comp15", "comp18",
    "comp19",
]
COLUMNS = [
    "instance", "algorithm", "expected_runs", "observed_runs", "hcv_median",
    "fitness_median", "feasible_count", "feasibility_rate", "runtime_median",
]


def load_rows(batch_glob: str) -> pd.DataFrame:
    rows = []
    for analysis_path in sorted(ROOT.glob(batch_glob)):
        analysis = json.loads(analysis_path.read_text())
        if analysis.get("status") != "complete":
            raise SystemExit(f"{analysis_path} is not complete: {analysis.get('status')}")
        for record in analysis["descriptive"]:
            if record["status"] != "complete":
                raise SystemExit(
                    f"{analysis_path}: {record['instance']}/{record['algorithm']} "
                    f"is {record['status']}"
                )
            rows.append({column: record[column] for column in COLUMNS})
        print(f"loaded {analysis_path.parent.name}: {len(analysis['descriptive'])} records")
    if not rows:
        raise SystemExit("no batch analysis files found")
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame, expected_instances: int) -> None:
    instances = sorted(frame["instance"].unique())
    if len(instances) != expected_instances:
        raise SystemExit(f"expected {expected_instances} instances, found {len(instances)}: {instances}")
    if sorted(frame["algorithm"].unique()) != sorted(ALGORITHMS):
        raise SystemExit(f"unexpected algorithm set: {sorted(frame['algorithm'].unique())}")
    duplicates = frame.duplicated(["instance", "algorithm"]).sum()
    if duplicates:
        raise SystemExit(f"{duplicates} duplicated instance/algorithm rows")
    mismatched = frame[frame["observed_runs"] != frame["expected_runs"]]
    if not mismatched.empty:
        raise SystemExit(f"incomplete runs:\n{mismatched}")


def statistics(frame: pd.DataFrame, instances: list[str]) -> dict:
    pivot = (
        frame[frame["instance"].isin(instances)]
        .pivot(index="instance", columns="algorithm", values="fitness_median")
        .loc[sorted(instances), ALGORITHMS]
    )
    matrix = pivot.to_numpy(dtype=float)
    friedman = friedman_test(matrix, ALGORITHMS)
    ranks = pd.DataFrame(matrix, index=pivot.index, columns=ALGORITHMS).rank(axis=1)
    return {
        "instances": list(pivot.index),
        "n_instances": int(matrix.shape[0]),
        "k_algorithms": len(ALGORITHMS),
        "metric": "Median internal scalar best_fitness across 30 seeds (lower is better)",
        "friedman": {
            "statistic": round(friedman["statistic"], 4),
            "p_value": friedman["p_value"],
            "significant": friedman["significant"],
        },
        "mean_ranks": {name: round(value, 4) for name, value in friedman["mean_ranks"].items()},
        "nemenyi_cd": round(compute_nemenyi_cd(len(ALGORITHMS), matrix.shape[0]), 4),
        "per_instance_ranks": {
            instance: {name: float(ranks.loc[instance, name]) for name in ALGORITHMS}
            for instance in pivot.index
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-glob", default="results/batch*/analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final-21instances")
    parser.add_argument("--expected-instances", type=int, default=21)
    args = parser.parse_args()

    frame = load_rows(args.batch_glob)
    validate(frame, args.expected_instances)

    available = sorted(frame["instance"].unique())
    full = statistics(frame, available)
    paper = statistics(frame, [i for i in PAPER_SUBSET_15 if i in available])

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = frame.sort_values(["instance", "algorithm"])[COLUMNS]
    summary.to_csv(output_dir / "per_instance_algorithm_summary.csv", index=False)

    payload = {
        "schema_version": 1,
        "source": "results/batch*/analysis.json",
        "runs": int(frame["observed_runs"].sum()),
        "algorithms": ALGORITHMS,
        "full_suite": full,
        "manuscript_subset_15": paper,
    }
    (output_dir / "friedman_nemenyi.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"\nfull suite: N={full['n_instances']} chi2={full['friedman']['statistic']} "
        f"p={full['friedman']['p_value']:.3e} CD={full['nemenyi_cd']}"
    )
    print(
        f"15-subset : N={paper['n_instances']} chi2={paper['friedman']['statistic']} "
        f"p={paper['friedman']['p_value']:.3e} CD={paper['nemenyi_cd']}"
    )
    for name in ALGORITHMS:
        print(f"  {name:<20} full rank {full['mean_ranks'][name]:.2f} | 15 rank {paper['mean_ranks'][name]:.2f}")
    print(f"\nwritten: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
