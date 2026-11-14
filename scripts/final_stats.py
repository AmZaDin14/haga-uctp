#!/usr/bin/env python3
"""Final statistics for the HAGA benchmark, computed from stored chromosomes.

Ground truth is re-derived from each run's chromosome with the current
evaluator (capacity soft, singleton compactness). comp15 is a duplicate of
comp03 and is excluded from all inferential tests; it remains in descriptive
tables only when requested.

Outputs a Markdown report and a JSON payload.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from src.loader import parse_ctt  # noqa: E402
from src.problem import TimetablingProblem  # noqa: E402

ALGORITHMS = ["Canonical_GA", "Discrete_PSO", "Simulated_Annealing", "HAGA"]
DUPLICATE = "comp15"


def load_instances() -> dict[str, tuple]:
    instances = {}
    for n in range(1, 22):
        name = f"comp{n:02d}"
        inst = parse_ctt(str(ROOT / "data" / "itc2007" / f"{name}.ctt"))
        instances[name] = (inst, TimetablingProblem(inst))
    return instances


def collect(pattern: str, instances: dict) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(pattern)):
        record = json.loads(Path(path).read_text())
        summary = record["summary"]
        instance = summary["instance"]
        problem = instances[instance][1]
        chromosome = np.array(record["best_chromosome"], dtype=int)
        hcv, scv, fitness = problem.evaluate(chromosome)
        rows.append({
            "instance": instance,
            "algorithm": summary["algorithm"],
            "seed": summary["seed"],
            "hcv": hcv,
            "scv": scv,
            "fitness": fitness,
            "feasible": hcv == 0,
        })
    return pd.DataFrame(rows)


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        default=str(ROOT / "results" / "batch*/comp*/runs/*/*.json"),
        help="glob of stored run JSONs",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "final-21instances")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    args = parser.parse_args()

    instances = load_instances()
    data = collect(args.runs, instances)
    if data.empty:
        raise SystemExit("no runs found")
    algorithms = [a for a in args.algorithms if a in set(data.algorithm)]

    # Per-instance medians, all instances (descriptive).
    medians = data.groupby(["instance", "algorithm"]).agg(
        scv_median=("scv", "median"),
        hcv_median=("hcv", "median"),
        feasible=("feasible", "sum"),
        runs=("seed", "count"),
    ).reset_index()

    # Inferential set: 20 unique instances.
    unique = sorted(set(data.instance) - {DUPLICATE})
    pivot = medians[medians.instance.isin(unique)].pivot(
        index="instance", columns="algorithm", values="scv_median"
    ).loc[unique, algorithms]
    ranks = pivot.rank(axis=1)
    chi2_stat, p_value = friedmanchisquare(*[pivot[a].to_numpy() for a in algorithms])
    n, k = pivot.shape
    mean_ranks = ranks.mean().to_dict()

    # Iman-Davenport (guarded: denominator is zero when ranks are unanimous).
    denominator = n * (k - 1) - chi2_stat
    iman_davenport = None if abs(denominator) < 1e-12 else (n - 1) * chi2_stat / denominator

    # Nemenyi CD.
    q = {2: 1.960, 3: 2.343, 4: 2.569}
    cd = {alpha: q[k] * math.sqrt(k * (k + 1) / (6 * n)) for alpha in (0.05,)}

    # Pairwise rank differences.
    pairs = []
    for i, a in enumerate(algorithms):
        for b in algorithms[i + 1:]:
            diff = abs(mean_ranks[a] - mean_ranks[b])
            pairs.append({
                "a": a,
                "b": b,
                "rank_diff": round(diff, 3),
                "significant_05": bool(diff > cd[0.05]),
            })

    # Per-instance paired Wilcoxon (HAGA vs SA), Holm-corrected.
    wilcoxon_rows = []
    if "HAGA" in algorithms and "Simulated_Annealing" in algorithms:
        raw_p = []
        for instance in unique:
            block = data[(data.instance == instance) & data.algorithm.isin(["HAGA", "Simulated_Annealing"])]
            wide = block.pivot(index="seed", columns="algorithm", values="scv").dropna()
            if len(wide) < 5:
                continue
            try:
                stat, p = wilcoxon(wide["HAGA"], wide["Simulated_Annealing"], zero_method="wilcox")
            except ValueError:
                continue
            diff = float((wide["HAGA"] - wide["Simulated_Annealing"]).median())
            wilcoxon_rows.append({"instance": instance, "p": float(p), "median_diff": diff})
            raw_p.append(float(p))
        if raw_p:
            adjusted = holm(raw_p)
            for row, adj in zip(wilcoxon_rows, adjusted):
                row["p_holm"] = adj

    # Feasibility with Wilson intervals.
    feasibility = []
    for algorithm in algorithms:
        subset = data[data.algorithm == algorithm]
        x = int(subset.feasible.sum())
        total = len(subset)
        lo, hi = wilson(x, total)
        feasibility.append({
            "algorithm": algorithm,
            "feasible": x,
            "runs": total,
            "rate": x / total if total else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        })

    payload = {
        "runs": int(len(data)),
        "instances_in_inference": unique,
        "excluded_duplicate": DUPLICATE,
        "algorithms": algorithms,
        "friedman": {"chi2": float(chi2_stat), "p": float(p_value), "n": n, "k": k},
        "iman_davenport": iman_davenport,
        "mean_ranks": {a: float(mean_ranks[a]) for a in algorithms},
        "nemenyi_cd_05": cd[0.05],
        "pairwise": pairs,
        "wilcoxon_haga_vs_sa": wilcoxon_rows,
        "feasibility": feasibility,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stats_report.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# HAGA benchmark — final statistics",
        "",
        f"- Runs analysed: **{len(data)}**",
        f"- Inferential instances: **{len(unique)}** ({DUPLICATE} excluded as a duplicate of comp03)",
        "",
        "## Mean ranks (20 unique instances)",
        "",
        "| Algorithm | Mean rank |",
        "|---|---|",
    ]
    for a in sorted(algorithms, key=lambda x: mean_ranks[x]):
        lines.append(f"| {a} | {mean_ranks[a]:.3f} |")
    lines += [
        "",
        f"Friedman chi2 = {chi2_stat:.4f}, p = {p_value:.3e}, df = {k - 1}",
        f"Iman-Davenport F = {iman_davenport if iman_davenport is None else round(iman_davenport, 4)}",
        f"Nemenyi CD (alpha=0.05) = {cd[0.05]:.4f}",
        "",
        "## Pairwise rank differences",
        "",
        "| A | B | Δ rank | Significant (0.05) |",
        "|---|---|---|---|",
    ]
    for row in pairs:
        lines.append(f"| {row['a']} | {row['b']} | {row['rank_diff']:.3f} | {'yes' if row['significant_05'] else 'no'} |")
    lines += ["", "## Feasibility", "", "| Algorithm | Feasible | Runs | Rate | 95% Wilson |", "|---|---|---|---|---|"]
    for row in feasibility:
        lines.append(
            f"| {row['algorithm']} | {row['feasible']} | {row['runs']} | {100*row['rate']:.1f}% | "
            f"[{100*row['ci_low']:.1f}, {100*row['ci_high']:.1f}] |"
        )
    if wilcoxon_rows:
        lines += ["", "## HAGA vs SA, per-instance paired Wilcoxon (Holm)", "", "| Instance | Median Δ SCV | p | p_Holm |", "|---|---|---|---|"]
        for row in wilcoxon_rows:
            lines.append(
                f"| {row['instance']} | {row['median_diff']:.1f} | {row['p']:.3e} | {row.get('p_holm', float('nan')):.3f} |"
            )
    (args.output_dir / "stats_report.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
