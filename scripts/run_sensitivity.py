# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportAny=false, reportDeprecated=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Hyperparameter Sensitivity Analysis Script for HAGA (RQ3).

Evaluates the stability and performance of HAGA under controlled variations of:
1. Slope sensitivity alpha in [0.10, 0.25, 0.40]
2. Slope sensitivity beta in [0.10, 0.20, 0.30]
3. Population size N in [30, 50, 100]

Enforces evaluation budget parity (B_max) and validates every solution with src/validator.py.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# Support direct execution
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.haga import HAGA
from src.loader import parse_ctt
from src.problem import TimetablingProblem, HARD_CONSTRAINT_WEIGHT
from src.validator import validate_solution


DEFAULT_INSTANCE = "data/itc2007/comp01.ctt"
DEFAULT_OUTPUT_DIR = "results/sensitivity"

CONFIGURATIONS = [
    # Baseline
    {"name": "Baseline (N=50, a=0.25, b=0.20)", "param": "nominal", "value": 0, "pop": 50, "alpha": 0.25, "beta": 0.20},
    # Alpha sweep
    {"name": "alpha=0.10 (low)", "param": "alpha", "value": 0.10, "pop": 50, "alpha": 0.10, "beta": 0.20},
    {"name": "alpha=0.40 (high)", "param": "alpha", "value": 0.40, "pop": 50, "alpha": 0.40, "beta": 0.20},
    # Beta sweep
    {"name": "beta=0.10 (low)", "param": "beta", "value": 0.10, "pop": 50, "alpha": 0.25, "beta": 0.10},
    {"name": "beta=0.30 (high)", "param": "beta", "value": 0.30, "pop": 50, "alpha": 0.25, "beta": 0.30},
    # Population size sweep
    {"name": "pop=30 (small)", "param": "pop_size", "value": 30, "pop": 30, "alpha": 0.25, "beta": 0.20},
    {"name": "pop=100 (large)", "param": "pop_size", "value": 100, "pop": 100, "alpha": 0.25, "beta": 0.20},
]


def _worker_sensitivity(task: tuple[str, int, float, float, int, int, str, str, Any]) -> dict[str, Any]:
    inst_path, pop_size, alpha, beta, max_evals, seed, config_name, param_name, param_val = task
    raw_inst = parse_ctt(inst_path)
    prob = TimetablingProblem(raw_inst)
    haga = HAGA(
        prob,
        pop_size=pop_size,
        alpha=alpha,
        beta=beta,
        max_evaluations=max_evals,
        random_seed=seed,
    )
    t0 = time.perf_counter()
    res = haga.run()
    elapsed = time.perf_counter() - t0

    best_chrom = res["best_chromosome"]
    audit = validate_solution(raw_inst, best_chrom)
    expected_fit = HARD_CONSTRAINT_WEIGHT * audit["hcv"] + audit["scv"]

    return {
        "config_name": str(config_name),
        "param": str(param_name),
        "param_value": param_val,
        "pop_size": int(pop_size),
        "alpha": float(alpha),
        "beta": float(beta),
        "seed": int(seed),
        "best_hcv": int(audit["hcv"]),
        "best_scv": int(audit["scv"]),
        "best_fitness": float(expected_fit),
        "feasible": bool(audit["feasible"]),
        "runtime_seconds": float(elapsed),
        "total_evaluations": int(res["total_evaluations"]),
        "generations": int(res["generations"]),
    }


def run_sensitivity(
    instance_path: str = DEFAULT_INSTANCE,
    evals: int = 50000,
    seeds: int = 10,
    workers: int = 10,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    smoke: bool = False,
) -> int:
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    actual_evals = 200 if smoke else evals
    actual_seeds = [101 + i for i in range(2 if smoke else seeds)]
    configs = CONFIGURATIONS[:3] if smoke else CONFIGURATIONS

    tasks: list[tuple[str, int, float, float, int, int, str, str, Any]] = []
    for c in configs:
        for s in actual_seeds:
            tasks.append((
                instance_path,
                int(c["pop"]),
                float(c["alpha"]),
                float(c["beta"]),
                actual_evals,
                s,
                str(c["name"]),
                str(c["param"]),
                c["value"],
            ))

    print(f"=== Running Hyperparameter Sensitivity Analysis (RQ3) ===")
    print(f"Instance: {instance_path}")
    print(f"Configurations ({len(configs)}): {[c['name'] for c in configs]}")
    print(f"Seeds ({len(actual_seeds)}): {actual_seeds}")
    print(f"Budget B_max: {actual_evals}")
    print(f"Total Runs: {len(tasks)}")

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        records = pool.map(_worker_sensitivity, tasks)

    total_time = time.perf_counter() - t_start
    print(f"All {len(records)} runs completed in {total_time:.2f} seconds.")

    df = pd.DataFrame(records)
    csv_file = out_path / "sensitivity_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved raw summary: {csv_file}")

    # Aggregated metrics
    agg = df.groupby("config_name").agg(
        feasible_count=("feasible", "sum"),
        total_runs=("feasible", "count"),
        hcv_median=("best_hcv", "median"),
        fitness_median=("best_fitness", "median"),
        fitness_iqr=("best_fitness", lambda x: float(np.percentile(x, 75) - np.percentile(x, 25))),
        runtime_median=("runtime_seconds", "median"),
    ).reset_index()
    agg["feasibility_rate"] = agg["feasible_count"] / agg["total_runs"]

    agg_file = out_path / "sensitivity_aggregated.csv"
    agg.to_csv(agg_file, index=False)
    print(f"Saved aggregated table: {agg_file}")
    print("\n=== Sensitivity Aggregated Results ===")
    print(agg.to_string())

    # Generate boxplot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        config_names = [str(c["name"]) for c in configs]
        data = [df[df["config_name"] == name]["best_fitness"].values for name in config_names]

        plt.boxplot(data, tick_labels=[name.replace(" ", "\n") for name in config_names])
        plt.title(f"HAGA Sensitivity Analysis - Best Fitness ({Path(instance_path).stem}, B_max={actual_evals})")
        plt.ylabel("Internal Fitness (Log Scale)")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot_file = out_path / "sensitivity_boxplot.png"
        plt.savefig(plot_file, dpi=200)
        plt.close()
        print(f"Saved boxplot: {plot_file}")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE, help="CTT instance path")
    parser.add_argument("--evals", type=int, default=50000, help="Budget per run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--workers", type=int, default=10, help="Parallel worker processes")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with 200 evals and 2 seeds")
    args = parser.parse_args()

    return run_sensitivity(
        instance_path=args.instance,
        evals=args.evals,
        seeds=args.seeds,
        workers=args.workers,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    raise SystemExit(main())
