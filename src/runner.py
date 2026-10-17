# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""
Unified Multiprocess Benchmark Runner for UCTP Optimization.
Supports:
1. Preliminary Feasibility Replication (Canonical GA vs HAGA)
2. Multi-Algorithm Comparative Benchmarking (Canonical GA, Discrete PSO, SA, HAGA)
3. 4-Stage Ablation Study (Canonical GA, GA+APC, GA+ELS, HAGA)
Enforces strict evaluation budget parity (B_max), multiprocess parallel execution,
and automated non-parametric statistical testing (Wilcoxon, Friedman, Nemenyi CD).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import time
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.canonical_ga import CanonicalGA
from src.haga import HAGA
from src.loader import parse_ctt
from src.problem import TimetablingProblem
from src.pso import DiscretePSO
from src.sa import SimulatedAnnealing
from src.stats import paired_wilcoxon_test, friedman_test, compute_descriptive_summary


DEFAULT_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
DEFAULT_INSTANCE = "data/itc2007/comp01.ctt"
OUTPUT_DIR = "results"


# ==============================================================
# Worker Functions for Multiprocessing
# ==============================================================

def _worker_canonical_ga(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    ga = CanonicalGA(problem, pop_size=pop_size, max_evaluations=max_evals, random_seed=seed)
    res = ga.run()
    res["seed"] = seed
    return res


def _worker_haga(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    haga = HAGA(problem, pop_size=pop_size, max_evaluations=max_evals, enable_apc=True, enable_els=True, random_seed=seed)
    res = haga.run()
    res["seed"] = seed
    return res


def _worker_pso(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    pso = DiscretePSO(problem, swarm_size=pop_size, max_evaluations=max_evals, random_seed=seed)
    res = pso.run()
    res["seed"] = seed
    return res


def _worker_sa(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, _pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    sa = SimulatedAnnealing(problem, max_evaluations=max_evals, random_seed=seed)
    res = sa.run()
    res["seed"] = seed
    return res


def _worker_ga_apc(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    haga = HAGA(problem, pop_size=pop_size, max_evaluations=max_evals, enable_apc=True, enable_els=False, random_seed=seed)
    res = haga.run()
    res["seed"] = seed
    return res


def _worker_ga_els(args: tuple[str, int, int, int]) -> dict[str, Any]:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    haga = HAGA(problem, pop_size=pop_size, max_evaluations=max_evals, enable_apc=False, enable_els=True, random_seed=seed)
    res = haga.run()
    res["seed"] = seed
    return res


# ==============================================================
# Experiment Orchestrator
# ==============================================================

def execute_suite(
    instance_path: str = DEFAULT_INSTANCE,
    algorithms: list[str] | None = None,
    seeds: list[int] | None = None,
    pop_size: int = 50,
    max_evals: int = 25000,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute parallel multi-algorithm or ablation experiments."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if algorithms is None:
        algorithms = ["Canonical_GA", "HAGA"]

    worker_map = {
        "Canonical_GA": _worker_canonical_ga,
        "HAGA": _worker_haga,
        "Discrete_PSO": _worker_pso,
        "Simulated_Annealing": _worker_sa,
        "GA_APC": _worker_ga_apc,
        "GA_ELS": _worker_ga_els,
    }

    print(f"=== Running Experiment Suite ===")
    print(f"Instance: {instance_path}")
    print(f"Algorithms: {algorithms}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Evaluation Quota B_max: {max_evals}")

    summary_records: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for algo in algorithms:
            worker = worker_map[algo]
            tasks = [(instance_path, pop_size, max_evals, s) for s in seeds]
            print(f"Executing {algo} across {len(tasks)} workers...")
            results = list(executor.map(worker, tasks))

            for res in results:
                s = int(res["seed"])
                summary_records.append({
                    "algorithm": res["algorithm"],
                    "seed": s,
                    "best_hcv": int(res["best_hcv"]),
                    "best_scv": int(res["best_scv"]),
                    "best_fitness": int(res["best_fitness"]),
                    "feasible": int(res["best_hcv"] == 0),
                    "evaluations": int(res["total_evaluations"]),
                    "generations": int(res["generations"]),
                    "runtime_seconds": float(res["runtime_seconds"]),
                })
                for row in res["history"]:
                    conv_item = {
                        "algorithm": res["algorithm"],
                        "seed": s,
                        "generation": int(row["generation"]),
                        "evaluations": int(row["evaluations"]),
                        "best_hcv": int(row["best_hcv"]),
                        "best_scv": int(row["best_scv"]),
                        "best_fitness": int(row["best_fitness"]),
                        "mean_fitness": float(row["mean_fitness"]),
                    }
                    if "h_norm" in row:
                        conv_item["h_norm"] = float(row["h_norm"])
                        conv_item["pc"] = float(row["pc"])
                        conv_item["pm"] = float(row["pm"])
                    convergence_records.append(conv_item)

    df_summary = pd.DataFrame(summary_records)
    df_convergence = pd.DataFrame(convergence_records)

    # Save CSV outputs
    sum_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    conv_path = os.path.join(OUTPUT_DIR, "experiment_convergence.csv")
    df_summary.to_csv(sum_path, index=False)
    df_convergence.to_csv(conv_path, index=False)
    print(f"Saved summary: {sum_path}")
    print(f"Saved convergence: {conv_path}")

    return df_summary, df_convergence


def analyze_results(df_summary: pd.DataFrame) -> None:
    """Perform descriptive analysis, Wilcoxon paired tests, and Friedman ranking."""
    print("\n" + "=" * 60)
    print("STATISTICAL BENCHMARK ANALYSIS")
    print("=" * 60)

    # 1. Descriptive Summary
    desc = compute_descriptive_summary(df_summary, "algorithm", "best_fitness")
    print("\n--- Descriptive Summary: Best Fitness ---")
    print(desc.to_string())

    # 2. Feasibility Rate
    feas_series = df_summary.groupby("algorithm")["feasible"].mean()
    print("\n--- Feasibility Rate (HCV == 0) ---")
    for alg, rate in feas_series.items():
        print(f"  {str(alg):20s}: {float(rate) * 100:5.1f}%")

    # 3. Pairwise Wilcoxon against HAGA
    algos = [str(x) for x in df_summary["algorithm"].unique() if x != "HAGA"]
    if "HAGA" in df_summary["algorithm"].values and algos:
        print("\n--- Pairwise Wilcoxon Signed-Rank Test (vs HAGA, alpha=0.05) ---")
        haga_scores = df_summary[df_summary["algorithm"] == "HAGA"].sort_values(by="seed")["best_fitness"].to_numpy()
        for other in algos:
            other_scores = df_summary[df_summary["algorithm"] == other].sort_values(by="seed")["best_fitness"].to_numpy()
            if len(haga_scores) == len(other_scores) and len(haga_scores) >= 6:
                w_res = paired_wilcoxon_test(other_scores, haga_scores, alternative="greater")
                sig_label = "SIGNIFICANT" if w_res["significant"] else "NOT SIGNIFICANT"
                print(f"  {other:20s} vs HAGA: W={w_res['statistic']:6.1f}, p={w_res['p_value']:.6f} ({sig_label})")

    # 4. Friedman Test if 3 or more algorithms present
    unique_algos = list(df_summary["algorithm"].unique())
    if len(unique_algos) >= 3:
        pivot = df_summary.pivot(index="seed", columns="algorithm", values="best_fitness").dropna()
        if len(pivot) >= 3:
            algo_cols = list(pivot.columns)
            matrix = pivot.to_numpy()
            f_res = friedman_test(matrix, algo_cols)
            print("\n--- Friedman Omnibus Test Across Seeds ---")
            print(f"  Chi-Square Stat: {f_res['statistic']:.4f}, p-value: {f_res['p_value']:.6e}")
            print(f"  Mean Ranks (Lower is Better):")
            for a, r in sorted(f_res["mean_ranks"].items(), key=lambda item: item[1]):
                print(f"    {a:20s}: {r:.2f}")


def generate_benchmark_plots(df_summary: pd.DataFrame, df_convergence: pd.DataFrame, filename_suffix: str = "") -> None:
    """Generate high-resolution scientific boxplots and convergence curves."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    unique_algos = list(df_summary["algorithm"].unique())

    # 1. Convergence trajectory
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for alg in unique_algos:
        sub = df_convergence[df_convergence["algorithm"] == alg]
        grouped = sub.groupby("evaluations")["best_fitness"]
        evals = sorted([int(x) for x in set(sub["evaluations"])])
        medians = [grouped.get_group(e).median() for e in evals if e in grouped.groups]
        ax.plot(evals, medians, label=f"{alg} (Median)", linewidth=1.8)

    ax.set_title("Trayektori Konvergensi Nilai Fitness Terbaik", fontsize=11, fontweight="bold")
    ax.set_xlabel("Jumlah Evaluasi Fungsi Objektif", fontsize=10)
    ax.set_ylabel("Fitness Terbaik ($F(S) = 1000 \\cdot HCV + SCV$)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    conv_path = os.path.join(OUTPUT_DIR, f"convergence_{filename_suffix}.png" if filename_suffix else "convergence.png")
    fig.savefig(conv_path)
    plt.close(fig)
    print(f"Saved convergence plot: {conv_path}")

    # 2. Boxplot distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    hcv_list = [df_summary[df_summary["algorithm"] == a]["best_hcv"] for a in unique_algos]
    axes[0].boxplot(hcv_list, tick_labels=unique_algos, patch_artist=True)
    axes[0].set_title("Distribusi Pelanggaran Kendala Kaku (HCV)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("HCV (Solusi Layak = 0)", fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    scv_list = [df_summary[df_summary["algorithm"] == a]["best_scv"] for a in unique_algos]
    axes[1].boxplot(scv_list, tick_labels=unique_algos, patch_artist=True)
    axes[1].set_title("Distribusi Penalti Kendala Lunak (SCV)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("SCV", fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Sebaran Kualitas Solusi Metaheuristik", fontsize=12, fontweight="bold")
    plt.tight_layout()
    box_path = os.path.join(OUTPUT_DIR, f"boxplot_{filename_suffix}.png" if filename_suffix else "boxplot.png")
    fig.savefig(box_path)
    plt.close(fig)
    print(f"Saved boxplot: {box_path}")


# ==============================================================
# CLI Entry Point
# ==============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="UCTP Metaheuristic Benchmark Runner")
    parser.add_argument("--mode", choices=["preliminary", "comparative", "ablation"], default="preliminary",
                        help="Benchmark mode: preliminary (GA vs HAGA), comparative (4 algorithms), or ablation (4 stages)")
    parser.add_argument("--instance", type=str, default=DEFAULT_INSTANCE, help="Path to CTT instance file")
    parser.add_argument("--evals", type=int, default=25000, help="Maximum evaluation budget B_max")
    parser.add_argument("--pop", type=int, default=50, help="Population / swarm size")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run (10 or 30)")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes")

    args = parser.parse_args()

    # Generate random seeds deterministically
    seed_list = [100 + i for i in range(1, args.seeds + 1)]

    if args.mode == "preliminary":
        algos = ["Canonical_GA", "HAGA"]
    elif args.mode == "comparative":
        algos = ["Canonical_GA", "Discrete_PSO", "Simulated_Annealing", "HAGA"]
    elif args.mode == "ablation":
        algos = ["Canonical_GA", "GA_APC", "GA_ELS", "HAGA"]
    else:
        algos = ["Canonical_GA", "HAGA"]

    df_sum, df_conv = execute_suite(
        instance_path=args.instance,
        algorithms=algos,
        seeds=seed_list,
        pop_size=args.pop,
        max_evals=args.evals,
        max_workers=args.workers,
    )

    analyze_results(df_sum)
    generate_benchmark_plots(df_sum, df_conv, filename_suffix=args.mode)


if __name__ == "__main__":
    main()
