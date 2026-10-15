"""
Preliminary Feasibility Study Experiment Runner.
Executes 10-seed comparison between Canonical GA and HAGA on ITC 2007 comp01.ctt.
Generates CSV summaries, statistical tests, and publication-ready charts.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import time
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.canonical_ga import CanonicalGA
from src.haga import HAGA
from src.loader import parse_ctt
from src.problem import TimetablingProblem


SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
DEFAULT_INSTANCE = "data/itc2007/comp01.ctt"
OUTPUT_DIR = "results"


def _worker_canonical_ga(args: Tuple[str, int, int, int]) -> Dict:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    ga = CanonicalGA(problem, pop_size=pop_size, max_evaluations=max_evals, random_seed=seed)
    res = ga.run()
    res["seed"] = seed
    res["pc_static"] = ga.pc
    res["pm_static"] = ga.pm
    return res


def _worker_haga(args: Tuple[str, int, int, int]) -> Dict:
    instance_path, pop_size, max_evals, seed = args
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    haga = HAGA(problem, pop_size=pop_size, max_evaluations=max_evals, random_seed=seed)
    res = haga.run()
    res["seed"] = seed
    return res


def run_experiment(
    instance_path: str = DEFAULT_INSTANCE,
    seeds: List[int] = SEEDS,
    pop_size: int = 50,
    max_evals: int = 25000,
    max_workers: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading problem instance: {instance_path}")
    instance = parse_ctt(instance_path)
    problem = TimetablingProblem(instance)
    print(f"Problem parsed: {instance.name} ({problem.total_lectures} lectures, {problem.total_slots} slots)")

    tasks_ga = [(instance_path, pop_size, max_evals, s) for s in seeds]
    tasks_haga = [(instance_path, pop_size, max_evals, s) for s in seeds]

    print(f"Executing {len(tasks_ga)} Canonical GA runs and {len(tasks_haga)} HAGA runs across {max_workers} parallel workers...")
    t0 = time.time()

    summary_records = []
    convergence_records = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Run Canonical GA
        print("Starting Canonical GA pool...")
        results_ga = list(executor.map(_worker_canonical_ga, tasks_ga))
        for res in results_ga:
            seed = res["seed"]
            summary_records.append({
                "algorithm": "Canonical_GA",
                "seed": seed,
                "best_hcv": res["best_hcv"],
                "best_scv": res["best_scv"],
                "best_fitness": res["best_fitness"],
                "feasible": int(res["best_hcv"] == 0),
                "evaluations": res["total_evaluations"],
                "generations": res["generations"],
                "runtime_seconds": res["runtime_seconds"],
            })
            for row in res["history"]:
                convergence_records.append({
                    "algorithm": "Canonical_GA",
                    "seed": seed,
                    "generation": row["generation"],
                    "evaluations": row["evaluations"],
                    "best_hcv": row["best_hcv"],
                    "best_scv": row["best_scv"],
                    "best_fitness": row["best_fitness"],
                    "mean_fitness": row["mean_fitness"],
                    "h_norm": np.nan,
                    "pc": res["pc_static"],
                    "pm": res["pm_static"],
                })

        # Run HAGA
        print("Starting HAGA pool...")
        results_haga = list(executor.map(_worker_haga, tasks_haga))
        for res in results_haga:
            seed = res["seed"]
            summary_records.append({
                "algorithm": "HAGA",
                "seed": seed,
                "best_hcv": res["best_hcv"],
                "best_scv": res["best_scv"],
                "best_fitness": res["best_fitness"],
                "feasible": int(res["best_hcv"] == 0),
                "evaluations": res["total_evaluations"],
                "generations": res["generations"],
                "runtime_seconds": res["runtime_seconds"],
            })
            for row in res["history"]:
                convergence_records.append({
                    "algorithm": "HAGA",
                    "seed": seed,
                    "generation": row["generation"],
                    "evaluations": row["evaluations"],
                    "best_hcv": row["best_hcv"],
                    "best_scv": row["best_scv"],
                    "best_fitness": row["best_fitness"],
                    "mean_fitness": row["mean_fitness"],
                    "h_norm": row.get("h_norm", np.nan),
                    "pc": row.get("pc", np.nan),
                    "pm": row.get("pm", np.nan),
                })

    total_time = time.time() - t0
    print(f"All 20 runs completed in {total_time:.1f} seconds!")

    df_summary = pd.DataFrame(summary_records)
    df_convergence = pd.DataFrame(convergence_records)

    summary_path = os.path.join(OUTPUT_DIR, "prapenelitian_summary.csv")
    convergence_path = os.path.join(OUTPUT_DIR, "prapenelitian_convergence.csv")

    df_summary.to_csv(summary_path, index=False)
    df_convergence.to_csv(convergence_path, index=False)
    print(f"Data saved to {summary_path} and {convergence_path}")

    return df_summary, df_convergence


def generate_statistics(df_summary: pd.DataFrame) -> Dict:
    """Compute descriptive and inferential statistics."""
    ga_data = df_summary[df_summary["algorithm"] == "Canonical_GA"]
    haga_data = df_summary[df_summary["algorithm"] == "HAGA"]

    stats_dict = {}
    for name, df in [("Canonical_GA", ga_data), ("HAGA", haga_data)]:
        stats_dict[name] = {
            "feasible_rate": float(df["feasible"].mean() * 100),
            "hcv_min": int(df["best_hcv"].min()),
            "hcv_median": float(df["best_hcv"].median()),
            "hcv_mean": float(df["best_hcv"].mean()),
            "hcv_std": float(df["best_hcv"].std()),
            "scv_min": int(df["best_scv"].min()),
            "scv_median": float(df["best_scv"].median()),
            "scv_iqr": float(df["best_scv"].quantile(0.75) - df["best_scv"].quantile(0.25)),
            "scv_mean": float(df["best_scv"].mean()),
            "scv_std": float(df["best_scv"].std()),
            "fit_median": float(df["best_fitness"].median()),
            "fit_iqr": float(df["best_fitness"].quantile(0.75) - df["best_fitness"].quantile(0.25)),
            "fit_mean": float(df["best_fitness"].mean()),
            "fit_std": float(df["best_fitness"].std()),
            "runtime_mean": float(df["runtime_seconds"].mean()),
            "runtime_std": float(df["runtime_seconds"].std()),
        }

    # Paired Wilcoxon Signed-Rank Test on best fitness
    wilc_fit = stats.wilcoxon(ga_data["best_fitness"], haga_data["best_fitness"], alternative="greater")
    # Paired Wilcoxon on SCV
    wilc_scv = stats.wilcoxon(ga_data["best_scv"], haga_data["best_scv"], alternative="greater")

    stats_dict["comparison"] = {
        "wilcoxon_fitness_stat": float(wilc_fit.statistic),
        "wilcoxon_fitness_pvalue": float(wilc_fit.pvalue),
        "wilcoxon_scv_stat": float(wilc_scv.statistic),
        "wilcoxon_scv_pvalue": float(wilc_scv.pvalue),
    }

    return stats_dict


def generate_plots(df_summary: pd.DataFrame, df_convergence: pd.DataFrame):
    """Generate professional scientific comparison plots."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Convergence Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for alg, color, label in [("Canonical_GA", "#D9534F", "Canonical GA"), ("HAGA", "#2E7D32", "HAGA (Proposed)")]:
        sub = df_convergence[df_convergence["algorithm"] == alg]
        grouped = sub.groupby("evaluations")["best_fitness"]
        evals = sorted(sub["evaluations"].unique())
        medians = [grouped.get_group(e).median() for e in evals if e in grouped.groups]
        q25 = [grouped.get_group(e).quantile(0.25) for e in evals if e in grouped.groups]
        q75 = [grouped.get_group(e).quantile(0.75) for e in evals if e in grouped.groups]

        ax.plot(evals, medians, label=f"{label} (Median)", color=color, linewidth=2.0)
        ax.fill_between(evals, q25, q75, color=color, alpha=0.18, label=f"{label} (IQR)")

    ax.set_title("Trayektori Konvergensi Evaluasi: Canonical GA vs HAGA (comp01.ctt)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Jumlah Evaluasi Fungsi Objektif ($B_{\\max} = 25.000$)", fontsize=10)
    ax.set_ylabel("Nilai Fitness Terbaik ($F(S) = 1000 \\cdot HCV + SCV$)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    conv_plot_path = os.path.join(OUTPUT_DIR, "convergence_comparison.png")
    fig.savefig(conv_plot_path)
    plt.close(fig)
    print(f"Convergence plot saved: {conv_plot_path}")

    # 2. Boxplot Comparison (Fitness / SCV)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    # HCV Boxplot
    hcv_data = [
        df_summary[df_summary["algorithm"] == "Canonical_GA"]["best_hcv"],
        df_summary[df_summary["algorithm"] == "HAGA"]["best_hcv"]
    ]
    axes[0].boxplot(hcv_data, tick_labels=["Canonical GA", "HAGA"], patch_artist=True,
                    boxprops=dict(facecolor="#D9534F", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2))
    axes[0].set_title("Distribusi Pelanggaran Hard Constraint (HCV)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("HCV (Solusi Layak = 0)", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # SCV Boxplot
    scv_data = [
        df_summary[df_summary["algorithm"] == "Canonical_GA"]["best_scv"],
        df_summary[df_summary["algorithm"] == "HAGA"]["best_scv"]
    ]
    axes[1].boxplot(scv_data, tick_labels=["Canonical GA", "HAGA"], patch_artist=True,
                    boxprops=dict(facecolor="#2E7D32", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2))
    axes[1].set_title("Distribusi Penalti Soft Constraint (SCV)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("SCV", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Perbandingan Sebaran Kualitas Solusi (10 Independent Seeds)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    box_plot_path = os.path.join(OUTPUT_DIR, "boxplot_comparison.png")
    fig.savefig(box_plot_path)
    plt.close(fig)
    print(f"Boxplot saved: {box_plot_path}")

    # 3. HAGA Diversity & Parameter Trace
    haga_conv = df_convergence[df_convergence["algorithm"] == "HAGA"]
    if not haga_conv.empty and "h_norm" in haga_conv.columns:
        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
        haga_grouped = haga_conv.groupby("generation")
        gens = sorted(haga_conv["generation"].unique())
        h_norm_mean = [haga_grouped.get_group(g)["h_norm"].mean() for g in gens]
        pc_mean = [haga_grouped.get_group(g)["pc"].mean() for g in gens]
        pm_mean = [haga_grouped.get_group(g)["pm"].mean() for g in gens]

        color1 = "#1F77B4"
        ax1.set_xlabel("Generasi Evolusi", fontsize=10)
        ax1.set_ylabel("Keragaman Populasi ($H_{norm}$)", color=color1, fontsize=10)
        ax1.plot(gens, h_norm_mean, color=color1, label="Normalized Hamming ($H_{norm}$)", linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 1.0)

        ax2 = ax1.twinx()
        color2 = "#FF7F0E"
        color3 = "#9467BD"
        ax2.set_ylabel("Probabilitas Genetika Dinamis ($P_c, P_m$)", fontsize=10)
        ax2.plot(gens, pc_mean, color=color2, linestyle="--", label="Crossover ($P_c$)", linewidth=1.8)
        ax2.plot(gens, pm_mean, color=color3, linestyle=":", label="Mutation ($P_m$)", linewidth=1.8)
        ax2.set_ylim(0, 1.0)

        plt.title("Dinamika Kontrol Adaptif HAGA: Keragaman Genotipik & Parameter", fontsize=11, fontweight="bold")
        fig.tight_layout()
        div_plot_path = os.path.join(OUTPUT_DIR, "haga_diversity_trace.png")
        fig.savefig(div_plot_path)
        plt.close(fig)
        print(f"Diversity trace saved: {div_plot_path}")


if __name__ == "__main__":
    df_sum, df_conv = run_experiment(pop_size=50, max_evals=25000)
    stats_res = generate_statistics(df_sum)
    print("\n--- STATISTICAL RESULTS ---")
    print(stats_res)
    generate_plots(df_sum, df_conv)
