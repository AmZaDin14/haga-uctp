"""Instance-level benchmark exports; seeds are nested repetitions, not test units."""

from __future__ import annotations

import json
from itertools import product
from numbers import Integral
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stats import friedman_test, paired_wilcoxon_test


_SUMMARY_COLUMNS = [
    "instance", "algorithm", "seed", "best_hcv", "best_scv", "best_fitness",
    "feasible", "evaluations", "generations", "runtime_seconds",
]
_KEYS = ["instance", "algorithm", "seed"]
_DESCRIPTION_COLUMNS = [
    "instance", "algorithm", "status", "expected_runs", "observed_runs",
    "hcv_median", "hcv_iqr", "fitness_median", "fitness_iqr",
    "feasible_count", "feasibility_rate", "feasible_scv_count",
    "feasible_scv_median", "feasible_scv_iqr", "runtime_median",
]
_METRIC = "Internal scalar best_fitness (lower is better); not official benchmark SCV"


def _summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty and not len(summary.columns):
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)
    missing = set(_SUMMARY_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    frame = summary.copy()
    if frame[_KEYS].isna().any().any():
        raise ValueError("Summary contains missing instance, algorithm, or seed keys")
    if not all(isinstance(value, str) for value in frame["instance"]):
        raise ValueError("Instance identities must be strings")
    if not all(isinstance(value, str) for value in frame["algorithm"]):
        raise ValueError("Algorithm identities must be strings")
    if not all(isinstance(value, Integral) and not isinstance(value, bool)
               for value in frame["seed"]):
        raise ValueError("Seed identities must be integers")
    if frame.duplicated(_KEYS).any():
        raise ValueError("Duplicate (instance, algorithm, seed) summary rows")
    for column in ("best_hcv", "best_scv", "best_fitness", "runtime_seconds", "feasible"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not frame["feasible"].isin([0, 1]).all():
        raise ValueError("Feasible must be 0 or 1 for every observed run")
    return frame


def _finite(values: pd.Series) -> np.ndarray:
    numbers = values.to_numpy(dtype=float, na_value=np.nan)
    return numbers[np.isfinite(numbers)]


def _median_iqr(values: pd.Series) -> tuple[float | None, float | None]:
    finite = _finite(values)
    if not len(finite):
        return None, None
    lower, upper = np.quantile(finite, [0.25, 0.75])
    return float(np.median(finite)), float(upper - lower)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _plot_convergence(
    summary: pd.DataFrame, convergence: pd.DataFrame, output_dir: Path,
) -> None:
    algorithms = list(summary["algorithm"].unique())
    trajectories: dict[str, list[pd.Series]] = {}
    unavailable = []
    for algorithm in algorithms:
        curves = []
        for seed in summary.loc[summary["algorithm"] == algorithm, "seed"]:
            rows = convergence.loc[
                (convergence["algorithm"] == algorithm) & (convergence["seed"] == seed)
            ]
            evaluations = pd.to_numeric(rows["evaluations"], errors="raise")
            fitness = pd.to_numeric(rows["best_fitness"], errors="raise")
            finite = np.isfinite(evaluations.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
                fitness.to_numpy(dtype=float, na_value=np.nan)
            )
            curve = pd.Series(
                fitness.loc[finite].to_numpy(dtype=float),
                index=evaluations.loc[finite].to_numpy(dtype=float),
            ).groupby(level=0).min().sort_index().cummin()
            if curve.empty:
                unavailable.append(f"{algorithm}: missing seed history; median not plotted")
                break
            curves.append(curve)
        else:
            if curves:
                trajectories[algorithm] = curves

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    try:
        all_curves = [curve for curves in trajectories.values() for curve in curves]
        plotted_values = []
        if all_curves:
            start = max(float(curve.index.min()) for curve in all_curves)
            end = min(float(curve.index.max()) for curve in all_curves)
            if start <= end:
                budgets = np.unique(np.concatenate([
                    np.array([start, end]),
                    *[curve.index.to_numpy(dtype=float) for curve in all_curves],
                ]))
                budgets = budgets[(budgets >= start) & (budgets <= end)]
                for algorithm, curves in trajectories.items():
                    # No backfill or extrapolation: every seed contributes at every budget.
                    aligned = [curve.reindex(curve.index.union(pd.Index(budgets))).sort_index()
                               .ffill().reindex(budgets).to_numpy(dtype=float)
                               for curve in curves]
                    median = np.median(np.vstack(aligned), axis=0)
                    plotted_values.extend(median.tolist())
                    ax.step(budgets, median, where="post", linewidth=1.8,
                            marker="o" if len(budgets) == 1 else None,
                            label=f"{algorithm} (median, n={len(curves)})")
            else:
                unavailable.append("No common observed evaluation range across seeds")
        if not plotted_values:
            unavailable.append("No aligned convergence data available")
        elif min(plotted_values) > 0:
            ax.set_yscale("log")
        if unavailable:
            ax.text(0.02, 0.02, "\n".join(unavailable), transform=ax.transAxes,
                    fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.85})
        ax.set_title("Best-fitness convergence: aligned seed medians", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Objective function evaluations", fontsize=10)
        ax.set_ylabel("Internal scalar fitness (not official benchmark SCV)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        if ax.lines:
            ax.legend(frameon=True, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "convergence.png")
    finally:
        plt.close(fig)


def _plot_distributions(summary: pd.DataFrame, output_dir: Path) -> None:
    algorithms = list(summary["algorithm"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    try:
        for index, algorithm in enumerate(algorithms, start=1):
            rows = summary.loc[summary["algorithm"] == algorithm]
            feasible = rows.loc[rows["feasible"] == 1]
            for axis, values, absent in (
                (axes[0], _finite(rows["best_hcv"]), "No finite HCV"),
                (axes[1], _finite(feasible["best_scv"]),
                 "No feasible runs" if feasible.empty else "No finite feasible SCV"),
            ):
                if len(values):
                    axis.boxplot([values], positions=[index], widths=0.6, patch_artist=True)
                else:
                    axis.text(index, 0.5, absent, transform=axis.get_xaxis_transform(),
                              rotation=90, ha="center", va="center", fontsize=8)
        for axis in axes:
            axis.set_xticks(range(1, len(algorithms) + 1), labels=algorithms, rotation=15)
            axis.set_xlim(0.5, max(len(algorithms), 1) + 0.5)
            axis.grid(True, linestyle="--", alpha=0.6)
        axes[0].set_title("Hard-constraint violations (all runs)", fontsize=10,
                          fontweight="bold")
        axes[0].set_ylabel("HCV (feasible = 0)", fontsize=9)
        axes[1].set_title("Soft-constraint violations (feasible runs only)", fontsize=10,
                          fontweight="bold")
        axes[1].set_ylabel("SCV", fontsize=9)
        fig.suptitle("Solution-quality distributions", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "boxplot.png")
    finally:
        plt.close(fig)


def export_instance(
    summary: pd.DataFrame, convergence: pd.DataFrame, output_dir: Path,
) -> None:
    """Export one instance, with best-so-far histories aligned on common budgets.

    A curve is restricted to the common observed range of every included seed;
    missing seed histories are annotated instead of silently changing its cohort.
    """
    frame = _summary_frame(summary)
    histories = convergence.copy()
    history_columns = [*_KEYS, "evaluations", "best_fitness"]
    if histories.empty and not len(histories.columns):
        histories = pd.DataFrame(columns=history_columns)
    missing = set(history_columns) - set(histories.columns)
    if missing:
        raise ValueError(f"Convergence is missing columns: {sorted(missing)}")
    instances = set(frame["instance"]) | set(histories["instance"])
    if len(instances) > 1:
        raise ValueError("export_instance accepts only one instance")
    summary_keys = set(frame[_KEYS].itertuples(index=False, name=None))
    history_keys = set(histories[_KEYS].itertuples(index=False, name=None))
    if not history_keys <= summary_keys:
        raise ValueError("Convergence contains run identities absent from summary")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "experiment_summary.csv", index=False)
    histories.to_csv(output_dir / "experiment_convergence.csv", index=False)
    style = ("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available
             else "default")
    with plt.style.context(style):
        _plot_convergence(frame, histories, output_dir)
        _plot_distributions(frame, output_dir)


def _inference(matrix: pd.DataFrame) -> dict[str, Any]:
    values = matrix.to_numpy(dtype=float)
    algorithms = list(matrix.columns)
    result: dict[str, Any] = {"alpha": 0.05, "unit": "instance", "alternative": "two-sided"}
    if len(matrix) < 2:
        result["status"] = "unavailable: at least two instances are required"
        return result
    if len(algorithms) < 3:
        result["friedman"] = {"status": "unavailable: at least three algorithms are required"}
    elif bool(np.all(values == values[:, :1])):
        result["friedman"] = {
            "status": "all_tied", "statistic": 0.0, "p_value": 1.0,
            "significant": False,
            "mean_ranks": {name: (len(algorithms) + 1) / 2 for name in algorithms},
            "n_instances": len(matrix), "k_algorithms": len(algorithms),
        }
    else:
        result["friedman"] = {"status": "ok", **friedman_test(values, algorithms)}
    if "HAGA" not in algorithms:
        result["wilcoxon"] = {"status": "unavailable: HAGA is not among the algorithms"}
        result["status"] = "completed"
        return result
    comparisons = []
    proposed = matrix["HAGA"].to_numpy(dtype=float)
    for baseline in algorithms:
        if baseline == "HAGA":
            continue
        reference = matrix[baseline].to_numpy(dtype=float)
        difference = reference - proposed
        test = paired_wilcoxon_test(reference, proposed, alternative="two-sided")
        statistic, p_value = float(test["statistic"]), float(test["p_value"])
        wins = int(np.sum(difference > 0))
        losses = int(np.sum(difference < 0))
        comparisons.append({
            "proposed": "HAGA", "baseline": baseline, "statistic": statistic,
            "p_value_raw": p_value, "instance_wins": wins,
            "instance_ties": int(np.sum(difference == 0)), "instance_losses": losses,
            "direction": ("HAGA" if wins > losses else baseline if losses > wins else "tied"),
        })
    running = 0.0
    for rank, comparison in enumerate(sorted(comparisons, key=lambda item: item["p_value_raw"])):
        running = max(running, (len(comparisons) - rank) * comparison["p_value_raw"])
        comparison["p_value_holm"] = min(1.0, running)
        comparison["significant_holm"] = comparison["p_value_holm"] < 0.05
    result["wilcoxon"] = {"status": "ok", "correction": "Holm", "comparisons": comparisons}
    result["status"] = "completed"
    return result


def aggregate_results(
    summary: pd.DataFrame,
    instance_ids: list[str],
    algorithm_names: list[str],
    seeds: list[int],
    output_dir: Path,
    *,
    inferential: bool = False,
) -> None:
    """Describe the requested grid, optionally testing paired instance medians.

    Descriptive exports retain missing cells and enumerate missing runs. Inference
    requires the exact complete grid and finite fitness for every repetition.
    """
    for name, identities in (("instances", instance_ids), ("algorithms", algorithm_names),
                             ("seeds", seeds)):
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate expected {name}")
        if not identities:
            raise ValueError(f"Expected {name} must not be empty")
    if not all(isinstance(value, str) for value in [*instance_ids, *algorithm_names]):
        raise ValueError("Expected instance and algorithm identities must be strings")
    if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in seeds):
        raise ValueError("Expected seed identities must be integers")
    frame = _summary_frame(summary)
    expected = list(product(instance_ids, algorithm_names, seeds))
    expected_set = set(expected)
    observed = set(frame[_KEYS].itertuples(index=False, name=None))
    unexpected = observed - expected_set
    if unexpected:
        raise ValueError(f"Unexpected run identities: {sorted(unexpected)}")
    missing = [dict(zip(_KEYS, key)) for key in expected if key not in observed]
    if inferential and missing:
        raise ValueError(f"Inference requires a complete grid: {len(missing)} runs are missing")
    if inferential and not np.isfinite(
        frame["best_fitness"].to_numpy(dtype=float, na_value=np.nan)
    ).all():
        raise ValueError("Inference requires finite best_fitness for every requested run")

    descriptions = []
    for instance, algorithm in product(instance_ids, algorithm_names):
        rows = frame.loc[(frame["instance"] == instance) & (frame["algorithm"] == algorithm)]
        feasible = rows.loc[rows["feasible"] == 1]
        hcv_median, hcv_iqr = _median_iqr(rows["best_hcv"])
        fitness_median, fitness_iqr = _median_iqr(rows["best_fitness"])
        scv_median, scv_iqr = _median_iqr(feasible["best_scv"])
        runtime_median, _ = _median_iqr(rows["runtime_seconds"])
        descriptions.append({
            "instance": instance, "algorithm": algorithm,
            "status": ("missing" if rows.empty else "complete" if len(rows) == len(seeds)
                       else "incomplete"),
            "expected_runs": len(seeds), "observed_runs": len(rows),
            "hcv_median": hcv_median, "hcv_iqr": hcv_iqr,
            "fitness_median": fitness_median, "fitness_iqr": fitness_iqr,
            "feasible_count": len(feasible),
            "feasibility_rate": len(feasible) / len(rows) if len(rows) else None,
            "feasible_scv_count": len(_finite(feasible["best_scv"])),
            "feasible_scv_median": scv_median, "feasible_scv_iqr": scv_iqr,
            "runtime_median": runtime_median,
        })
    descriptive = pd.DataFrame(descriptions, columns=_DESCRIPTION_COLUMNS)
    matrix = descriptive.pivot(index="instance", columns="algorithm", values="fitness_median")
    matrix = matrix.reindex(index=instance_ids, columns=algorithm_names)
    analysis: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "mode": "inferential" if inferential else "descriptive_only",
        "inferential": inferential, "metric": _METRIC,
        "statistical_unit": "instance", "seed_role": "nested repetition within instance",
        "completeness": {
            "status": "complete" if not missing else "incomplete",
            "expected_runs": len(expected), "observed_runs": len(observed),
            "missing_runs_count": len(missing), "missing_runs": missing,
            "missing_run_meaning": "No completed record: failed, pending, or unattempted",
            "expected_instances": len(instance_ids),
            "complete_instances": sum(
                all((instance, algorithm, seed) in observed
                    for algorithm, seed in product(algorithm_names, seeds))
                for instance in instance_ids
            ),
        },
        "descriptive": descriptions,
        "notes": [
            "Feasibility rates use observed runs; expected and observed counts expose missingness.",
            "SCV summaries use feasible runs only; missing/non-finite statistics are null.",
            "Fitness comparisons use each instance's median across its requested seeds.",
            "Failed attempts are not completion records and remain missing from this grid.",
        ],
    }
    if inferential:
        analysis["inference"] = _inference(matrix)
    else:
        analysis["notes"].append("Descriptive output only; no inferential claims or p-values.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "experiment_summary.csv", index=False)
    descriptive.to_csv(output_dir / "per_instance_algorithm_summary.csv", index=False)
    matrix.to_csv(output_dir / "instance_fitness_medians.csv", index_label="instance")
    (output_dir / "analysis.json").write_text(
        json.dumps(_json_safe(analysis), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
