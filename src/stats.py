"""
Statistical Testing Suite for UCTP Metaheuristic Evaluation.
Implements non-parametric statistical comparisons following Demšar (2006):
- Wilcoxon Signed-Rank Test (paired comparison)
- Friedman Test (omnibus ranking across multiple benchmark instances)
- Nemenyi Post-Hoc Test & Critical Difference (CD) computation
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


# Studentized range critical values q_alpha for the two-tailed Nemenyi test
# (Demšar 2006, Table 5), by significance level and number of algorithms.
NEMENYI_Q_ALPHA: dict[float, dict[int, float]] = {
    0.01: {2: 2.576, 3: 2.913, 4: 3.113, 5: 3.255, 6: 3.364, 7: 3.452, 8: 3.526, 9: 3.590, 10: 3.646},
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


def paired_wilcoxon_test(
    scores_baseline: list[float] | np.ndarray,
    scores_proposed: list[float] | np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute Wilcoxon signed-rank test on paired runs."""
    diff = np.array(scores_baseline) - np.array(scores_proposed)
    if bool(np.all(diff == 0)):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": 0.0,
        }

    res = stats.wilcoxon(scores_baseline, scores_proposed, alternative=alternative)
    stat = float(res[0])  # pyright: ignore[reportArgumentType]
    p_val = float(res[1])  # pyright: ignore[reportArgumentType]

    return {
        "statistic": stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
        "mean_diff": float(np.mean(diff)),
    }


def friedman_test(
    rank_matrix: np.ndarray,
    algorithm_names: list[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Execute Friedman test across instances.
    rank_matrix: shape (N_instances, k_algorithms).
    """
    n_instances = rank_matrix.shape[0]
    k_algos = rank_matrix.shape[1]
    assert len(algorithm_names) == k_algos

    algo_measurements = [rank_matrix[:, i] for i in range(k_algos)]
    res = stats.friedmanchisquare(*algo_measurements)

    ranks = np.zeros_like(rank_matrix, dtype=np.float64)
    for i in range(n_instances):
        ranks[i] = stats.rankdata(rank_matrix[i, :])

    mean_ranks = {
        algorithm_names[i]: float(np.mean(ranks[:, i]))
        for i in range(k_algos)
    }

    stat = float(res[0])
    p_val = float(res[1])

    return {
        "statistic": stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
        "mean_ranks": mean_ranks,
        "n_instances": n_instances,
        "k_algorithms": k_algos,
    }


def compute_nemenyi_cd(
    num_algorithms: int,
    num_instances: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute Critical Difference (CD) for Nemenyi test:
    CD = q_alpha * sqrt(k * (k + 1) / (6 * N))

    The critical value depends on both alpha and k; both must be tabulated
    (Demšar's q table covers k = 2..10 and alpha in {0.01, 0.05, 0.10}).
    """
    try:
        table = NEMENYI_Q_ALPHA[alpha]
    except KeyError as error:
        supported = sorted(NEMENYI_Q_ALPHA)
        raise ValueError(f"alpha must be one of {supported}, got {alpha!r}") from error
    try:
        q = table[num_algorithms]
    except KeyError as error:
        raise ValueError(
            f"num_algorithms must be in 2..10 for the Nemenyi q table, got {num_algorithms}"
        ) from error

    k = num_algorithms
    n = num_instances
    return float(q * math.sqrt((k * (k + 1)) / (6.0 * n)))


def compute_descriptive_summary(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
) -> pd.DataFrame:
    """Generate publication-ready descriptive metrics: Mean, Std, Median, IQR, Min, Max."""
    grouped = df.groupby(group_col)[metric_col]
    summary = pd.DataFrame({
        "Mean": grouped.mean(),
        "Std": grouped.std(),
        "Median": grouped.median(),
        "IQR": grouped.quantile(0.75) - grouped.quantile(0.25),
        "Min": grouped.min(),
        "Max": grouped.max(),
    })
    return summary
