"""Consumer-visible statistical unit, missing-data and feasibility guarantees."""
import json
from pathlib import Path
import tempfile
import unittest
from typing import ClassVar

import pandas as pd

from src.benchmark_analysis import aggregate_results


class BenchmarkAnalysisTests(unittest.TestCase):
    instances: ClassVar[list[str]] = ["comp01", "comp07", "comp12"]
    algorithms: ClassVar[list[str]] = ["Canonical_GA", "Simulated_Annealing", "HAGA"]
    seeds: ClassVar[list[int]] = [101, 102, 103]

    def rows(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"instance": instance, "algorithm": algorithm, "seed": seed,
             "best_hcv": 0, "best_scv": value, "best_fitness": value,
             "feasible": 1, "evaluations": 200, "generations": 2,
             "runtime_seconds": 0.1}
            for instance in self.instances for algorithm in self.algorithms
            for seed, value in zip(self.seeds, [10, 20, 900])
        ])

    def aggregate(self, rows: pd.DataFrame, *, inferential: bool = False):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        aggregate_results(rows, self.instances, self.algorithms, self.seeds,
                          output, inferential=inferential)
        return json.loads((output / "analysis.json").read_text()), output

    def test_inference_uses_one_seed_median_per_instance_and_handles_ties(self):
        report, output = self.aggregate(self.rows(), inferential=True)
        matrix = pd.read_csv(output / "instance_fitness_medians.csv", index_col="instance")
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue((matrix == 20).all().all())
        friedman = report["inference"]["friedman"]
        self.assertEqual(friedman["n_instances"], 3)
        self.assertEqual(friedman["p_value"], 1.0)
        for comparison in report["inference"]["wilcoxon"]["comparisons"]:
            self.assertEqual(comparison["instance_ties"], 3)
            self.assertEqual(comparison["p_value_holm"], 1.0)
            self.assertFalse(comparison["significant_holm"])

    def test_infeasible_scv_does_not_enter_feasible_cost_summary(self):
        rows = self.rows()
        mask = (rows["instance"] == "comp01") & (rows["algorithm"] == "HAGA")
        indices = rows.index[mask]
        rows.loc[indices[0], ["best_hcv", "best_scv", "best_fitness", "feasible"]] = [1, 99999, 199999, 0]
        report, _ = self.aggregate(rows)
        cell = next(item for item in report["descriptive"]
                    if item["instance"] == "comp01" and item["algorithm"] == "HAGA")
        self.assertEqual(cell["feasible_count"], 2)
        self.assertEqual(cell["feasible_scv_median"], 460.0)
        self.assertAlmostEqual(cell["feasibility_rate"], 2 / 3)
        self.assertNotIn("inference", report)

    def test_missing_run_is_reported_and_cannot_be_silently_dropped_for_inference(self):
        rows = self.rows().iloc[:-1]
        report, _ = self.aggregate(rows)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["completeness"]["missing_runs"], [
            {"instance": "comp12", "algorithm": "HAGA", "seed": 103},
        ])
        with self.assertRaisesRegex(ValueError, "complete grid"):
            self.aggregate(rows, inferential=True)

    def test_duplicate_run_cannot_change_seed_weight(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.aggregate(pd.concat([rows, rows.iloc[:1]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
