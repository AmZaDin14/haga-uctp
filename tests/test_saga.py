# pyright: reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false
"""Unit tests for SAGA Reconstructed (Cooperative Coevolution + Fuzzy Control)."""
import unittest
from pathlib import Path
from typing import ClassVar, override

from src.loader import ProblemInstance, parse_ctt
from src.problem import HARD_CONSTRAINT_WEIGHT, TimetablingProblem
from src.saga import SAGA_Reconstructed
from src.validator import validate_solution


class SAGATests(unittest.TestCase):
    instance_path: ClassVar[Path] = Path("data/itc2007/comp01.ctt")
    instance: ProblemInstance
    problem: TimetablingProblem

    @override
    def setUp(self) -> None:
        self.instance = parse_ctt(str(self.instance_path))
        self.problem = TimetablingProblem(self.instance)
    def test_strict_evaluation_budget_compliance(self) -> None:
        """Verify SAGA terminates without exceeding the allocated evaluation budget."""
        budget = 150
        saga = SAGA_Reconstructed(
            self.problem,
            pop_size=10,
            max_evaluations=budget,
            random_seed=42,
        )
        res = saga.run()

        self.assertLessEqual(res["total_evaluations"], budget + 5)
        self.assertGreaterEqual(res["total_evaluations"], budget - 20)

    def test_chromosome_integrity_and_independent_validation(self) -> None:
        """Verify best_chromosome produced by SAGA is complete and matches validator exactly."""
        saga = SAGA_Reconstructed(
            self.problem,
            pop_size=10,
            max_evaluations=200,
            random_seed=123,
        )
        res = saga.run()

        chrom = res["best_chromosome"]
        self.assertEqual(len(chrom), self.instance.total_lectures)

        audit = validate_solution(self.instance, chrom)
        self.assertEqual(audit["hcv"], res["best_hcv"])
        self.assertEqual(audit["scv"], res["best_scv"])
        expected_fit = HARD_CONSTRAINT_WEIGHT * audit["hcv"] + audit["scv"]
        self.assertEqual(expected_fit, res["best_fitness"])

    def test_fuzzy_history_records(self) -> None:
        """Verify generation history captures dynamic Pc and Pm within bounds."""
        saga = SAGA_Reconstructed(
            self.problem,
            pop_size=10,
            max_evaluations=150,
            random_seed=999,
        )
        res = saga.run()
        history = res["history"]
        self.assertGreaterEqual(len(history), 1)

        for record in history:
            self.assertIn("pc", record)
            self.assertIn("pm", record)
            self.assertGreaterEqual(record["pc"], 0.40)
            self.assertLessEqual(record["pc"], 0.95)
            self.assertGreaterEqual(record["pm"], 0.10)
            self.assertLessEqual(record["pm"], 0.90)


if __name__ == "__main__":
    unittest.main()
