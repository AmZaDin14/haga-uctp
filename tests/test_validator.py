"""Unit tests for independent solution validator and official schedule export."""
import tempfile
from typing import ClassVar

import unittest
from pathlib import Path

import numpy as np

from src.loader import parse_ctt
from src.problem import HARD_CONSTRAINT_WEIGHT, TimetablingProblem
from src.validator import export_solution, validate_solution


class ValidatorTests(unittest.TestCase):
    instance_path: ClassVar[Path] = Path("data/itc2007/comp01.ctt")

    def setUp(self):
        self.instance = parse_ctt(str(self.instance_path))
        self.problem = TimetablingProblem(self.instance)

    def test_validator_matches_problem_evaluator_exactly(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            chrom = rng.integers(0, self.problem.total_slots, size=self.problem.total_lectures)
            prob_hcv, prob_scv, prob_fit = self.problem.evaluate(chrom)
            audit = validate_solution(self.instance, chrom)

            self.assertEqual(audit["hcv"], prob_hcv)
            self.assertEqual(audit["scv"], prob_scv)
            self.assertEqual(audit["feasible"], bool(prob_hcv == 0))
            self.assertEqual(
                prob_fit,
                HARD_CONSTRAINT_WEIGHT * audit["hcv"] + audit["scv"]
            )

    def test_strict_feasibility_dominance(self):
        # A solution with HCV=1 must always have fitness > any feasible solution with HCV=0
        # even if the infeasible solution has SCV=0 and the feasible has large SCV
        infeasible_fitness = HARD_CONSTRAINT_WEIGHT * 1 + 0
        max_conceivable_scv = 50_000
        feasible_fitness = HARD_CONSTRAINT_WEIGHT * 0 + max_conceivable_scv
        self.assertLess(feasible_fitness, infeasible_fitness)

    def test_chromosome_length_validation(self):
        with self.assertRaises(ValueError):
            validate_solution(self.instance, [0] * (self.instance.total_lectures - 1))
        with self.assertRaises(ValueError):
            validate_solution(self.instance, [0] * (self.instance.total_lectures + 1))

    def test_export_solution_format(self):
        rng = np.random.default_rng(123)
        chrom = rng.integers(0, self.problem.total_slots, size=self.problem.total_lectures)

        with tempfile.TemporaryDirectory() as tmpdir:
            sol_path = Path(tmpdir) / "test_output.sol"
            export_solution(self.instance, chrom, sol_path)

            self.assertTrue(sol_path.is_file())
            lines = sol_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), self.instance.total_lectures)

            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 4)
                course_id, room_id, day, period = parts
                self.assertIn(course_id, self.instance.courses)
                self.assertIn(room_id, [r.id for r in self.instance.rooms])
                self.assertGreaterEqual(int(day), 0)
                self.assertLess(int(day), self.instance.num_days)
                self.assertGreaterEqual(int(period), 0)
                self.assertLess(int(period), self.instance.periods_per_day)


if __name__ == "__main__":
    unittest.main()
