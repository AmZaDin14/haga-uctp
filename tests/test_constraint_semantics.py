"""Conformance tests for the official ITC 2007 CB-CTT constraint semantics.

These tests pin down the three modelling decisions that were wrong in the first
experiment: room capacity is HARD, hard conflicts are counted pairwise, and a
curriculum day with a single lecture contains one isolated lecture.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

from src.loader import parse_ctt
from src.problem import TimetablingProblem
from src.validator import validate_solution

TINY = """Name: tiny
Courses: 1
Rooms: 1
Days: 1
Periods_per_day: 2
Curricula: 1
Constraints: 0

COURSES:
c01 t01 1 1 {students}

ROOMS:
r01 5

CURRICULA:
cur01 1 c01

UNAVAILABILITY_CONSTRAINTS:
END.
"""


def _write_tiny(tmpdir: str, students: int) -> str:
    path = Path(tmpdir) / "tiny.ctt"
    path.write_text(TINY.format(students=students))
    return str(path)


class OfficialConstraintSemanticsTests(unittest.TestCase):
    """Capacity is hard, conflicts are pairwise, singleton days are isolated."""

    instance_path: ClassVar[Path] = Path("data/itc2007/comp01.ctt")

    def test_single_lecture_day_counts_as_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_write_tiny(tmpdir, students=5))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            hcv, scv, fitness = problem.evaluate(chromosome)
            audit = validate_solution(instance, chromosome)

            self.assertEqual(hcv, 0)
            self.assertEqual(scv, 2, "a lone lecture on a day is isolated (+2)")
            self.assertEqual(audit["scv_breakdown"]["curriculum_compactness"], 2)
            self.assertEqual(fitness, scv)

    def test_capacity_excess_is_hard_not_soft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_write_tiny(tmpdir, students=10))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            hcv, scv, fitness = problem.evaluate(chromosome)
            audit = validate_solution(instance, chromosome)

            self.assertEqual(hcv, 1, "10 students in a 5-seat room is one hard violation")
            self.assertEqual(audit["hcv_breakdown"]["room_capacity"], 1)
            self.assertNotIn("room_capacity", audit["scv_breakdown"])
            self.assertEqual(scv, 2, "only compactness remains in SCV")
            self.assertEqual(fitness, 100_000 + 2)

    def test_room_occupancy_counts_pairs(self) -> None:
        instance = parse_ctt(str(self.instance_path))
        problem = TimetablingProblem(instance)
        chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

        _, _, _ = problem.evaluate(chromosome)
        audit = validate_solution(instance, chromosome)
        n = problem.total_lectures

        self.assertEqual(
            audit["hcv_breakdown"]["room_occupancy"],
            n * (n - 1) // 2,
            "k lectures in one room-slot violate C(k,2) constraints",
        )

    def test_capacity_violations_are_in_conflict_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_write_tiny(tmpdir, students=10))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            conflicted = problem.get_conflicted_lectures(chromosome)

            self.assertEqual(conflicted, [0], "capacity violations must be targeted by ELS")

    def test_soft_targets_exclude_hard_only_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_write_tiny(tmpdir, students=5))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            self.assertEqual(problem.get_conflicted_lectures(chromosome), [])
            self.assertEqual(problem.get_soft_violation_lectures(chromosome), [0])


if __name__ == "__main__":
    unittest.main()
