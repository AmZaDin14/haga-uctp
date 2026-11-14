"""Pin the official ITC 2007 CB-CTT objective semantics.

Reference: cpsolver-itc2007 (Tomas Muller), CttPlacement#getRoomCapPenalty and
CttCurricula#getCompactPenalty. Room capacity is a SOFT penalty (students above
capacity, one point each); a lecture with no adjacent lecture in its
curriculum on the same day, including a lone lecture, is isolated (+2).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
c01 t01 {lectures} 1 {students}

ROOMS:
r01 5

CURRICULA:
cur01 1 c01

UNAVAILABILITY_CONSTRAINTS:
END.
"""


def _tiny(tmpdir: str, students: int, lectures: int = 1) -> str:
    path = Path(tmpdir) / "tiny.ctt"
    path.write_text(TINY.format(students=students, lectures=lectures))
    return str(path)


class OfficialObjectiveTests(unittest.TestCase):
    def test_capacity_excess_is_soft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_tiny(tmpdir, students=10))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            hcv, scv, fitness = problem.evaluate(chromosome)
            audit = validate_solution(instance, chromosome)

            self.assertEqual(hcv, 0, "capacity is not a hard constraint")
            self.assertEqual(scv, 5 + 2, "5 excess students + 2 isolated lecture")
            self.assertEqual(audit["scv_breakdown"]["room_capacity"], 5)
            self.assertEqual(fitness, scv)

    def test_single_lecture_day_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_tiny(tmpdir, students=5))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            hcv, scv, _ = problem.evaluate(chromosome)
            audit = validate_solution(instance, chromosome)

            self.assertEqual(hcv, 0)
            self.assertEqual(scv, 2, "a lone lecture on a day is isolated (+2)")
            self.assertEqual(audit["scv_breakdown"]["curriculum_compactness"], 2)

    def test_room_occupancy_is_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_tiny(tmpdir, students=5, lectures=2))
            problem = TimetablingProblem(instance)
            # Both lectures in the same room-slot: one room overlap and one
            # teacher overlap, both hard.
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            hcv, _, _ = problem.evaluate(chromosome)
            self.assertEqual(hcv, 3)


if __name__ == "__main__":
    unittest.main()


class SoftTargetingTests(unittest.TestCase):
    def test_soft_violation_lectures_flag_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = parse_ctt(_tiny(tmpdir, students=5))
            problem = TimetablingProblem(instance)
            chromosome = np.zeros(problem.total_lectures, dtype=np.int32)

            self.assertEqual(problem.get_conflicted_lectures(chromosome), [])
            self.assertEqual(problem.get_soft_violation_lectures(chromosome), [0])
