"""Constructive initialisation for CB-CTT.

Two phases, the standard decomposition for course timetabling:

1. DSATUR-style timeslot assignment: order lectures by saturation degree and
   place each in a timeslot that is free of teacher, curriculum and
   unavailability conflicts and whose room multiset can still house the
   enrolled students.
2. Room matching per timeslot: largest course first, smallest fitting room
   (optimal for this threshold-matching structure).

The construction is deterministic given a seed and is shared by every solver,
so no algorithm receives a privileged starting point. It is a pre-processing
step, not part of the evaluation budget.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.loader import ProblemInstance
from src.problem import TimetablingProblem


def _room_feasible(demands: list[int], capacities: list[int]) -> bool:
    """Can these course sizes be matched to distinct rooms by capacity?"""
    rooms = sorted(capacities)
    for demand in sorted(demands, reverse=True):
        fit = [room for room in rooms if room >= demand]
        if not fit:
            return False
        rooms.remove(fit[0])
    return True


def constructive_schedule(
    instance: ProblemInstance,
    problem: TimetablingProblem,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_lectures = problem.total_lectures
    num_timeslots = problem.total_timeslots
    num_rooms = problem.num_rooms
    periods_per_day = problem.periods_per_day

    idx_to_course = {idx: cid for cid, idx in instance.course_id_to_idx.items()}
    course_of = [int(problem.lecture_courses[i]) for i in range(num_lectures)]
    course_ids = [idx_to_course[c] for c in course_of]
    teachers = [instance.courses[cid].teacher for cid in course_ids]
    students = [instance.courses[cid].num_students for cid in course_ids]
    curricula = [problem.course_to_curricula[c] for c in course_of]

    unavailable: list[set[int]] = [set() for _ in range(num_lectures)]
    for cid, day, period in instance.unavailability:
        for i, lecture_course in enumerate(course_ids):
            if lecture_course == cid:
                unavailable[i].add(day * periods_per_day + period)

    capacities = sorted(problem.room_capacities)
    teacher_slots: dict[str, set[int]] = defaultdict(set)
    curriculum_slots: dict[int, set[int]] = defaultdict(set)
    slot_demands: dict[int, list[int]] = defaultdict(list)
    assignment = [-1] * num_lectures
    unassigned = set(range(num_lectures))

    def forbidden(i: int) -> set[int]:
        blocked = set(unavailable[i])
        blocked |= teacher_slots[teachers[i]]
        for curriculum in curricula[i]:
            blocked |= curriculum_slots[curriculum]
        return blocked

    while unassigned:
        best_lecture = None
        best_key: tuple[Any, ...] | None = None
        for i in unassigned:
            key = (
                len(forbidden(i)),
                -len(curricula[i]),
                -students[i],
                rng.random(),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_lecture = i
        i = int(best_lecture)
        blocked = forbidden(i)

        feasible = [
            t for t in range(num_timeslots)
            if t not in blocked
            and _room_feasible(slot_demands[t] + [students[i]], capacities)
        ]
        candidates = feasible or [t for t in range(num_timeslots) if t not in blocked]
        if not candidates:
            candidates = list(range(num_timeslots))

        timeslot = min(candidates, key=lambda t: (len(slot_demands[t]), rng.random()))
        assignment[i] = timeslot
        teacher_slots[teachers[i]].add(timeslot)
        for curriculum in curricula[i]:
            curriculum_slots[curriculum].add(timeslot)
        slot_demands[timeslot].append(students[i])
        unassigned.remove(i)

    chromosome = np.zeros(num_lectures, dtype=np.int64)
    by_timeslot: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(assignment):
        by_timeslot[t].append(i)

    for t, lectures in by_timeslot.items():
        free = sorted(range(num_rooms), key=lambda r: problem.room_capacities[r])
        for i in sorted(lectures, key=lambda lec: -students[lec]):
            fitting = [r for r in free if problem.room_capacities[r] >= students[i]]
            room = fitting[0] if fitting else (free[0] if free else 0)
            if room in free:
                free.remove(room)
            chromosome[i] = t * num_rooms + room

    return chromosome


def best_constructive_schedule(
    instance: ProblemInstance,
    problem: TimetablingProblem,
    seeds: int = 12,
    max_evaluations_ignored: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Restart the randomised construction and keep the best by (HCV, SCV)."""
    best: tuple[int, int, np.ndarray] | None = None
    for seed in range(seeds):
        chromosome = constructive_schedule(instance, problem, seed=seed)
        hcv, scv, _ = problem.evaluate(chromosome)
        if best is None or (hcv, scv) < (best[0], best[1]):
            best = (hcv, scv, chromosome)
        if hcv == 0:
            break
    assert best is not None
    return best


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import sys
    from src.loader import parse_ctt
    from src.validator import validate_solution

    name = sys.argv[1] if len(sys.argv) > 1 else "comp01"
    instance = parse_ctt(f"data/itc2007/{name}.ctt")
    problem = TimetablingProblem(instance)
    hcv, scv, chromosome = best_constructive_schedule(instance, problem)
    audit = validate_solution(instance, chromosome)
    print(f"{name}: HCV={hcv} SCV={scv} breakdown={audit['hcv_breakdown']}")
