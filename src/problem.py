"""
ITC 2007 Curriculum-based Course Timetabling (CB-CTT) Constraint Evaluator.
Computes Hard Constraint Violations (HCV), Soft Constraint Violations (SCV),
and composite fitness with budget tracking.
"""

from typing import Dict, List, Set, Tuple
import numpy as np

from src.loader import ProblemInstance

HARD_CONSTRAINT_WEIGHT: int = 100_000


class TimetablingProblem:
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.num_courses = instance.num_courses
        self.num_rooms = instance.num_rooms
        self.num_days = instance.num_days
        self.periods_per_day = instance.periods_per_day
        self.total_timeslots = instance.total_timeslots
        self.total_slots = instance.total_slots
        self.total_lectures = instance.total_lectures

        # Precompute lookup arrays
        self.lecture_courses = np.array(instance.lecture_to_course, dtype=np.int32)
        
        course_names = list(instance.courses.keys())
        self.course_students = np.zeros(self.num_courses, dtype=np.int32)
        self.course_min_days = np.zeros(self.num_courses, dtype=np.int32)
        self.course_teachers = np.zeros(self.num_courses, dtype=np.int32)
        
        teacher_names = list(set(c.teacher for c in instance.courses.values()))
        teacher_to_idx = {t: i for i, t in enumerate(teacher_names)}
        
        for c_id, course in instance.courses.items():
            idx = instance.course_id_to_idx[c_id]
            self.course_students[idx] = course.num_students
            self.course_min_days[idx] = course.min_working_days
            self.course_teachers[idx] = teacher_to_idx[course.teacher]

        self.room_capacities = np.array([r.capacity for r in instance.rooms], dtype=np.int32)

        # Unavailability table: [num_courses, total_timeslots]
        self.unavailable = np.zeros((self.num_courses, self.total_timeslots), dtype=bool)
        for c_id, day, period in instance.unavailability:
            c_idx = instance.course_id_to_idx[c_id]
            t_idx = day * self.periods_per_day + period
            if 0 <= t_idx < self.total_timeslots:
                self.unavailable[c_idx, t_idx] = True

        # Teacher conflict matrix: [num_courses, num_courses]
        self.teacher_conflict = np.zeros((self.num_courses, self.num_courses), dtype=bool)
        for i in range(self.num_courses):
            for j in range(self.num_courses):
                if i != j and self.course_teachers[i] == self.course_teachers[j]:
                    self.teacher_conflict[i, j] = True

        # Curriculum membership: [num_curricula, num_courses]
        self.num_curricula = len(instance.curricula)
        self.curriculum_courses: List[List[int]] = []
        self.course_to_curricula: List[List[int]] = [[] for _ in range(self.num_courses)]
        
        for curr_idx, (curr_id, c_list) in enumerate(instance.curricula.items()):
            c_indices = [instance.course_id_to_idx[c] for c in c_list if c in instance.course_id_to_idx]
            self.curriculum_courses.append(c_indices)
            for c_idx in c_indices:
                self.course_to_curricula[c_idx].append(curr_idx)

        # Curriculum conflict matrix: [num_courses, num_courses]
        self.curriculum_conflict = np.zeros((self.num_courses, self.num_courses), dtype=bool)
        for c_indices in self.curriculum_courses:
            for i in c_indices:
                for j in c_indices:
                    if i != j:
                        self.curriculum_conflict[i, j] = True

    def decode(self, chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 1D chromosome to (timeslots, rooms)."""
        timeslots = chromosome // self.num_rooms
        rooms = chromosome % self.num_rooms
        return timeslots, rooms

    def evaluate(self, chromosome: np.ndarray) -> Tuple[int, int, int]:
        """
        Evaluate chromosome and return (HCV, SCV, composite_fitness).
        composite_fitness = HARD_CONSTRAINT_WEIGHT * HCV + SCV (Strict Feasibility Dominance)
        """
        timeslots, rooms = self.decode(chromosome)
        
        # --- HARD CONSTRAINTS (HCV) ---
        hcv = 0

        # 1. Room occupancy: at most 1 lecture per (timeslot, room)
        slot_counts = {}
        for slot in chromosome:
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
        for count in slot_counts.values():
            if count > 1:
                hcv += (count - 1)

        # 2. Teacher conflict & 3. Curriculum conflict (grouped by timeslot)
        ts_to_courses: List[List[int]] = [[] for _ in range(self.total_timeslots)]
        for i, ts in enumerate(timeslots):
            c_idx = self.lecture_courses[i]
            ts_to_courses[ts].append(c_idx)

        for ts, c_list in enumerate(ts_to_courses):
            if len(c_list) <= 1:
                continue
            # Check teacher collisions in this timeslot
            seen_teachers = {}
            for c in c_list:
                t = self.course_teachers[c]
                seen_teachers[t] = seen_teachers.get(t, 0) + 1
            for count in seen_teachers.values():
                if count > 1:
                    hcv += (count - 1)

            # Check curriculum collisions in this timeslot
            for curr_indices in self.curriculum_courses:
                count = sum(1 for c in c_list if c in curr_indices)
                if count > 1:
                    hcv += (count - 1)

        # 4. Teacher/Course Unavailability
        for i, ts in enumerate(timeslots):
            c_idx = self.lecture_courses[i]
            if self.unavailable[c_idx, ts]:
                hcv += 1

        # --- SOFT CONSTRAINTS (SCV) ---
        scv = 0

        # 1. Room Capacity: students > capacity
        for i, r in enumerate(rooms):
            c_idx = self.lecture_courses[i]
            cap = self.room_capacities[r]
            students = self.course_students[c_idx]
            if students > cap:
                scv += (students - cap)

        # 2. Minimum Working Days
        # Course lectures grouped by course
        course_days: List[Set[int]] = [set() for _ in range(self.num_courses)]
        course_rooms: List[Set[int]] = [set() for _ in range(self.num_courses)]
        for i, ts in enumerate(timeslots):
            c_idx = self.lecture_courses[i]
            day = ts // self.periods_per_day
            course_days[c_idx].add(day)
            course_rooms[c_idx].add(rooms[i])

        for c_idx in range(self.num_courses):
            distinct_days = len(course_days[c_idx])
            min_days = self.course_min_days[c_idx]
            if distinct_days < min_days:
                scv += 5 * (min_days - distinct_days)

        # 3. Room Stability: distinct rooms used for course > 1
        for c_idx in range(self.num_courses):
            distinct_rooms = len(course_rooms[c_idx])
            if distinct_rooms > 1:
                scv += (distinct_rooms - 1)

        # 4. Curriculum Compactness: isolated lectures in a curriculum
        # Group lectures by curriculum, day -> set of periods
        for curr_idx, c_indices in enumerate(self.curriculum_courses):
            c_set = set(c_indices)
            curr_day_periods: List[Set[int]] = [set() for _ in range(self.num_days)]
            
            for i, ts in enumerate(timeslots):
                c_idx = self.lecture_courses[i]
                if c_idx in c_set:
                    day = ts // self.periods_per_day
                    period = ts % self.periods_per_day
                    curr_day_periods[day].add(period)

            for day_periods in curr_day_periods:
                if len(day_periods) <= 1:
                    continue
                for p in day_periods:
                    # Isolated if neither (p - 1) nor (p + 1) is in day_periods
                    if (p - 1) not in day_periods and (p + 1) not in day_periods:
                        scv += 2

        fitness = HARD_CONSTRAINT_WEIGHT * hcv + scv
        return hcv, scv, fitness

    def get_conflicted_lectures(self, chromosome: np.ndarray) -> List[int]:
        """Identify which lectures are involved in hard constraint violations."""
        timeslots, rooms = self.decode(chromosome)
        conflicted = set()

        # Room conflicts
        slot_map: Dict[int, List[int]] = {}
        for i, slot in enumerate(chromosome):
            slot_map.setdefault(slot, []).append(i)
        for lecs in slot_map.values():
            if len(lecs) > 1:
                conflicted.update(lecs)

        # Teacher & curriculum conflicts
        ts_map: Dict[int, List[int]] = {}
        for i, ts in enumerate(timeslots):
            ts_map.setdefault(ts, []).append(i)

        for ts, lecs in ts_map.items():
            if len(lecs) <= 1:
                continue
            # Teacher
            teacher_map: Dict[int, List[int]] = {}
            for lec in lecs:
                c = self.lecture_courses[lec]
                t = self.course_teachers[c]
                teacher_map.setdefault(t, []).append(lec)
            for t_lecs in teacher_map.values():
                if len(t_lecs) > 1:
                    conflicted.update(t_lecs)

            # Curriculum
            for curr_indices in self.curriculum_courses:
                curr_lecs = [lec for lec in lecs if self.lecture_courses[lec] in curr_indices]
                if len(curr_lecs) > 1:
                    conflicted.update(curr_lecs)

        # Unavailability
        for i, ts in enumerate(timeslots):
            c_idx = self.lecture_courses[i]
            if self.unavailable[c_idx, ts]:
                conflicted.add(i)

        return list(conflicted)
