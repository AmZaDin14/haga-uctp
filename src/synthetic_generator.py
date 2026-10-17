"""
Synthetic UCTP Instance Generator for Scalability and Stress Testing.
Generates controlled benchmark instances in standard ITC 2007 CTT format.
Supports variable course counts (E in {50, 100, 200}) and target room utilization.
"""

import math
import os
from typing import Dict, List, Optional
import numpy as np


def generate_synthetic_ctt(
    name: str,
    num_courses: int,
    target_utilization: float = 0.70,
    days: int = 5,
    periods_per_day: int = 8,
    min_lectures: int = 2,
    max_lectures: int = 4,
    random_seed: int = 42,
) -> str:
    """Generate a synthetic UCTP instance in standard CTT format."""
    rng = np.random.default_rng(random_seed)
    total_slots_per_room = days * periods_per_day

    # 1. Generate Courses
    # Each course has 2 to 4 lectures (typical academic weighting: 2, 3, or 4 SKS)
    lectures_per_course = rng.integers(min_lectures, max_lectures + 1, size=num_courses)
    total_lectures = int(np.sum(lectures_per_course))

    # 2. Compute Rooms needed for target utilization
    # Utilization = total_lectures / (num_rooms * total_slots_per_room)
    min_rooms_for_capacity = math.ceil(total_lectures / (target_utilization * total_slots_per_room))
    num_rooms = max(3, min_rooms_for_capacity)

    # 3. Generate Teachers (approx 1 teacher per 2-3 courses)
    num_teachers = max(5, int(num_courses / 2.2))
    course_teachers = [f"T{rng.integers(1, num_teachers + 1):03d}" for _ in range(num_courses)]

    # 4. Generate Students per course (20 to 100)
    course_students = rng.integers(20, 101, size=num_courses)

    # 5. Generate Room Capacities (25 to 120)
    room_capacities = rng.integers(30, 121, size=num_rooms)

    # 6. Generate Curricula (groups of courses that cannot overlap)
    # Typical university has 4 to 8 active cohorts/semesters
    num_curricula = max(4, int(num_courses / 8))
    curricula: Dict[str, List[str]] = {f"Curr{c+1:02d}": [] for c in range(num_curricula)}

    for i in range(num_courses):
        course_id = f"C{i+1:03d}"
        # Each course belongs to 1 primary curriculum, 20% probability of secondary
        curr_idx = rng.integers(0, num_curricula)
        curricula[f"Curr{curr_idx+1:02d}"].append(course_id)
        if rng.random() < 0.20:
            sec_idx = (curr_idx + rng.integers(1, num_curricula)) % num_curricula
            curricula[f"Curr{sec_idx+1:02d}"].append(course_id)

    # 7. Generate Unavailability Constraints (e.g., teacher unavailabilities)
    # Approx 5-10% of total slots have teacher unavailabilities
    unavailabilities = []
    for i in range(num_courses):
        course_id = f"C{i+1:03d}"
        # 2 random slots unavailable
        if rng.random() < 0.35:
            d = rng.integers(0, days)
            p = rng.integers(0, periods_per_day)
            unavailabilities.append((course_id, d, p))

    # Format into CTT string
    lines = [
        f"Name: {name}",
        f"Courses: {num_courses}",
        f"Rooms: {num_rooms}",
        f"Days: {days}",
        f"Periods_per_day: {periods_per_day}",
        f"Curricula: {num_curricula}",
        f"Constraints: {len(unavailabilities)}",
        "",
        "COURSES:",
    ]

    for i in range(num_courses):
        course_id = f"C{i+1:03d}"
        t_id = course_teachers[i]
        n_lec = lectures_per_course[i]
        min_days = min(days, max(1, n_lec - 1))
        num_stud = course_students[i]
        lines.append(f"{course_id} {t_id} {n_lec} {min_days} {num_stud}")

    lines.extend(["", "ROOMS:"])
    for r in range(num_rooms):
        room_id = f"R{r+1:02d}"
        cap = room_capacities[r]
        lines.append(f"{room_id} {cap}")

    lines.extend(["", "CURRICULA:"])
    for curr_id, c_list in curricula.items():
        if c_list:
            lines.append(f"{curr_id} {len(c_list)} " + " ".join(c_list))

    lines.extend(["", "UNAVAILABILITY_CONSTRAINTS:"])
    for course_id, d, p in unavailabilities:
        lines.append(f"{course_id} {d} {p}")

    lines.extend(["", "END."])
    return "\n".join(lines) + "\n"


def create_synthetic_benchmark_suite(output_dir: str = "data/synthetic") -> List[str]:
    """Generate the three proposal-specified synthetic instances: E=50, 100, 200."""
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []

    configs = [
        ("synthetic_50", 50, 42),
        ("synthetic_100", 100, 43),
        ("synthetic_200", 200, 44),
    ]

    for name, n_courses, seed in configs:
        content = generate_synthetic_ctt(name, n_courses, target_utilization=0.70, random_seed=seed)
        filepath = os.path.join(output_dir, f"{name}.ctt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        generated_paths.append(filepath)
        print(f"Generated {filepath} (Courses: {n_courses})")

    return generated_paths


if __name__ == "__main__":
    create_synthetic_benchmark_suite()
