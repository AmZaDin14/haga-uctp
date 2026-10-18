"""Independent solution validator and official ITC 2007 CB-CTT schedule exporter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.loader import ProblemInstance, parse_ctt


def decode_chromosome(instance: ProblemInstance, chromosome: list[int] | np.ndarray) -> tuple[list[int], list[int]]:
    """Decode 1D chromosome to parallel lists of timeslots and room indices."""
    chrom = np.asarray(chromosome, dtype=np.int32)
    timeslots = (chrom // instance.num_rooms).tolist()
    rooms = (chrom % instance.num_rooms).tolist()
    return timeslots, rooms


def validate_solution(
    instance: ProblemInstance,
    chromosome: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Independently audit all hard and soft constraint violations of a schedule.

    Returns full diagnostic breakdown:
    - Hard Constraint Violations (HCV):
      1. room_occupancy: at most one lecture per (room, timeslot)
      2. teacher_conflict: at most one lecture per teacher at any timeslot
      3. curriculum_conflict: at most one lecture per curriculum at any timeslot
      4. unavailability: lectures scheduled in prohibited course timeslots
    - Soft Constraint Violations (SCV):
      1. room_capacity: excess students over room capacity
      2. min_working_days: penalty of 5 per missing day below course threshold
      3. curriculum_compactness: penalty of 2 per isolated lecture
      4. room_stability: penalty of 1 per additional distinct room per course
    """
    chrom = np.asarray(chromosome, dtype=np.int32)
    if len(chrom) != instance.total_lectures:
        raise ValueError(
            f"Chromosome length {len(chrom)} does not match instance lectures {instance.total_lectures}"
        )
    if np.any(chrom < 0) or np.any(chrom >= instance.total_slots):
        raise ValueError(f"Gene values outside valid slot domain [0, {instance.total_slots})")

    timeslots, rooms = decode_chromosome(instance, chrom)
    idx_to_course = {idx: c_id for c_id, idx in instance.course_id_to_idx.items()}
    rooms_list = instance.rooms

    # ==============================================================
    # 1. HARD CONSTRAINTS
    # ==============================================================

    # 1.1 Room Occupancy: at most one lecture per (room, timeslot)
    room_slot_counts: dict[int, int] = {}
    for slot in chrom:
        slot_val = int(slot)
        room_slot_counts[slot_val] = room_slot_counts.get(slot_val, 0) + 1
    hcv_room = sum(count - 1 for count in room_slot_counts.values() if count > 1)

    # 1.2 Teacher Conflict: at most one lecture per teacher per timeslot
    teacher_slot_counts: dict[tuple[str, int], int] = {}
    for lec_idx, t_idx in enumerate(timeslots):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        teacher = instance.courses[c_id].teacher
        key = (teacher, t_idx)
        teacher_slot_counts[key] = teacher_slot_counts.get(key, 0) + 1
    hcv_teacher = sum(count - 1 for count in teacher_slot_counts.values() if count > 1)

    # 1.3 Curriculum Conflict: at most one lecture per curriculum per timeslot
    curr_slot_counts: dict[tuple[str, int], int] = {}
    for lec_idx, t_idx in enumerate(timeslots):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        for curr_id, courses in instance.curricula.items():
            if c_id in courses:
                key = (curr_id, t_idx)
                curr_slot_counts[key] = curr_slot_counts.get(key, 0) + 1
    hcv_curriculum = sum(count - 1 for count in curr_slot_counts.values() if count > 1)

    # 1.4 Unavailability: lecture scheduled in prohibited timeslot
    hcv_unavail = 0
    for lec_idx, t_idx in enumerate(timeslots):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        day = t_idx // instance.periods_per_day
        period = t_idx % instance.periods_per_day
        if (c_id, day, period) in instance.unavailability:
            hcv_unavail += 1

    hcv_total = hcv_room + hcv_teacher + hcv_curriculum + hcv_unavail

    # ==============================================================
    # 2. SOFT CONSTRAINTS
    # ==============================================================

    # 2.1 Room Capacity: students exceeding room capacity
    scv_capacity = 0
    for lec_idx, r_idx in enumerate(rooms):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        students = instance.courses[c_id].num_students
        capacity = rooms_list[r_idx].capacity
        if students > capacity:
            scv_capacity += (students - capacity)

    # 2.2 Minimum Working Days: penalty of 5 per day below min_working_days
    scv_min_days = 0
    course_days: dict[str, set[int]] = {c_id: set() for c_id in instance.courses}
    for lec_idx, t_idx in enumerate(timeslots):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        day = t_idx // instance.periods_per_day
        course_days[c_id].add(day)

    for c_id, days in course_days.items():
        min_days = instance.courses[c_id].min_working_days
        used_days = len(days)
        if used_days < min_days:
            scv_min_days += 5 * (min_days - used_days)

    # 2.3 Curriculum Compactness: penalty of 2 for isolated lectures
    scv_compactness = 0
    for curr_id, curr_course_ids in instance.curricula.items():
        curr_course_set = set(curr_course_ids)
        # Group unique periods by day for this curriculum
        curr_day_periods: list[set[int]] = [set() for _ in range(instance.num_days)]
        for lec_idx, t_idx in enumerate(timeslots):
            c_idx = instance.lecture_to_course[lec_idx]
            c_id = idx_to_course[c_idx]
            if c_id in curr_course_set:
                day = t_idx // instance.periods_per_day
                period = t_idx % instance.periods_per_day
                curr_day_periods[day].add(period)

        for day_periods in curr_day_periods:
            if len(day_periods) <= 1:
                continue
            for p in day_periods:
                if (p - 1) not in day_periods and (p + 1) not in day_periods:
                    scv_compactness += 2
    # 2.4 Room Stability: penalty of 1 per additional distinct room
    scv_room_stability = 0
    course_rooms: dict[str, set[int]] = {c_id: set() for c_id in instance.courses}
    for lec_idx, r_idx in enumerate(rooms):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        course_rooms[c_id].add(r_idx)


    for c_id, distinct_rooms in course_rooms.items():
        if len(distinct_rooms) > 1:
            scv_room_stability += len(distinct_rooms) - 1

    scv_total = scv_capacity + scv_min_days + scv_compactness + scv_room_stability

    return {
        "valid": True,
        "feasible": bool(hcv_total == 0),
        "hcv": int(hcv_total),
        "scv": int(scv_total),
        "hcv_breakdown": {
            "room_occupancy": int(hcv_room),
            "teacher_conflict": int(hcv_teacher),
            "curriculum_conflict": int(hcv_curriculum),
            "unavailability": int(hcv_unavail),
        },
        "scv_breakdown": {
            "room_capacity": int(scv_capacity),
            "min_working_days": int(scv_min_days),
            "curriculum_compactness": int(scv_compactness),
            "room_stability": int(scv_room_stability),
        },
    }


def export_solution(
    instance: ProblemInstance,
    chromosome: list[int] | np.ndarray,
    output_path: str | Path,
) -> None:
    """Export schedule to official ITC 2007 CB-CTT format (.sol).

    File syntax: Each line represents one lecture assignment:
    <CourseID> <RoomID> <Day> <Period>
    """
    timeslots, rooms = decode_chromosome(instance, chromosome)
    idx_to_course = {idx: c_id for c_id, idx in instance.course_id_to_idx.items()}
    rooms_list = instance.rooms

    lines: list[str] = []
    for lec_idx in range(instance.total_lectures):
        c_idx = instance.lecture_to_course[lec_idx]
        c_id = idx_to_course[c_idx]
        r_idx = rooms[lec_idx]
        room_id = rooms_list[r_idx].id
        t_idx = timeslots[lec_idx]
        day = t_idx // instance.periods_per_day
        period = t_idx % instance.periods_per_day
        lines.append(f"{c_id} {room_id} {day} {period}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_and_validate_run(
    instance_ctt_path: str | Path,
    run_json_path: str | Path,
    export_sol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load persisted run JSON, independently audit its best_chromosome, and optionally export .sol."""
    instance = parse_ctt(str(instance_ctt_path))
    with open(run_json_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    chromosome = run_data["best_chromosome"]
    audit = validate_solution(instance, chromosome)

    summary = run_data["summary"]
    matches_hcv = bool(audit["hcv"] == summary["best_hcv"])
    matches_scv = bool(audit["scv"] == summary["best_scv"])
    matches_feasible = bool(audit["feasible"] == bool(summary["feasible"]))

    if not (matches_hcv and matches_scv and matches_feasible):
        raise ValueError(
            f"Evaluator mismatch for {run_json_path}:\n"
            f"  Persisted: HCV={summary['best_hcv']}, SCV={summary['best_scv']}, Feasible={summary['feasible']}\n"
            f"  Audit    : HCV={audit['hcv']}, SCV={audit['scv']}, Feasible={audit['feasible']}"
        )

    if export_sol_path is not None:
        export_solution(instance, chromosome, export_sol_path)

    return {
        "instance": instance.name,
        "algorithm": run_data["algorithm"],
        "seed": run_data["seed"],
        "matches_persisted_summary": True,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ITC 2007 CB-CTT Schedule Validator & Exporter")
    parser.add_argument("--instance", required=True, help="Path to .ctt instance file")
    parser.add_argument("--run-json", required=True, help="Path to persisted run .json record")
    parser.add_argument("--export", help="Optional output path for official .sol file")
    args = parser.parse_args()

    result = load_and_validate_run(args.instance, args.run_json, args.export)
    audit = result["audit"]

    print("==================================================")
    print(f"SCHEDULE VALIDATION AUDIT: {result['instance']} ({result['algorithm']}, Seed {result['seed']})")
    print("==================================================")
    status_label = "FEASIBLE (Valid Schedule)" if audit["feasible"] else "INFEASIBLE (Violations Present)"
    print(f"Status     : {status_label}")
    print(f"Total HCV  : {audit['hcv']}")
    print(f"Total SCV  : {audit['scv']}")
    print("\n--- Hard Constraint Violations (HCV Breakdown) ---")
    for k, v in audit["hcv_breakdown"].items():
        print(f"  {k:25s}: {v}")
    print("\n--- Soft Constraint Violations (SCV Breakdown) ---")
    for k, v in audit["scv_breakdown"].items():
        print(f"  {k:25s}: {v}")

    if args.export:
        print(f"\nExported official schedule to: {args.export}")


if __name__ == "__main__":
    main()
