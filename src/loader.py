"""
ITC 2007 Curriculum-based Course Timetabling (CB-CTT) Parser.
Parses .ctt benchmark instances into structured problem representations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class Course:
    id: str
    teacher: str
    num_lectures: int
    min_working_days: int
    num_students: int


@dataclass
class Room:
    id: str
    capacity: int


@dataclass
class ProblemInstance:
    name: str
    num_courses: int
    num_rooms: int
    num_days: int
    periods_per_day: int
    num_curricula: int
    num_constraints: int
    
    courses: Dict[str, Course] = field(default_factory=dict)
    rooms: List[Room] = field(default_factory=list)
    curricula: Dict[str, List[str]] = field(default_factory=dict)
    unavailability: Set[Tuple[str, int, int]] = field(default_factory=set)  # (course_id, day, period)
    
    # Precomputed mappings
    room_id_to_idx: Dict[str, int] = field(default_factory=dict)
    course_id_to_idx: Dict[str, int] = field(default_factory=dict)
    lecture_to_course: List[int] = field(default_factory=list)  # lecture_idx -> course_idx
    total_lectures: int = 0
    total_timeslots: int = 0
    total_slots: int = 0

    def __post_init__(self):
        self.total_timeslots = self.num_days * self.periods_per_day
        self.total_slots = self.total_timeslots * self.num_rooms


def parse_ctt(filepath: str) -> ProblemInstance:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    headers = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.endswith(':'):
            break
        if ':' in line:
            parts = line.split(':', 1)
            headers[parts[0].strip()] = parts[1].strip()
        idx += 1

    instance = ProblemInstance(
        name=headers.get('Name', 'Unknown'),
        num_courses=int(headers.get('Courses', 0)),
        num_rooms=int(headers.get('Rooms', 0)),
        num_days=int(headers.get('Days', 0)),
        periods_per_day=int(headers.get('Periods_per_day', 0)),
        num_curricula=int(headers.get('Curricula', 0)),
        num_constraints=int(headers.get('Constraints', 0)),
    )

    current_section = None
    while idx < len(lines):
        line = lines[idx]
        if line == 'END.':
            break
        if line.endswith(':'):
            current_section = line[:-1]
            idx += 1
            continue

        parts = line.split()
        if not parts:
            idx += 1
            continue

        if current_section == 'COURSES':
            # Format: <CourseID> <TeacherID> <NumLectures> <MinWorkingDays> <NumStudents>
            course_id = parts[0]
            teacher_id = parts[1]
            num_lectures = int(parts[2])
            min_days = int(parts[3])
            num_students = int(parts[4])
            instance.courses[course_id] = Course(
                id=course_id,
                teacher=teacher_id,
                num_lectures=num_lectures,
                min_working_days=min_days,
                num_students=num_students
            )
        elif current_section == 'ROOMS':
            # Format: <RoomID> <Capacity>
            room_id = parts[0]
            capacity = int(parts[1])
            instance.rooms.append(Room(id=room_id, capacity=capacity))
        elif current_section == 'CURRICULA':
            # Format: <CurriculumID> <NumCourses> <CourseID_1> ...
            curr_id = parts[0]
            c_courses = parts[2:]
            instance.curricula[curr_id] = c_courses
        elif current_section == 'UNAVAILABILITY_CONSTRAINTS':
            # Format: <CourseID> <Day> <Period>
            course_id = parts[0]
            day = int(parts[1])
            period = int(parts[2])
            instance.unavailability.add((course_id, day, period))

        idx += 1

    # Build index mappings
    instance.room_id_to_idx = {r.id: i for i, r in enumerate(instance.rooms)}
    course_list = list(instance.courses.keys())
    instance.course_id_to_idx = {c: i for i, c in enumerate(course_list)}

    # Build flat lecture assignment list
    lecture_to_course = []
    for c_id, course in instance.courses.items():
        c_idx = instance.course_id_to_idx[c_id]
        for _ in range(course.num_lectures):
            lecture_to_course.append(c_idx)

    instance.lecture_to_course = lecture_to_course
    instance.total_lectures = len(lecture_to_course)
    instance.__post_init__()

    return instance
