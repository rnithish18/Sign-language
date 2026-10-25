"""
learning_content.py
--------------------
Static lesson definitions for Interactive Learning Mode. Each lesson is a
small curated sequence of gestures with a friendly description, used both
for hearing-impaired users practicing precision and for hearing users
learning sign language from scratch.

Keeping content data-driven (plain dict/list literals) makes it trivial to
extend: add a new lesson dict to LESSONS and it shows up in the UI.
"""

from typing import Dict, List

LESSONS: Dict[str, List[dict]] = {
    "ASL": [
        {
            "id": "asl_basics_1",
            "title": "Lesson 1: A-E",
            "description": "The first five letters of the ASL alphabet.",
            "targets": ["A", "B", "C", "D", "E"],
        },
        {
            "id": "asl_basics_2",
            "title": "Lesson 2: F-J",
            "description": "Continuing the alphabet.",
            "targets": ["F", "G", "H", "I", "J"],
        },
        {
            "id": "asl_basics_3",
            "title": "Lesson 3: K-O",
            "description": "Keep building muscle memory.",
            "targets": ["K", "L", "M", "N", "O"],
        },
        {
            "id": "asl_basics_4",
            "title": "Lesson 4: P-T",
            "description": "Almost through the alphabet.",
            "targets": ["P", "Q", "R", "S", "T"],
        },
        {
            "id": "asl_basics_5",
            "title": "Lesson 5: U-Z",
            "description": "Finish the ASL alphabet.",
            "targets": ["U", "V", "W", "X", "Y", "Z"],
        },
        {
            "id": "asl_controls",
            "title": "Lesson 6: Control Gestures",
            "description": "Hands-free editing: space, del/backspace, clear, speak.",
            "targets": ["SPACE", "DEL", "CLEAR", "SPEAK"],
        },
        {
            "id": "asl_word_hello",
            "title": "Lesson 7: Spell 'HELLO'",
            "description": "Put letters together into your first word.",
            "targets": ["H", "E", "L", "L", "O"],
        },
    ],
    "ISL": [
        {
            "id": "isl_basics_1",
            "title": "Lesson 1: A-E",
            "description": "The first five letters of the ISL alphabet.",
            "targets": ["A", "B", "C", "D", "E"],
        },
        {
            "id": "isl_basics_2",
            "title": "Lesson 2: F-J",
            "description": "Continuing the ISL alphabet (many ISL letters are two-handed).",
            "targets": ["F", "G", "H", "I", "J"],
        },
        {
            "id": "isl_controls",
            "title": "Lesson 3: Control Gestures",
            "description": "Hands-free editing: space, backspace, clear, speak.",
            "targets": ["SPACE", "BACKSPACE", "CLEAR", "SPEAK"],
        },
        {
            "id": "isl_word_hi",
            "title": "Lesson 4: Spell 'HI'",
            "description": "Your first ISL word.",
            "targets": ["H", "I"],
        },
    ],
}


def get_lessons(language: str) -> List[dict]:
    return LESSONS.get(language.upper(), [])


def get_lesson_by_id(language: str, lesson_id: str) -> dict:
    for lesson in get_lessons(language):
        if lesson["id"] == lesson_id:
            return lesson
    raise KeyError(f"No lesson '{lesson_id}' for language '{language}'")
