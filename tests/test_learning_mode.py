"""
test_learning_mode.py
----------------------
Tests for LessonSession and Practice Mistakes ordering.
"""

import pytest
from unittest.mock import MagicMock
from src.learning_mode import LessonSession, build_lesson_menu


def _make_store(letter_attempts=None, letter_correct=None, lessons_completed=None):
    """Build a mock ProgressStore with controllable per-letter stats."""
    store = MagicMock()
    attempts = letter_attempts or {}
    correct = letter_correct or {}
    completed = lessons_completed or []

    store.get_progress.return_value = {
        "letter_attempts": attempts,
        "letter_correct": correct,
        "lessons_completed": completed,
    }

    def accuracy_for(username, language, target):
        a = attempts.get(target, 0)
        c = correct.get(target, 0)
        return round((c / a) * 100, 1) if a > 0 else 0.0

    store.accuracy_for.side_effect = accuracy_for
    store.overall_accuracy.return_value = 70.0
    return store


# ---------------------------------------------------------------------------
# Practice Mistakes ordering
# ---------------------------------------------------------------------------

def test_practice_mistakes_resurfaces_worst_letters():
    """Letters with lowest accuracy should appear first in practice targets."""
    store = _make_store(
        letter_attempts={"A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        letter_correct= {"A": 2,  "B": 9,  "C": 5,  "D": 1,  "E": 8},
    )
    session = LessonSession(
        language="ASL",
        lesson_id="practice_mistakes",
        username="test_user",
        progress_store=store,
    )
    # Targets should contain the worst letters (D=10%, A=20%, C=50%)
    assert "D" in session.targets
    assert "A" in session.targets
    # Perfect/near-perfect (B=90%, E=80%) may or may not be included
    # but worst should be at the head when iterated
    worst_idx_D = session.targets.index("D")
    worst_idx_B = session.targets.index("B") if "B" in session.targets else 999
    assert worst_idx_D < worst_idx_B or "B" not in session.targets


def test_practice_mistakes_fallback_no_attempts():
    """If the user has made no attempts yet, fallback to ['A','B','C']."""
    store = _make_store()
    session = LessonSession(
        language="ASL",
        lesson_id="practice_mistakes",
        username="new_user",
        progress_store=store,
    )
    # Should not crash and should have some targets
    assert len(session.targets) > 0


def test_practice_mistakes_each_repeated_twice():
    """Each letter in practice targets is repeated exactly twice."""
    store = _make_store(
        letter_attempts={"A": 5, "B": 5},
        letter_correct= {"A": 1, "B": 2},
    )
    session = LessonSession(
        language="ASL",
        lesson_id="practice_mistakes",
        username="u",
        progress_store=store,
    )
    from collections import Counter
    counts = Counter(session.targets)
    for letter, count in counts.items():
        assert count == 2, f"Expected each letter repeated twice; got {letter}={count}"


def test_practice_mistakes_appears_in_lesson_menu():
    """build_lesson_menu should always include the practice_mistakes entry."""
    store = _make_store()
    store.get_progress.return_value = {
        "letter_attempts": {}, "letter_correct": {}, "lessons_completed": []
    }
    menu = build_lesson_menu("ASL", "u", store)
    ids = [l["id"] for l in menu]
    assert "practice_mistakes" in ids


# ---------------------------------------------------------------------------
# Normal lesson session
# ---------------------------------------------------------------------------

def test_lesson_session_submit_correct(monkeypatch):
    """submit_prediction returns True on match and advances index."""
    import src.learning_mode as learning_mode_mod
    monkeypatch.setattr(
        learning_mode_mod, "get_lesson_by_id",
        lambda lang, lid: {"targets": ["A", "B", "C"],
                           "title": "Test", "description": ""},
    )
    store = _make_store()
    session = LessonSession("ASL", "asl_basics_1", "u", store)

    assert session.current_target == "A"
    result = session.submit_prediction("A")
    assert result is True
    assert session.current_index == 1
    assert session.current_target == "B"


def test_lesson_session_submit_wrong(monkeypatch):
    import src.learning_mode as learning_mode_mod
    monkeypatch.setattr(
        learning_mode_mod, "get_lesson_by_id",
        lambda lang, lid: {"targets": ["A", "B"],
                           "title": "Test", "description": ""},
    )
    store = _make_store()
    session = LessonSession("ASL", "asl_basics_1", "u", store)

    result = session.submit_prediction("X")
    assert result is False
    assert session.current_index == 0   # didn't advance


def test_lesson_finishes_when_all_correct(monkeypatch):
    import src.learning_mode as learning_mode_mod
    monkeypatch.setattr(
        learning_mode_mod, "get_lesson_by_id",
        lambda lang, lid: {"targets": ["A"],
                           "title": "Test", "description": ""},
    )
    store = _make_store()
    session = LessonSession("ASL", "asl_basics_1", "u", store)
    session.submit_prediction("A")
    assert session.finished is True
    summary = session.summary()
    assert summary["score_pct"] == 100.0
