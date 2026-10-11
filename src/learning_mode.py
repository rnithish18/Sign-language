"""
learning_mode.py
-----------------
Session controller for Interactive Learning Mode. Wraps a GesturePredictor
around a lesson's target sequence, giving instant right/wrong feedback and
recording results into ProgressStore for progress tracking.

Used by app.py; kept separate from the Streamlit code so it is UI-agnostic
and unit-testable.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.learning_content import get_lesson_by_id, get_lessons
from utils.progress_store import ProgressStore


@dataclass
class LessonSession:
    language: str
    lesson_id: str
    username: str
    progress_store: ProgressStore
    targets: List[str] = field(default_factory=list)
    current_index: int = 0
    correct_count: int = 0
    attempts_this_target: int = 0
    finished: bool = False

    def __post_init__(self):
        if self.lesson_id == "practice_mistakes":
            self.title = "Practice Mistakes"
            self.description = "Review your most missed signs."
            self._generate_practice_targets()
        else:
            lesson = get_lesson_by_id(self.language, self.lesson_id)
            self.targets = list(lesson["targets"])
            self.title = lesson["title"]
            self.description = lesson["description"]

    def _generate_practice_targets(self):
        progress = self.progress_store.get_progress(self.username, self.language)
        attempts = progress.get("letter_attempts", {})
        
        accuracies = []
        for target, count in attempts.items():
            if count > 0:
                acc = self.progress_store.accuracy_for(self.username, self.language, target)
                accuracies.append((acc, target))
                
        accuracies.sort(key=lambda x: x[0])
        
        # Take the 5 worst targets (under 100% accuracy)
        mistakes = [target for acc, target in accuracies if acc < 100.0][:5]
        
        if not mistakes:
            # Fallback if perfect or new
            mistakes = [target for acc, target in accuracies][:5]
            if not mistakes:
                mistakes = ["A", "B", "C"]
                
        self.targets = mistakes * 2  # Practice each twice

    @property
    def current_target(self) -> Optional[str]:
        if self.finished or self.current_index >= len(self.targets):
            return None
        return self.targets[self.current_index]

    @property
    def progress_fraction(self) -> float:
        return self.current_index / len(self.targets) if self.targets else 1.0

    def submit_prediction(self, predicted_label: str) -> bool:
        """
        Called each time the predictor commits a gesture while a lesson is
        active. Returns True if it matched the current target (and advances
        the lesson); False otherwise (and logs a failed attempt).
        """
        target = self.current_target
        if target is None:
            return False

        self.attempts_this_target += 1
        is_correct = predicted_label == target
        self.progress_store.record_attempt(self.username, self.language, target, is_correct)

        if is_correct:
            self.correct_count += 1
            self.current_index += 1
            self.attempts_this_target = 0
            if self.current_index >= len(self.targets):
                self.finished = True
                self.progress_store.mark_lesson_complete(self.username, self.language, self.lesson_id)
        return is_correct

    def summary(self) -> dict:
        total = len(self.targets)
        return {
            "title": self.title,
            "completed": self.finished,
            "correct": self.correct_count,
            "total": total,
            "score_pct": round((self.correct_count / total) * 100, 1) if total else 0.0,
        }


def build_lesson_menu(language: str, username: str, progress_store: ProgressStore) -> List[dict]:
    """Returns lesson metadata annotated with completion status, for the UI list."""
    lessons = get_lessons(language)
    completed = set(progress_store.get_progress(username, language)["lessons_completed"])
    menu = []
    for lesson in lessons:
        menu.append(
            {
                **lesson,
                "completed": lesson["id"] in completed,
            }
        )
        
    menu.append(
        {
            "id": "practice_mistakes",
            "title": "Practice Mistakes",
            "description": "Review signs with low accuracy scores.",
            "targets": [],
            "completed": False,
        }
    )
    return menu
