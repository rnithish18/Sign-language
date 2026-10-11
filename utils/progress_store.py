"""
progress_store.py
------------------
Minimal JSON-file-backed persistence for learning-mode progress. No
database needed for a single-user desktop/Streamlit app; this keeps a
per-user record of which lessons are complete and per-letter accuracy.

Structure of the JSON file:
{
  "users": {
    "<username>": {
      "ASL": {
        "lessons_completed": ["asl_basics_1", ...],
        "letter_attempts": {"A": 12, "B": 7, ...},
        "letter_correct":  {"A": 10, "B": 5, ...}
      },
      "ISL": { ... }
    }
  }
}
"""

import json
import os
import threading
from typing import Dict

_LOCK = threading.Lock()


class ProgressStore:
    def __init__(self, path: str):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"users": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with _LOCK:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)

    def _user_lang(self, username: str, language: str) -> dict:
        users = self._data.setdefault("users", {})
        user = users.setdefault(username, {})
        lang = user.setdefault(
            language,
            {"lessons_completed": [], "letter_attempts": {}, "letter_correct": {}},
        )
        return lang

    def record_attempt(self, username: str, language: str, target: str, correct: bool):
        lang = self._user_lang(username, language)
        lang["letter_attempts"][target] = lang["letter_attempts"].get(target, 0) + 1
        if correct:
            lang["letter_correct"][target] = lang["letter_correct"].get(target, 0) + 1
        self._save()

    def mark_lesson_complete(self, username: str, language: str, lesson_id: str):
        lang = self._user_lang(username, language)
        if lesson_id not in lang["lessons_completed"]:
            lang["lessons_completed"].append(lesson_id)
        self._save()

    def get_progress(self, username: str, language: str) -> dict:
        return self._user_lang(username, language)

    def accuracy_for(self, username: str, language: str, target: str) -> float:
        lang = self._user_lang(username, language)
        attempts = lang["letter_attempts"].get(target, 0)
        correct = lang["letter_correct"].get(target, 0)
        return round((correct / attempts) * 100, 1) if attempts else 0.0

    def overall_accuracy(self, username: str, language: str) -> float:
        lang = self._user_lang(username, language)
        total_attempts = sum(lang["letter_attempts"].values())
        total_correct = sum(lang["letter_correct"].values())
        return round((total_correct / total_attempts) * 100, 1) if total_attempts else 0.0
