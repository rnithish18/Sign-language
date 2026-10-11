"""
user_profiles.py
-----------------
Lightweight profile manager built on top of ProgressStore's JSON file.
Handles:
  - Listing all known users
  - Sanitizing/validating usernames
  - Deleting a user profile
  - Persisting per-user preferences (language, TTS backend, settings)

No database is required — the same JSON file used by ProgressStore is
extended with a top-level "profiles" key for preferences.
"""

import json
import os
import re
import threading
from typing import Dict, List, Optional

_LOCK = threading.Lock()

# Username rules: 1-32 chars, alphanumeric + underscore/hyphen/space
_USERNAME_RE = re.compile(r"^[\w\s\-]{1,32}$")

DEFAULT_PREFERENCES = {
    "language": "ASL",
    "tts_backend": "offline",
    "conf_thresh": 0.65,
    "hold_frames": 12,
    "auto_correct": True,
}


def sanitize_username(raw: str) -> str:
    """Strip leading/trailing whitespace; collapse internal whitespace to single space."""
    return re.sub(r"\s+", " ", raw.strip())


def validate_username(username: str) -> Optional[str]:
    """
    Returns None if valid, or an error message string if invalid.
    """
    if not username:
        return "Username cannot be empty."
    if not _USERNAME_RE.match(username):
        return "Username must be 1-32 characters: letters, numbers, spaces, hyphens, underscores."
    return None


class UserProfileManager:
    """
    Thin wrapper around the shared progress JSON file that adds a
    "profiles" section for per-user preferences and a user registry.
    """

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
        return {"users": {}, "profiles": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with _LOCK:
            # Re-read before writing to avoid stomping concurrent writes
            on_disk = self._load()
            on_disk.update(self._data)
            with open(self.path, "w") as f:
                json.dump(on_disk, f, indent=2)
        self._data = self._load()

    # ------------------------------------------------------------------
    # User registry
    # ------------------------------------------------------------------

    def list_users(self) -> List[str]:
        """Return sorted list of all known usernames (from both progress and profiles)."""
        self._data = self._load()
        users_in_progress = set(self._data.get("users", {}).keys())
        users_in_profiles = set(self._data.get("profiles", {}).keys())
        all_users = sorted(users_in_progress | users_in_profiles)
        return all_users

    def user_exists(self, username: str) -> bool:
        return username in self.list_users()

    def delete_user(self, username: str):
        """Remove all progress and preferences for a user."""
        self._data = self._load()
        self._data.get("users", {}).pop(username, None)
        self._data.get("profiles", {}).pop(username, None)
        self._save()

    # ------------------------------------------------------------------
    # Per-user preferences
    # ------------------------------------------------------------------

    def get_preferences(self, username: str) -> dict:
        self._data = self._load()
        stored = self._data.get("profiles", {}).get(username, {})
        return {**DEFAULT_PREFERENCES, **stored}

    def save_preferences(self, username: str, prefs: dict):
        self._data = self._load()
        profiles = self._data.setdefault("profiles", {})
        existing = profiles.get(username, {})
        existing.update(prefs)
        profiles[username] = existing
        self._save()
