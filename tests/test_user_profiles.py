"""
test_user_profiles.py
----------------------
Tests for username validation, preference persistence, user listing, and
deletion in UserProfileManager.
"""

import json
import os
import tempfile

import pytest

from utils.user_profiles import (
    UserProfileManager,
    DEFAULT_PREFERENCES,
    sanitize_username,
    validate_username,
)


# ---------------------------------------------------------------------------
# Username validation
# ---------------------------------------------------------------------------

def test_validate_username_valid():
    assert validate_username("Alice") is None
    assert validate_username("user-1") is None
    assert validate_username("John Smith") is None
    assert validate_username("a") is None
    assert validate_username("a" * 32) is None


def test_validate_username_empty():
    err = validate_username("")
    assert err is not None
    assert "empty" in err.lower()


def test_validate_username_too_long():
    err = validate_username("a" * 33)
    assert err is not None


def test_validate_username_bad_chars():
    err = validate_username("user@name!")
    assert err is not None


def test_sanitize_username():
    assert sanitize_username("  Alice  ") == "Alice"
    assert sanitize_username("John   Smith") == "John Smith"
    assert sanitize_username("  ") == ""


# ---------------------------------------------------------------------------
# UserProfileManager
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_profile_path(tmp_path):
    return str(tmp_path / "progress.json")


def test_list_users_empty(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    assert pm.list_users() == []


def test_save_and_load_preferences(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    prefs = {**DEFAULT_PREFERENCES, "language": "ISL", "conf_thresh": 0.8}
    pm.save_preferences("Alice", prefs)

    loaded = pm.get_preferences("Alice")
    assert loaded["language"] == "ISL"
    assert loaded["conf_thresh"] == 0.8
    # Keys not in saved prefs fall back to defaults
    assert loaded["tts_backend"] == DEFAULT_PREFERENCES["tts_backend"]


def test_list_users_after_save(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    pm.save_preferences("Alice", {})
    pm.save_preferences("Bob", {})
    users = pm.list_users()
    assert "Alice" in users
    assert "Bob" in users


def test_user_exists(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    assert not pm.user_exists("Alice")
    pm.save_preferences("Alice", {})
    assert pm.user_exists("Alice")


def test_delete_user(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    pm.save_preferences("Alice", {"language": "ISL"})
    assert pm.user_exists("Alice")

    pm.delete_user("Alice")
    assert not pm.user_exists("Alice")
    # Prefs gone too
    loaded = pm.get_preferences("Alice")
    assert loaded == DEFAULT_PREFERENCES


def test_get_preferences_unknown_user_returns_defaults(tmp_profile_path):
    pm = UserProfileManager(tmp_profile_path)
    loaded = pm.get_preferences("NonExistent")
    assert loaded == DEFAULT_PREFERENCES
