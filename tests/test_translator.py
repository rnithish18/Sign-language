"""
test_translator.py
-------------------
Phase C tests — all network calls are mocked, tests pass without internet.

Tests:
 - English returns input unchanged (no API call)
 - Successful translation calls deep_translator once
 - Cache hit returns immediately without calling API again
 - Offline / exception path returns original text + error message
 - ImportError path (deep_translator not installed) returns gracefully
 - Empty input returns unchanged
 - clear_cache() works
"""

import pytest
from unittest.mock import MagicMock, patch

from src.translator import translate, clear_cache


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the translation cache before each test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# English passthrough
# ---------------------------------------------------------------------------

def test_english_returns_unchanged_no_api():
    result = translate("Hello", "en")
    assert result["translated"] == "Hello"
    assert result["error"] is None
    assert result["cached"] is True  # English is always "cached" (no API call)


# ---------------------------------------------------------------------------
# Successful translation
# ---------------------------------------------------------------------------

def test_successful_tamil_translation():
    mock_translator = MagicMock()
    mock_translator.return_value.translate.return_value = "வணக்கம்"

    with patch.dict("sys.modules", {"deep_translator": MagicMock(
        GoogleTranslator=mock_translator
    )}):
        result = translate("Hello", "ta")

    assert result["translated"] == "வணக்கம்"
    assert result["error"] is None
    assert result["cached"] is False


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------

def test_cache_hit_no_second_api_call():
    call_count = {"n": 0}

    def fake_translate(text):
        call_count["n"] += 1
        return "நன்றி"

    mock_gt = MagicMock()
    mock_gt.return_value.translate.side_effect = fake_translate

    with patch.dict("sys.modules", {"deep_translator": MagicMock(
        GoogleTranslator=mock_gt
    )}):
        r1 = translate("Thank you", "ta")
        r2 = translate("Thank you", "ta")   # second call — should hit cache

    assert call_count["n"] == 1           # API called only once
    assert r2["cached"] is True
    assert r2["translated"] == "நன்றி"


# ---------------------------------------------------------------------------
# Offline / network error fallback
# ---------------------------------------------------------------------------

def test_offline_fallback_returns_original():
    mock_gt = MagicMock()
    mock_gt.return_value.translate.side_effect = ConnectionError("No network")

    with patch.dict("sys.modules", {"deep_translator": MagicMock(
        GoogleTranslator=mock_gt
    )}):
        result = translate("Water", "ta")

    assert result["translated"] == "Water"  # falls back to original
    assert result["error"] is not None
    assert "offline" in result["error"].lower() or "error" in result["error"].lower()


# ---------------------------------------------------------------------------
# ImportError (deep_translator not installed)
# ---------------------------------------------------------------------------

def test_import_error_fallback():
    with patch.dict("sys.modules", {"deep_translator": None}):
        result = translate("Yes", "hi")

    assert result["translated"] == "Yes"
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string_returns_unchanged():
    result = translate("", "ta")
    assert result["translated"] == ""
    assert result["error"] is None


def test_whitespace_only_returns_unchanged():
    result = translate("   ", "hi")
    assert result["translated"] == "   "


def test_hindi_translation():
    mock_gt = MagicMock()
    mock_gt.return_value.translate.return_value = "हाँ"

    with patch.dict("sys.modules", {"deep_translator": MagicMock(
        GoogleTranslator=mock_gt
    )}):
        result = translate("Yes", "hi")

    assert result["translated"] == "हाँ"
    assert result["target_lang"] == "hi"
