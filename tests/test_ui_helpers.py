"""
test_ui_helpers.py
------------------
Unit tests for ui_helpers.py:
- Stream status pill (Live/Offline)
- Hand tracking pill (Hand Tracking/No hand)
- Real hold-counter progress dots
- Phrase calibration detection and formatting
- Pill HTML rendering
"""

from src.config import THEME_GREEN, THEME_RED, THEME_TEXT_MUTED
from src.ui_helpers import (
    build_analyzing_dots,
    build_hand_tracking_pill,
    build_stream_status_pill,
    format_phrase_chip_label,
    is_phrase_calibrated,
    render_html_pill,
)


def test_stream_status_pill_live():
    pill = build_stream_status_pill(is_playing=True)
    assert pill["label"] == "Live"
    assert pill["dot_color"] == THEME_GREEN
    assert pill["text_color"] == THEME_GREEN
    assert "rgba" in pill["bg_color"]


def test_stream_status_pill_offline():
    pill = build_stream_status_pill(is_playing=False)
    assert pill["label"] == "Offline"
    assert pill["text_color"] == THEME_TEXT_MUTED
    assert "#64748b" in pill["dot_color"]


def test_hand_tracking_pill_detected():
    pill = build_hand_tracking_pill(hand_detected=True)
    assert pill["label"] == "Hand Tracking"
    assert pill["dot_color"] == THEME_GREEN
    assert pill["text_color"] == THEME_GREEN


def test_hand_tracking_pill_no_hand():
    pill = build_hand_tracking_pill(hand_detected=False)
    assert pill["label"] == "No hand"
    assert pill["dot_color"] == THEME_RED


def test_build_analyzing_dots():
    # 0% progress -> all empty
    dots_0 = build_analyzing_dots(0.0, total_dots=12)
    assert dots_0 == "○○○○○○○○○○○○"
    assert dots_0.count("●") == 0

    # 50% progress -> 6 filled, 6 empty
    dots_half = build_analyzing_dots(0.5, total_dots=12)
    assert dots_half == "●●●●●●○○○○○○"

    # 100% progress -> all filled
    dots_full = build_analyzing_dots(1.0, total_dots=12)
    assert dots_full == "●●●●●●●●●●●●"

    # Clamping behavior
    assert build_analyzing_dots(-0.5, total_dots=10) == "○○○○○○○○○○"
    assert build_analyzing_dots(1.5, total_dots=10) == "●●●●●●●●●●"


def test_is_phrase_calibrated():
    # None metadata
    assert is_phrase_calibrated("PHRASE_HELP", None) is True

    # needs_calibration list
    meta_cal = {"needs_calibration": ["PHRASE_HELP", "PHRASE_NO"]}
    assert is_phrase_calibrated("PHRASE_HELP", meta_cal) is False
    assert is_phrase_calibrated("PHRASE_YES", meta_cal) is True

    # labels_with_real_data list
    meta_real = {"labels_with_real_data": ["PHRASE_YES", "PHRASE_WATER"]}
    assert is_phrase_calibrated("PHRASE_YES", meta_real) is True
    assert is_phrase_calibrated("PHRASE_HELP", meta_real) is False


def test_format_phrase_chip_label():
    label_cal = format_phrase_chip_label("PHRASE_HELP", "I need help", is_calibrated=True)
    assert "✋" in label_cal
    assert "I need help" in label_cal
    assert "Needs cal" not in label_cal

    label_uncal = format_phrase_chip_label("PHRASE_HELP", "I need help", is_calibrated=False)
    assert "✋" in label_uncal
    assert "I need help" in label_uncal
    assert "Needs cal" in label_uncal


def test_render_html_pill():
    pill = build_stream_status_pill(is_playing=True)
    html = render_html_pill(pill)
    assert "<span" in html
    assert "Live" in html
    assert THEME_GREEN in html
