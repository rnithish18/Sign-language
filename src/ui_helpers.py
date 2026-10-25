"""
ui_helpers.py
-------------
Pure UI helper functions for SignBridge:
- Status pill state construction (Live/Offline, Hand Tracking/No hand)
- Real hold-counter progress dot visualization
- Model calibration checking and phrase chip formatting
- Status pill HTML generation
"""

from typing import Optional

from src.config import (
    PHRASE_ICONS,
    THEME_GREEN,
    THEME_RED,
    THEME_TEXT_MUTED,
)


def build_stream_status_pill(is_playing: bool) -> dict:
    """
    Build status pill properties for the main header WebRTC stream status.

    Returns dict with keys: label, dot_color, text_color, bg_color, border_color.
    """
    if is_playing:
        return {
            "label": "Live",
            "dot_color": THEME_GREEN,
            "text_color": THEME_GREEN,
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "rgba(16, 185, 129, 0.35)",
        }
    return {
        "label": "Offline",
        "dot_color": "#64748b",
        "text_color": THEME_TEXT_MUTED,
        "bg_color": "rgba(100, 116, 139, 0.15)",
        "border_color": "rgba(100, 116, 139, 0.3)",
    }


def build_hand_tracking_pill(hand_detected: bool) -> dict:
    """
    Build status pill properties for the camera feed hand-tracking indicator.

    Returns dict with keys: label, dot_color, text_color, bg_color, border_color.
    """
    if hand_detected:
        return {
            "label": "Hand Tracking",
            "dot_color": THEME_GREEN,
            "text_color": THEME_GREEN,
            "bg_color": "rgba(16, 185, 129, 0.15)",
            "border_color": "rgba(16, 185, 129, 0.35)",
        }
    return {
        "label": "No hand",
        "dot_color": THEME_RED,
        "text_color": "#ff8888",
        "bg_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "rgba(239, 68, 68, 0.3)",
    }


def build_analyzing_dots(progress: float, total_dots: int = 12) -> str:
    """
    Generate progress dots driven strictly by the real hold-counter progress (0.0 to 1.0).

    Returns a string of filled and empty circle characters.
    """
    clamped = max(0.0, min(1.0, float(progress)))
    filled = int(round(clamped * total_dots))
    return "●" * filled + "○" * (total_dots - filled)


def is_phrase_calibrated(phrase_label: str, metadata: Optional[dict]) -> bool:
    """
    Determine whether a phrase gesture is calibrated (has real trained data).

    Returns False if metadata explicitly lists the phrase in 'needs_calibration'
    or if 'labels_with_real_data' is present and does not include the phrase.
    """
    if not metadata:
        return True
    needs_cal = metadata.get("needs_calibration")
    if needs_cal is not None:
        return phrase_label not in needs_cal
    labels_real = metadata.get("labels_with_real_data")
    if labels_real is not None:
        return phrase_label in labels_real
    return True


def format_phrase_chip_label(phrase_label: str, phrase_text: str, is_calibrated: bool) -> str:
    """
    Format the text label for a phrase chip button, including its icon
    and a clear 'Needs cal' state when the label has no real training data.
    """
    icon = PHRASE_ICONS.get(phrase_label, "💬")
    if is_calibrated:
        return f"{icon}  {phrase_text}"
    return f"{icon}  {phrase_text}\n⚠️ Needs cal"


def render_html_pill(pill_data: dict) -> str:
    """Render a pill dictionary into a styled inline HTML badge."""
    label = pill_data["label"]
    dot_color = pill_data["dot_color"]
    text_color = pill_data["text_color"]
    bg_color = pill_data["bg_color"]
    border_color = pill_data["border_color"]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;padding:4px 12px;'
        f'border-radius:9999px;font-size:0.8rem;font-weight:600;letter-spacing:0.02em;'
        f'background:{bg_color};color:{text_color};border:1px solid {border_color};">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_color};'
        f'display:inline-block;box-shadow:0 0 6px {dot_color};"></span>{label}</span>'
    )
