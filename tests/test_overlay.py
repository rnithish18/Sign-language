"""
test_overlay.py
---------------
Unit test for SignProcessor.draw_live_letter_overlay().

Verifies headlessly (no WebRTC, no camera) that the static overlay helper
actually modifies pixel data in the frame under the three key scenarios:
  1. Hand detected + high confidence  → gold letter drawn, frame is modified.
  2. Hand detected + low confidence   → grey letter drawn, frame is modified.
  3. No hand / no label               → frame is NOT modified (no stale letter).

Import strategy: we mock streamlit_webrtc before importing app so the module
loads cleanly in a non-Streamlit environment (same trick as test_app.py).
"""

import sys
from unittest.mock import MagicMock

# ── Mock streamlit-webrtc before app.py is imported ──────────────────────────
_mock_webrtc = MagicMock()
_mock_webrtc.webrtc_streamer.return_value.video_processor = None
sys.modules.setdefault("streamlit_webrtc", _mock_webrtc)

import numpy as np
import pytest

from app import SignProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a solid-grey C-contiguous writable frame."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    assert img.flags["C_CONTIGUOUS"]
    assert img.flags.writeable
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDrawLiveLetterOverlay:
    """Pixel-level unit tests for the static overlay helper."""

    def test_high_confidence_modifies_frame(self):
        """
        When hand is detected and confidence ≥ threshold, the frame should
        have pixels changed (the letter is drawn).
        """
        img = blank_frame()
        before = img.copy()

        SignProcessor.draw_live_letter_overlay(
            img,
            smoothed_label="A",
            confidence=0.90,
            hand_detected=True,
            confidence_threshold=0.65,
        )

        assert not np.array_equal(img, before), (
            "Frame should be modified when hand is detected and confidence is high."
        )

    def test_low_confidence_modifies_frame(self):
        """
        When confidence < threshold the overlay still draws (grey/muted letter),
        so the frame should still be modified.
        """
        img = blank_frame()
        before = img.copy()

        SignProcessor.draw_live_letter_overlay(
            img,
            smoothed_label="B",
            confidence=0.30,
            hand_detected=True,
            confidence_threshold=0.65,
        )

        assert not np.array_equal(img, before), (
            "Frame should be modified even when confidence is below threshold "
            "(grey best-guess overlay should still draw)."
        )

    def test_no_hand_leaves_frame_unchanged(self):
        """
        When hand_detected is False the function must return early and leave
        the frame completely untouched (no stale letter displayed).
        """
        img = blank_frame()
        before = img.copy()

        SignProcessor.draw_live_letter_overlay(
            img,
            smoothed_label="C",
            confidence=0.95,
            hand_detected=False,
            confidence_threshold=0.65,
        )

        assert np.array_equal(img, before), (
            "Frame must NOT be modified when no hand is detected."
        )

    def test_no_label_leaves_frame_unchanged(self):
        """
        When smoothed_label is None (e.g. below threshold with no best-guess)
        the frame must not be modified.
        """
        img = blank_frame()
        before = img.copy()

        SignProcessor.draw_live_letter_overlay(
            img,
            smoothed_label=None,
            confidence=0.95,
            hand_detected=True,
            confidence_threshold=0.65,
        )

        assert np.array_equal(img, before), (
            "Frame must NOT be modified when smoothed_label is None."
        )

    def test_high_conf_uses_brighter_pixels_than_low_conf(self):
        """
        High-confidence overlay uses a bold gold colour; low-confidence uses
        grey.  Verify that the mean brightness in the top-left region is
        different between the two cases (not identical colour).
        """
        img_hi = blank_frame()
        img_lo = blank_frame()

        SignProcessor.draw_live_letter_overlay(
            img_hi, smoothed_label="D", confidence=0.95,
            hand_detected=True, confidence_threshold=0.65,
        )
        SignProcessor.draw_live_letter_overlay(
            img_lo, smoothed_label="D", confidence=0.30,
            hand_detected=True, confidence_threshold=0.65,
        )

        # Compare mean channel values in top-left 100×200 region
        roi_hi = img_hi[:100, :200].mean()
        roi_lo = img_lo[:100, :200].mean()

        assert roi_hi != roi_lo, (
            "High-conf and low-conf overlays should use visually distinct colours."
        )

    def test_frame_stays_contiguous_after_overlay(self):
        """
        The overlay uses cv2.addWeighted which may interact with numpy views.
        Confirm the frame remains C-contiguous and writable after the call.
        """
        img = blank_frame()

        SignProcessor.draw_live_letter_overlay(
            img, smoothed_label="E", confidence=0.80,
            hand_detected=True, confidence_threshold=0.65,
        )

        assert img.flags["C_CONTIGUOUS"], "Frame must remain C-contiguous after overlay."
        assert img.flags.writeable, "Frame must remain writable after overlay."
