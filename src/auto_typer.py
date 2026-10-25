"""
auto_typer.py
-------------
Auto-typing state machine for SignBridge.

Converts a continuous stream of gesture predictions into typed characters, words,
and control actions in a TextBuffer.

Rules:
1. When the same label is predicted with confidence >= MIN_CONF (0.80) for a
   continuous hold of >= HOLD_SECONDS (0.8s), append it to the recognized text exactly once.
2. After appending, require the hand to change to a different label, or leave
   the frame for >= HAND_LOST_TIMEOUT_SECONDS (0.5s), before the same letter can be typed again.
   This prevents "CCCCC", while allowing real double letters (e.g. "LL" in "HELLO")
   once the hand disappears or the label changes and returns.
3. Special labels:
   - SPACE appends a space (finishes current word in TextBuffer)
   - CLEAR empties the text buffer
   - DEL / BACKSPACE removes the last character
   - SPEAK triggers text-to-speech
   - Ignore labels flagged "needs calibration" (e.g. PHRASE_* without real training data)
4. Reset the hold timer whenever the hand is lost or the label changes.
"""

import time
from typing import Any, Callable, Optional, Set

import numpy as np

from src.config import (
    GESTURE_BACKSPACE,
    GESTURE_CLEAR,
    GESTURE_DEL,
    GESTURE_SPACE,
    GESTURE_SPEAK,
    HAND_LOST_TIMEOUT_SECONDS,
    HOLD_SECONDS,
    MIN_CONF,
)
from src.text_buffer import TextBuffer


class AutoTyper:
    def __init__(
        self,
        text_buffer: Optional[TextBuffer] = None,
        hold_seconds: float = HOLD_SECONDS,
        min_confidence: float = MIN_CONF,
        hand_lost_timeout: float = HAND_LOST_TIMEOUT_SECONDS,
        repeat_delay_seconds: Optional[float] = None,
        needs_calibration: Optional[Set[str]] = None,
        on_speak: Optional[Callable[[str], None]] = None,
    ):
        self.text_buffer = text_buffer if text_buffer is not None else TextBuffer(auto_correct=True)
        self.hold_seconds = float(hold_seconds)
        self.min_confidence = float(min_confidence)
        effective_timeout = (
            repeat_delay_seconds if repeat_delay_seconds is not None else hand_lost_timeout
        )
        self.hand_lost_timeout = float(effective_timeout)
        self.repeat_delay_seconds = float(effective_timeout)
        self.needs_calibration = set(needs_calibration) if needs_calibration is not None else set()
        self.on_speak = on_speak

        self._held_label: Optional[str] = None
        self._hold_start_time: Optional[float] = None
        self._last_committed_label: Optional[str] = None
        self._hand_lost_time: Optional[float] = None
        self._progress: float = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def held_label(self) -> Optional[str]:
        return self._held_label

    @property
    def last_committed_label(self) -> Optional[str]:
        return self._last_committed_label

    def reset(self) -> None:
        """Reset state machine counters and locks."""
        self._held_label = None
        self._hold_start_time = None
        self._last_committed_label = None
        self._hand_lost_time = None
        self._progress = 0.0

    def process(
        self,
        label: Optional[str],
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Process a single prediction event.

        Parameters
        ----------
        label : str | None
            The predicted label (or None if no hand detected).
        confidence : float
            Prediction confidence (0.0 to 1.0).
        timestamp : float | None
            Current timestamp in seconds. Uses time.time() if None.

        Returns
        -------
        dict with keys:
            "committed_label": str | None (set only on the exact frame a letter commits)
            "progress": float (0.0 to 1.0)
            "held_label": str | None
            "text": str (current sentence from text_buffer)
            "hand_detected": bool
        """
        if timestamp is None:
            timestamp = time.time()

        is_valid = (
            label is not None
            and confidence >= self.min_confidence
            and label not in self.needs_calibration
        )

        if not is_valid:
            # Hand is lost or confidence is low -> reset hold timer
            self._held_label = None
            self._hold_start_time = None
            self._progress = 0.0

            if self._hand_lost_time is None:
                self._hand_lost_time = timestamp
            else:
                # If hand has been gone for >= hand_lost_timeout, clear repeat lock
                if (timestamp - self._hand_lost_time) >= self.hand_lost_timeout:
                    self._last_committed_label = None

            return {
                "committed_label": None,
                "progress": 0.0,
                "held_label": None,
                "text": self.text_buffer.sentence,
                "hand_detected": False,
            }

        # Valid hand detection with sufficient confidence
        self._hand_lost_time = None

        if label != self._held_label:
            # Label changed! Reset hold timer for the new label
            self._held_label = label
            self._hold_start_time = timestamp
            self._progress = 0.0
            # If changing to a different label, release repeat lock for previous label
            if label != self._last_committed_label:
                self._last_committed_label = None

        # Calculate hold progress
        elapsed = timestamp - (self._hold_start_time if self._hold_start_time is not None else timestamp)
        self._progress = min(max(elapsed / max(self.hold_seconds, 1e-6), 0.0), 1.0)

        committed = None
        if (elapsed + 1e-5) >= self.hold_seconds:
            # Hold threshold reached! Check if we can commit this label
            if self._last_committed_label != label:
                committed = label
                self._last_committed_label = label
                self._apply_label(label)

        return {
            "committed_label": committed,
            "progress": round(self._progress, 2),
            "held_label": self._held_label,
            "text": self.text_buffer.sentence,
            "hand_detected": True,
        }

    def process_frame(
        self,
        feature_vector: Optional[np.ndarray],
        predictor: Any,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Convenience method to process a raw feature vector using a predictor.
        """
        if feature_vector is None or predictor is None:
            return self.process(label=None, confidence=0.0, timestamp=timestamp)

        label, confidence = predictor.raw_predict(feature_vector)
        return self.process(label=label, confidence=confidence, timestamp=timestamp)

    def _apply_label(self, label: str) -> None:
        """Apply committed gesture to the underlying text buffer."""
        if label == GESTURE_CLEAR:
            self.text_buffer.clear()
        elif label in (GESTURE_DEL, GESTURE_BACKSPACE, "DEL"):
            self.text_buffer.handle_label("DEL")
        elif label == GESTURE_SPACE:
            self.text_buffer.handle_label("SPACE")
        elif label == GESTURE_SPEAK:
            self.text_buffer.handle_label("SPEAK")
            if self.on_speak:
                self.on_speak(self.text_buffer.sentence)
        else:
            self.text_buffer.handle_label(label)
