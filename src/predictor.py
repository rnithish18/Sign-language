"""
predictor.py
------------
Loads a trained model + label encoder and turns a stream of per-frame
landmark feature vectors into stable, confident gesture predictions.

Two things make raw per-frame classifier output unusable directly:
1. Frame-to-frame jitter -> flickering predictions.
2. No sense of "the user is actively holding a sign" vs "hand in transit
   between signs" -> letters would spam into the text buffer.

GesturePredictor solves both with a sliding-window majority vote plus a
"hold-to-commit" state machine (see HOLD_FRAMES_TO_COMMIT in config.py).
"""

import os
from collections import Counter, deque
from typing import Optional, Tuple

import joblib
import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    COOLDOWN_FRAMES_AFTER_COMMIT,
    HOLD_FRAMES_TO_COMMIT,
    PREDICTION_SMOOTHING_WINDOW,
    label_encoder_path,
    metadata_path,
    model_path,
)


class ModelNotFoundError(Exception):
    pass


class GesturePredictor:
    def __init__(self, language: str):
        self.language = language
        mpath, lpath = model_path(language), label_encoder_path(language)
        if not (os.path.exists(mpath) and os.path.exists(lpath)):
            raise ModelNotFoundError(
                f"No trained model for '{language}'. Run train_model.py first.\n"
                f"Expected: {mpath}"
            )
        self.clf = joblib.load(mpath)
        self.encoder = joblib.load(lpath)

        # Load metadata sidecar (written by train_model.py). Backward-compat:
        # if the sidecar doesn't exist we assume the model is real (legacy).
        import json
        mdata_path = metadata_path(language)
        if os.path.exists(mdata_path):
            with open(mdata_path, "r") as f:
                self.metadata: dict = json.load(f)
        else:
            self.metadata = {"data_source": "unknown", "synthetic": False}

        self._window: deque = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
        self._held_label: Optional[str] = None
        self._held_count: int = 0
        self._cooldown: int = 0
        
        # Make these dynamic
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.hold_frames_to_commit = HOLD_FRAMES_TO_COMMIT

    @property
    def is_synthetic(self) -> bool:
        """True if this model was trained on synthetic/generated data."""
        return bool(self.metadata.get("synthetic", False))

    def raw_predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """Single-frame prediction with confidence, no smoothing/state."""
        proba = self.clf.predict_proba([feature_vector])[0]
        best_idx = int(np.argmax(proba))
        label = self.encoder.inverse_transform([best_idx])[0]
        confidence = float(proba[best_idx])
        return label, confidence

    def update(self, feature_vector: Optional[np.ndarray]) -> dict:
        """
        Feed one frame's feature vector (or None if no hand detected).
        Returns a dict describing the current recognition state:
            {
              "smoothed_label": str | None,
              "confidence": float,
              "committed_label": str | None,   # set only on the frame a sign is "typed"
              "progress": float,                # 0..1 hold-to-commit progress bar
            }
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if feature_vector is None:
            self._window.clear()
            self._held_label = None
            self._held_count = 0
            return {"smoothed_label": None, "confidence": 0.0, "committed_label": None, "progress": 0.0}

        label, confidence = self.raw_predict(feature_vector)
        if confidence < self.confidence_threshold:
            label = None

        self._window.append(label)
        smoothed_label, votes = self._majority_vote()
        smoothed_confidence = votes / max(len(self._window), 1)

        committed = None
        progress = 0.0

        if smoothed_label is None or self._cooldown > 0:
            self._held_label = None
            self._held_count = 0
        else:
            if smoothed_label == self._held_label:
                self._held_count += 1
            else:
                self._held_label = smoothed_label
                self._held_count = 1

            progress = min(self._held_count / self.hold_frames_to_commit, 1.0)

            if self._held_count >= self.hold_frames_to_commit:
                committed = smoothed_label
                self._held_count = 0
                self._held_label = None
                self._cooldown = COOLDOWN_FRAMES_AFTER_COMMIT
                self._window.clear()

        return {
            "smoothed_label": smoothed_label,
            "confidence": round(smoothed_confidence, 2),
            "committed_label": committed,
            "progress": round(progress, 2),
        }

    def _majority_vote(self) -> Tuple[Optional[str], int]:
        valid = [l for l in self._window if l is not None]
        if not valid:
            return None, 0
        counts = Counter(valid)
        label, votes = counts.most_common(1)[0]
        return label, votes

    def reset(self):
        self._window.clear()
        self._held_label = None
        self._held_count = 0
        self._cooldown = 0
