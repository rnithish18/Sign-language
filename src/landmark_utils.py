"""
landmark_utils.py
------------------
Thin wrapper around MediaPipe Hands that turns a BGR camera frame into a
flat, normalized feature vector suitable for a scikit-learn classifier.

Normalization strategy:
1. Translate all landmarks so the wrist (landmark 0) is the origin.
2. Scale by the max distance from the wrist so the feature vector is
   invariant to how close/far the hand is from the camera.
This keeps the model robust to hand size and camera distance, which matters
a lot more than it seems for gesture classifiers trained on small datasets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from exc

mp_hands = mp.solutions.hands if hasattr(mp, "solutions") else None
mp_drawing = mp.solutions.drawing_utils if hasattr(mp, "solutions") else None
mp_drawing_styles = mp.solutions.drawing_styles if hasattr(mp, "solutions") else None

from src.config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    NUM_LANDMARKS,
)

@dataclass
class HandResult:
    """Container for a single detected hand."""
    landmarks_raw: List[Tuple[float, float, float]]  # image-normalized (0-1) coords from MediaPipe
    feature_vector: np.ndarray                        # wrist-centered, scale-normalized, flattened (63-dim)
    handedness: str                                    # "Left" or "Right"
    handedness_score: float

@dataclass
class FrameResult:
    """Container for the whole frame, combining hands into a unified feature vector."""
    hands: List[HandResult]
    feature_vector: np.ndarray  # 128-dim for MAX_NUM_HANDS=2, or 64-dim if MAX_NUM_HANDS=1


class HandLandmarkExtractor:
    """
    Wraps mp.solutions.hands.Hands with sensible defaults and exposes a
    simple `process(frame_bgr)` -> List[HandResult] API.

    Usage:
        extractor = HandLandmarkExtractor()
        results = extractor.process(frame)
        for hand in results:
            model.predict([hand.feature_vector])
        extractor.close()
    """

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        static_image_mode: bool = False,
    ):
        self._hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[FrameResult]:
        """Run MediaPipe on a BGR frame and return normalized features for the frame."""
        import cv2  # local import keeps this module importable without cv2 in pure-inference contexts

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        mp_result = self._hands.process(frame_rgb)

        hands_out: List[HandResult] = []
        if not mp_result.multi_hand_landmarks:
            return None

        handedness_list = mp_result.multi_handedness or []
        for idx, hand_landmarks in enumerate(mp_result.multi_hand_landmarks):
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            feature_vector = normalize_landmarks(raw)

            label, score = "Unknown", 0.0
            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                label, score = classification.label, classification.score

            hands_out.append(
                HandResult(
                    landmarks_raw=raw,
                    feature_vector=feature_vector,
                    handedness=label,
                    handedness_score=score,
                )
            )
        
        # Build the combined feature vector for the frame
        from src.config import MAX_NUM_HANDS
        
        frame_features = []
        # We always process Left then Right (if MAX_NUM_HANDS==2)
        # Or just the first found if MAX_NUM_HANDS==1 (though we pad based on slot)
        
        # Helper to find a hand by handedness
        def get_hand(target_handedness: str) -> Optional[HandResult]:
            # Exact match first
            for h in hands_out:
                if h.handedness == target_handedness:
                    return h
            # Fallback if somehow label is wrong but we need a slot and we only have 1 hand and MAX_NUM_HANDS=1
            # But the user said: "Sort/assign hand slots by MediaPipe's handedness label"
            return None
            
        slots = ["Left", "Right"] if MAX_NUM_HANDS == 2 else ["Left"] # If 1 hand, we just use one slot (assume Left or just first)
        
        if MAX_NUM_HANDS == 1:
            # For 1 hand, we just take the first detected hand, regardless of Left/Right, 
            # to keep backward logic somewhat sane for 1-hand mode (though it gets a presence bit now).
            h = hands_out[0] if hands_out else None
            presence = 1.0 if h else 0.0
            features = h.feature_vector if h else np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
            frame_features.extend([presence])
            frame_features.extend(features.tolist())
        else:
            # 2 hands: strictly Left slot then Right slot
            for slot in slots:
                h = get_hand(slot)
                presence = 1.0 if h else 0.0
                features = h.feature_vector if h else np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
                frame_features.extend([presence])
                frame_features.extend(features.tolist())
                
        return FrameResult(
            hands=hands_out,
            feature_vector=np.array(frame_features, dtype=np.float32)
        )

    def draw(self, frame_bgr: np.ndarray, mp_result) -> np.ndarray:
        """Draw MediaPipe's hand skeleton overlay on the frame (for UI display)."""
        if mp_result and mp_result.multi_hand_landmarks:
            for hand_landmarks in mp_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def process_raw(self, frame_bgr: np.ndarray):
        """Expose the raw MediaPipe result too, useful when the caller wants to draw()."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        return self._hands.process(frame_rgb)

    def process_from_raw(self, frame_bgr: np.ndarray, mp_result) -> Optional[FrameResult]:
        """
        Build a FrameResult from an already-computed MediaPipe mp_result
        (e.g. obtained via process_raw) without running MediaPipe a second time.
        This is the preferred path in the webrtc recv() callback: call
        process_raw() once on the clean frame, pass the result here for
        feature extraction, then call draw() for the overlay.
        """
        if mp_result is None or not mp_result.multi_hand_landmarks:
            return None

        hands_out: List[HandResult] = []
        handedness_list = mp_result.multi_handedness or []
        for idx, hand_landmarks in enumerate(mp_result.multi_hand_landmarks):
            raw = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            feature_vector = normalize_landmarks(raw)

            label, score = "Unknown", 0.0
            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                label, score = classification.label, classification.score

            hands_out.append(
                HandResult(
                    landmarks_raw=raw,
                    feature_vector=feature_vector,
                    handedness=label,
                    handedness_score=score,
                )
            )

        from src.config import MAX_NUM_HANDS

        frame_features = []

        def get_hand(target_handedness: str) -> Optional[HandResult]:
            for h in hands_out:
                if h.handedness == target_handedness:
                    return h
            return None

        if MAX_NUM_HANDS == 1:
            h = hands_out[0] if hands_out else None
            presence = 1.0 if h else 0.0
            features = h.feature_vector if h else np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
            frame_features.extend([presence])
            frame_features.extend(features.tolist())
        else:
            for slot in ["Left", "Right"]:
                h = get_hand(slot)
                presence = 1.0 if h else 0.0
                features = h.feature_vector if h else np.zeros(NUM_LANDMARKS * 3, dtype=np.float32)
                frame_features.extend([presence])
                frame_features.extend(features.tolist())

        return FrameResult(
            hands=hands_out,
            feature_vector=np.array(frame_features, dtype=np.float32)
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_feature_vector(
    frame_bgr: np.ndarray,
    extractor: "HandLandmarkExtractor",
    mirror: bool = True,
) -> "Optional[FrameResult]":
    """
    Canonical single entry point for BGR frame → 128-dim feature vector.

    All three callers (data_collection.py, cli_app.py, app.py recv()) must go
    through this function so that mirror, BGR→RGB conversion, and slot ordering
    are always identical between training-data recording and inference.

    Parameters
    ----------
    frame_bgr : np.ndarray
        C-contiguous, writeable HxWx3 BGR uint8 array exactly as delivered by
        cv2.VideoCapture.read() or av.VideoFrame.to_ndarray(format="bgr24")
        *before* any flip.
    extractor : HandLandmarkExtractor
        Shared extractor instance; must already be initialised.
    mirror : bool
        True (default) = flip horizontally so the image is a selfie-view mirror.
        Must be True for all webcam and webrtc sources.  Set False only for
        static image imports (import_public_dataset) where frames are already
        canonical.

    Returns
    -------
    FrameResult or None
        None if no hand is detected.
    """
    import cv2  # local import — keeps module importable without OpenCV in tests

    if not frame_bgr.flags["C_CONTIGUOUS"] or not frame_bgr.flags.writeable:
        frame_bgr = np.ascontiguousarray(frame_bgr)

    if mirror:
        frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1, :])

    mp_raw = extractor.process_raw(frame_bgr)
    return extractor.process_from_raw(frame_bgr, mp_raw)


def normalize_landmarks(raw_landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Convert MediaPipe's 21 (x, y, z) landmarks into a translation- and
    scale-invariant flat feature vector of length NUM_LANDMARKS * 3.
    """
    pts = np.array(raw_landmarks, dtype=np.float32)  # shape (21, 3)
    if pts.shape[0] != NUM_LANDMARKS:
        # Pad or truncate defensively; should not normally happen.
        pts = np.resize(pts, (NUM_LANDMARKS, 3))

    wrist = pts[0].copy()
    centered = pts - wrist

    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist < 1e-6:
        max_dist = 1e-6
    normalized = centered / max_dist

    return normalized.flatten()


def landmarks_to_csv_row(
    feature_vector: np.ndarray,
    label: str,
    source: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List:
    """Build a CSV row.

    Column order (when all fields present):
        label, source, session_id, f0, ..., fN

    Backward-compatible: if source is None, row is [label, f0, ..., fN].
    If source is provided but session_id is None, row is [label, source, f0, ..., fN].
    """
    if source is not None and session_id is not None:
        return [label, source, session_id] + feature_vector.tolist()
    if source is not None:
        return [label, source] + feature_vector.tolist()
    return [label] + feature_vector.tolist()

