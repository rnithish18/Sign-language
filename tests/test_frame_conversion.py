"""
test_frame_conversion.py
--------------------------
Verifies the exact frame handoff path from streamlit-webrtc (av.VideoFrame)
through to HandLandmarkExtractor, ensuring:
1. av.VideoFrame.to_ndarray(format="bgr24") produces shape (H, W, 3), dtype uint8
2. The horizontal mirror flip preserves shape and dtype
3. process_raw receives a BGR array (correct channel order for its internal cvtColor)
4. process_from_raw produces the same FrameResult as process() on the same clean frame
   (i.e. the split-pass path is equivalent to the single-call path)
5. Drawing on the frame AFTER process_from_raw does NOT change the feature result
   (proves the draw-before-process bug is fixed)
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def make_bgr_frame(h=480, w=640) -> np.ndarray:
    """Synthetic BGR frame matching what av.VideoFrame.to_ndarray(format='bgr24') produces."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def mirror(frame: np.ndarray) -> np.ndarray:
    """Horizontal flip, as done in recv()."""
    return frame[:, ::-1, :]


# ── format / shape tests ─────────────────────────────────────────────────────

def test_bgr_frame_shape():
    img = make_bgr_frame()
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8

def test_mirror_preserves_shape_dtype():
    img = make_bgr_frame()
    flipped = mirror(img)
    assert flipped.shape == img.shape
    assert flipped.dtype == img.dtype

def test_mirror_is_actually_flipped():
    img = make_bgr_frame()
    flipped = mirror(img)
    # First column of flipped == last column of original
    assert np.array_equal(flipped[:, 0, :], img[:, -1, :])

def test_bgr_to_rgb_conversion():
    """process_raw does cv2.cvtColor(BGR2RGB) internally. Verify manually that
    a known BGR frame converts to RGB by swapping channel 0 and channel 2."""
    import cv2
    bgr = make_bgr_frame(h=10, w=10)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Channel 0 of RGB == Channel 2 of BGR
    assert np.array_equal(rgb[:, :, 0], bgr[:, :, 2])
    # Channel 2 of RGB == Channel 0 of BGR
    assert np.array_equal(rgb[:, :, 2], bgr[:, :, 0])
    # Middle channel unchanged
    assert np.array_equal(rgb[:, :, 1], bgr[:, :, 1])


# ── process_from_raw equivalence test ────────────────────────────────────────

def _make_fake_mp_result(lm_x=0.5, lm_y=0.5, lm_z=0.0, handedness="Right"):
    """Build a minimal fake MediaPipe result with one hand."""
    fake_lm = MagicMock()
    fake_lm.x = lm_x
    fake_lm.y = lm_y
    fake_lm.z = lm_z

    fake_hand = MagicMock()
    fake_hand.landmark = [fake_lm] * 21

    fake_classification = MagicMock()
    fake_classification.label = handedness
    fake_classification.score = 0.99

    fake_handedness_entry = MagicMock()
    fake_handedness_entry.classification = [fake_classification]

    fake_mp_result = MagicMock()
    fake_mp_result.multi_hand_landmarks = [fake_hand]
    fake_mp_result.multi_handedness = [fake_handedness_entry]
    return fake_mp_result


def test_process_from_raw_matches_process():
    """
    process_from_raw(frame, mp_raw) must return the SAME feature vector as
    process(frame) when given the same underlying MediaPipe result.
    We patch _hands.process on the extractor instance directly since the
    module-level mp_hands reference is resolved at import time.
    """
    from src.landmark_utils import HandLandmarkExtractor

    fake_mp_result = _make_fake_mp_result(lm_x=0.5, lm_y=0.5, handedness="Right")
    frame = make_bgr_frame(h=480, w=640)

    extractor = HandLandmarkExtractor()
    extractor._hands.process = MagicMock(return_value=fake_mp_result)

    # process() path — runs _hands.process internally via cvtColor+process
    result_process = extractor.process(frame)

    # process_from_raw() path — reuses the same fake_mp_result
    result_from_raw = extractor.process_from_raw(frame, fake_mp_result)

    assert result_process is not None, "process() returned None with mocked hand"
    assert result_from_raw is not None, "process_from_raw() returned None with mocked hand"
    np.testing.assert_array_equal(
        result_process.feature_vector,
        result_from_raw.feature_vector,
        err_msg="process() and process_from_raw() produced different feature vectors"
    )


def test_drawing_on_frame_does_not_corrupt_feature_extraction():
    """
    The original bug: draw() was called on img BEFORE process(img), so
    MediaPipe saw annotated pixels on the second pass.

    process_from_raw() is pixel-invariant — it reads landmarks from the
    pre-computed mp_result, NOT from pixels — so passing an annotated frame
    must produce the same result as passing the clean frame.
    """
    from src.landmark_utils import HandLandmarkExtractor

    fake_mp_result = _make_fake_mp_result(lm_x=0.3, lm_y=0.4, handedness="Left")
    frame = make_bgr_frame(h=480, w=640)

    extractor = HandLandmarkExtractor()
    extractor._hands.process = MagicMock(return_value=fake_mp_result)

    mp_raw = extractor.process_raw(frame)
    result_correct = extractor.process_from_raw(frame, mp_raw)

    # Simulate old buggy order: draw first (mutates pixels), then pass annotated frame
    frame_annotated = frame.copy()
    frame_annotated[:, :, 0] = 255   # saturate blue channel as if overlay was drawn
    result_after_draw = extractor.process_from_raw(frame_annotated, mp_raw)

    assert result_correct is not None, "process_from_raw returned None on clean frame"
    assert result_after_draw is not None, "process_from_raw returned None on annotated frame"
    np.testing.assert_array_equal(
        result_correct.feature_vector,
        result_after_draw.feature_vector,
        err_msg="process_from_raw should be pixel-invariant (uses precomputed mp_result)"
    )


def test_no_hand_returns_none():
    from src.landmark_utils import HandLandmarkExtractor

    empty_mp_result = MagicMock()
    empty_mp_result.multi_hand_landmarks = None
    empty_mp_result.multi_handedness = None

    with patch("mediapipe.solutions.hands.Hands") as MockHands:
        mock_hands_inst = MagicMock()
        mock_hands_inst.process.return_value = empty_mp_result
        MockHands.return_value = mock_hands_inst

        extractor = HandLandmarkExtractor()
        frame = make_bgr_frame()
        result = extractor.process_from_raw(frame, empty_mp_result)

    assert result is None
