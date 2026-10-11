"""
test_array_contiguity.py
--------------------------
Verifies that the recv() contiguity fix correctly handles non-contiguous
and/or read-only numpy arrays — the exact scenario that av.VideoFrame.to_ndarray()
produces and that caused cv2.rectangle to throw:

    cv2.error: (-5:Bad argument) in function 'rectangle'
    > Layout of the output array img is incompatible with cv::Mat

We don't need a live av.VideoFrame; we reproduce the exact failure condition
with pure numpy (non-contiguous arrays via slicing/transposing) and confirm
that the fix (np.ascontiguousarray + writability guard) makes them safe for
OpenCV drawing operations.
"""
import numpy as np
import cv2
import pytest


# ---------------------------------------------------------------------------
# Helpers — construct arrays that mimic av.VideoFrame.to_ndarray() pathology
# ---------------------------------------------------------------------------

def make_non_contiguous_via_slice() -> np.ndarray:
    """Reverse-stride slicing ([:, ::-1, :]) is exactly what the mirror flip
    does, and is the most common way av frames become non-contiguous."""
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    return base[:, ::-1, :]


def make_non_contiguous_via_transpose() -> np.ndarray:
    """Transposing then slicing back to (H, W, 3) produces a non-contiguous
    array with non-trivial strides."""
    base = np.zeros((3, 480, 640), dtype=np.uint8)
    return np.transpose(base, (1, 2, 0))


def make_readonly_array() -> np.ndarray:
    """Simulates a read-only array (some av backends return this)."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr.flags.writeable = False
    return arr


def make_non_contiguous_and_readonly() -> np.ndarray:
    """Worst-case: both non-contiguous AND read-only."""
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    view = base[:, ::-1, :]
    # Can't set writeable=False on a non-contiguous array directly in all
    # numpy versions, so we use base's flag before slicing.
    base.flags.writeable = False
    return base[:, ::-1, :]


def apply_contiguity_fix(img: np.ndarray) -> np.ndarray:
    """The exact fix applied in app.py recv()."""
    img = np.ascontiguousarray(img)
    if not img.flags.writeable:
        img = img.copy()
    return img


# ---------------------------------------------------------------------------
# Tests: verify the fix produces a contiguous, writable array
# ---------------------------------------------------------------------------

class TestContiguityFix:
    """Verify that apply_contiguity_fix makes any array safe for cv2 drawing."""

    @pytest.mark.parametrize("factory,description", [
        (make_non_contiguous_via_slice, "reverse-stride slice (mirror flip)"),
        (make_non_contiguous_via_transpose, "transposed array"),
        (make_readonly_array, "read-only array"),
        (make_non_contiguous_and_readonly, "non-contiguous + read-only"),
    ])
    def test_fix_produces_contiguous_writable(self, factory, description):
        """After the fix, the array must be C-contiguous and writable."""
        img = factory()
        fixed = apply_contiguity_fix(img)
        assert fixed.flags["C_CONTIGUOUS"], \
            f"Array not contiguous after fix ({description})"
        assert fixed.flags.writeable, \
            f"Array not writable after fix ({description})"
        assert fixed.shape == img.shape, \
            f"Shape changed after fix ({description})"
        assert fixed.dtype == img.dtype, \
            f"dtype changed after fix ({description})"

    @pytest.mark.parametrize("factory,description", [
        (make_non_contiguous_via_slice, "reverse-stride slice (mirror flip)"),
        (make_non_contiguous_via_transpose, "transposed array"),
        (make_readonly_array, "read-only array"),
        (make_non_contiguous_and_readonly, "non-contiguous + read-only"),
    ])
    def test_cv2_rectangle_succeeds_after_fix(self, factory, description):
        """cv2.rectangle must NOT throw after the fix — this is the actual
        failing call from the bug report."""
        img = factory()
        fixed = apply_contiguity_fix(img)
        h, w = fixed.shape[:2]
        # This is the exact call from _draw_overlay that was crashing:
        cv2.rectangle(fixed, (0, h - 60), (300, h), (20, 30, 50), -1)
        # If we get here without exception, the fix works.

    @pytest.mark.parametrize("factory,description", [
        (make_non_contiguous_via_slice, "reverse-stride slice (mirror flip)"),
        (make_non_contiguous_via_transpose, "transposed array"),
        (make_readonly_array, "read-only array"),
        (make_non_contiguous_and_readonly, "non-contiguous + read-only"),
    ])
    def test_cv2_putText_succeeds_after_fix(self, factory, description):
        """cv2.putText also mutates the array and would fail on
        non-contiguous input."""
        img = factory()
        fixed = apply_contiguity_fix(img)
        cv2.putText(fixed, "test", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 60), 1)

    def test_already_contiguous_writable_is_noop(self):
        """A normal contiguous writable array should pass through cheaply."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        fixed = apply_contiguity_fix(img)
        assert fixed.flags["C_CONTIGUOUS"]
        assert fixed.flags.writeable
        # Should share memory (no copy needed)
        assert np.shares_memory(img, fixed)


class TestOriginalArrayFailsWithoutFix:
    """Confirm the bug: cv2.rectangle DOES fail on the unfixed array."""

    def test_non_contiguous_slice_fails_cv2_rectangle(self):
        """Without the fix, cv2.rectangle throws on a reverse-strided array."""
        img = make_non_contiguous_via_slice()
        assert not img.flags["C_CONTIGUOUS"], \
            "Precondition: array should be non-contiguous"
        h, w = img.shape[:2]
        with pytest.raises(cv2.error):
            cv2.rectangle(img, (0, h - 60), (300, h), (20, 30, 50), -1)

    def test_readonly_array_fails_cv2_rectangle(self):
        """Without the fix, cv2.rectangle throws on a read-only array."""
        img = make_readonly_array()
        assert not img.flags.writeable, \
            "Precondition: array should be read-only"
        h, w = img.shape[:2]
        with pytest.raises((cv2.error, ValueError)):
            cv2.rectangle(img, (0, h - 60), (300, h), (20, 30, 50), -1)


class TestMirrorFlipContiguity:
    """Verify that the double-fix pattern (before + after mirror) works
    end-to-end, matching the actual recv() code path."""

    def test_full_recv_path_simulation(self):
        """Simulate the full recv() array path:
        to_ndarray -> ascontiguousarray -> mirror -> ascontiguousarray -> draw
        """
        # Simulate av.VideoFrame.to_ndarray returning a non-contiguous array
        raw = make_non_contiguous_via_slice()
        assert not raw.flags["C_CONTIGUOUS"]

        # First fix (post-to_ndarray)
        img = np.ascontiguousarray(raw)
        if not img.flags.writeable:
            img = img.copy()
        assert img.flags["C_CONTIGUOUS"]
        assert img.flags.writeable

        # Mirror flip (creates non-contiguous view again)
        img = img[:, ::-1, :]
        assert not img.flags["C_CONTIGUOUS"]

        # Second fix (post-mirror)
        img = np.ascontiguousarray(img)
        assert img.flags["C_CONTIGUOUS"]
        assert img.flags.writeable

        # Now all drawing calls should work
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 60), (300, h), (20, 30, 50), -1)
        cv2.putText(img, "Hold to confirm", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
