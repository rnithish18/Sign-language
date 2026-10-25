"""
test_verify_labels.py
---------------------
Unit tests for VerificationSession (src/verify_labels.py).

No camera, no OpenCV window, no network.  All tests run by injecting
scripted key sequences into session.handle_key() and inspecting the
resulting state + written files.
"""

import os
import tempfile
import pytest

from src.verify_labels import (
    VerificationSession,
    PASS_KEYS, FAIL_KEYS, SKIP_KEYS, QUIT_KEYS,
    LabelResult,
)

# Convenience single-byte key values
KEY_Y = ord("y")
KEY_N = ord("n")
KEY_S = ord("s")
KEY_Q = ord("q")
KEY_ESC = 27


# ---------------------------------------------------------------------------
# LabelResult unit tests
# ---------------------------------------------------------------------------

class TestLabelResult:
    def test_initial_state(self):
        r = LabelResult("A")
        assert r.outcome is None
        assert r.misrecognised_as is None

    def test_mark_pass(self):
        r = LabelResult("A")
        r.mark_pass()
        assert r.outcome == "PASS"
        assert r.misrecognised_as is None

    def test_mark_fail_with_prediction(self):
        r = LabelResult("A")
        r.mark_fail("B")
        assert r.outcome == "FAIL"
        assert r.misrecognised_as == "B"

    def test_mark_fail_without_prediction(self):
        r = LabelResult("A")
        r.mark_fail(None)
        assert r.outcome == "FAIL"
        assert r.misrecognised_as is None

    def test_mark_skip(self):
        r = LabelResult("A")
        r.mark_skip()
        assert r.outcome == "SKIP"

    def test_as_row_pass(self):
        r = LabelResult("SPACE")
        r.mark_pass()
        row = r.as_row()
        assert "SPACE" in row
        assert "PASS" in row

    def test_as_row_fail_misrecog(self):
        r = LabelResult("A")
        r.mark_fail("B")
        row = r.as_row()
        assert "FAIL" in row
        assert "B" in row

    def test_mark_fail_no_hand(self):
        r = LabelResult("A")
        r.mark_fail("NO_HAND")
        assert r.outcome == "FAIL"
        assert r.no_hand is True
        assert r.misrecognised_as == "NO_HAND"
        assert "no hand detected" in r.as_row().lower()


# ---------------------------------------------------------------------------
# VerificationSession state-machine tests
# ---------------------------------------------------------------------------

class TestVerificationSession:
    LABELS = ["A", "B", "C"]

    def _session(self, labels=None, **kwargs):
        return VerificationSession(labels or self.LABELS, "ASL", **kwargs)

    # Construction
    def test_initial_label(self):
        s = self._session()
        assert s.current_label == "A"
        assert s._index == 0
        assert not s.is_complete
        assert not s.aborted

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError):
            VerificationSession([], "ASL")

    def test_on_advance_called_at_start(self):
        calls = []
        self._session(on_advance=lambda lbl, i, t: calls.append((lbl, i, t)))
        assert calls == [("A", 0, 3)]

    # y → advance
    def test_pass_advances(self):
        s = self._session()
        s.handle_key(KEY_Y)
        assert s._index == 1
        assert s.current_label == "B"
        assert s.results[0].outcome == "PASS"

    # n → advance with misrecog recorded
    def test_fail_records_prediction(self):
        s = self._session()
        s.handle_key(KEY_N, current_prediction="Z")
        assert s.results[0].outcome == "FAIL"
        assert s.results[0].misrecognised_as == "Z"
        assert s._index == 1

    # s → advance, mark skip
    def test_skip_advances(self):
        s = self._session()
        s.handle_key(KEY_S)
        assert s.results[0].outcome == "SKIP"
        assert s._index == 1

    # q → aborted
    def test_quit_aborts(self):
        s = self._session()
        result = s.handle_key(KEY_Q)
        assert s.aborted
        assert s.is_complete
        assert result is False

    # ESC → aborted
    def test_esc_aborts(self):
        s = self._session()
        s.handle_key(KEY_ESC)
        assert s.aborted

    # Full run through all labels
    def test_full_pass_run(self):
        s = self._session()
        for _ in self.LABELS:
            assert not s.is_complete
            s.handle_key(KEY_Y)
        assert s.is_complete
        assert all(r.outcome == "PASS" for r in s.results)

    def test_mixed_run(self):
        # A=PASS, B=FAIL(misrecog=X), C=SKIP
        s = self._session()
        s.handle_key(KEY_Y)              # A → PASS
        s.handle_key(KEY_N, "X")         # B → FAIL, misrecog=X
        s.handle_key(KEY_S)              # C → SKIP
        assert s.is_complete
        assert s.results[0].outcome == "PASS"
        assert s.results[1].outcome == "FAIL"
        assert s.results[1].misrecognised_as == "X"
        assert s.results[2].outcome == "SKIP"

    # on_advance callback progression
    def test_on_advance_sequence(self):
        advances = []
        s = self._session(
            on_advance=lambda lbl, i, t: advances.append(lbl)
        )
        s.handle_key(KEY_Y)  # A done → advance to B
        s.handle_key(KEY_Y)  # B done → advance to C
        # After C is done the session is complete → no more advance
        s.handle_key(KEY_Y)
        assert advances == ["A", "B", "C"]

    # handle_key after completion is a no-op
    def test_handle_key_after_complete_is_noop(self):
        s = VerificationSession(["A"], "ASL")
        s.handle_key(KEY_Y)
        assert s.is_complete
        # Further key presses should return False and not crash
        result = s.handle_key(KEY_Y)
        assert result is False

    # Capital letters work too
    def test_capital_y_passes(self):
        s = self._session()
        s.handle_key(ord("Y"))
        assert s.results[0].outcome == "PASS"

    def test_capital_n_fails(self):
        s = self._session()
        s.handle_key(ord("N"), "M")
        assert s.results[0].outcome == "FAIL"


# ---------------------------------------------------------------------------
# summary_lines / write_results
# ---------------------------------------------------------------------------

class TestSummaryAndOutput:
    LABELS = ["A", "B", "C", "D"]

    def test_summary_lines_counts(self):
        s = VerificationSession(self.LABELS, "ASL")
        s.handle_key(KEY_Y)      # A → PASS
        s.handle_key(KEY_N, "Z") # B → FAIL
        s.handle_key(KEY_S)      # C → SKIP
        # D left pending (aborted via q)
        s.handle_key(KEY_Q)

        lines = s.summary_lines()
        text = "\n".join(lines)
        assert "PASS: 1" in text
        assert "FAIL: 1" in text
        assert "SKIP: 1" in text
        assert "aborted" in text.lower()

    def test_summary_lines_all_pass(self):
        s = VerificationSession(["X", "Y"], "ISL")
        s.handle_key(KEY_Y)
        s.handle_key(KEY_Y)
        lines = "\n".join(s.summary_lines())
        assert "PASS: 2" in lines
        assert "FAIL: 0" in lines

    def test_summary_lines_separates_no_hand_from_misrecog(self):
        s = VerificationSession(["A", "B", "C"], "ASL")
        s.handle_key(KEY_Y)
        s.handle_key(KEY_N, "NO_HAND")
        s.handle_key(KEY_N, "Z")
        text = "\n".join(s.summary_lines())
        assert "misrecognised: 1" in text
        assert "no hand: 1" in text
        assert "NO_HAND (no hand detected)" in text

    def test_write_results_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = VerificationSession(["A", "B"], "ASL")
            s.handle_key(KEY_Y)
            s.handle_key(KEY_N, "C")
            fpath = s.write_results(tmpdir)
            assert os.path.isfile(fpath), f"Expected file at {fpath}"
            content = open(fpath, encoding="utf-8").read()
            assert "PASS" in content
            assert "FAIL" in content

    def test_write_results_filename_contains_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = VerificationSession(["A"], "ISL")
            s.handle_key(KEY_Y)
            fpath = s.write_results(tmpdir)
            assert "isl" in os.path.basename(fpath).lower()

    def test_write_results_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as base:
            out = os.path.join(base, "nested", "results")
            s = VerificationSession(["A"], "ASL")
            s.handle_key(KEY_Y)
            fpath = s.write_results(out)
            assert os.path.isdir(out)
            assert os.path.isfile(fpath)
