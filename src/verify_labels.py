"""
verify_labels.py
----------------
State machine for the per-label live verification checklist.

Deliberately contains *no* camera / OpenCV / cv2 calls so it can be unit-
tested without a display or webcam.  All I/O is injected through callbacks.

Typical use:
    session = VerificationSession(labels, language)
    key = read_key_from_user()
    session.handle_key(key, current_prediction)
    if session.is_complete:
        session.write_results(output_dir)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, List, Optional


PASS_KEYS = frozenset([ord("y"), ord("Y")])
FAIL_KEYS = frozenset([ord("n"), ord("N")])
SKIP_KEYS = frozenset([ord("s"), ord("S")])
QUIT_KEYS = frozenset([ord("q"), ord("Q"), 27])  # 27 = ESC


class LabelResult:
    """Record for one tested label."""

    def __init__(self, label: str):
        self.label = label
        self.outcome: Optional[str] = None        # "PASS" | "FAIL" | "SKIP"
        self.misrecognised_as: Optional[str] = None

    def mark_pass(self):
        self.outcome = "PASS"
        self.misrecognised_as = None

    def mark_fail(self, prediction: Optional[str] = None):
        self.outcome = "FAIL"
        self.misrecognised_as = prediction  # None if no hand / unclear

    def mark_skip(self):
        self.outcome = "SKIP"
        self.misrecognised_as = None

    def as_row(self) -> str:
        mis = self.misrecognised_as or "—"
        return f"{self.label:<20} {self.outcome or 'PENDING':<6}  misrecognised_as={mis}"


class VerificationSession:
    """
    Drives a label-by-label verification checklist.

    Parameters
    ----------
    labels : list[str]
        Ordered list of labels to test (e.g. ['A', 'B', ..., 'SPACE', ...]).
    language : str
        Used only for the output filename (e.g. "ASL").
    on_advance : optional callable(current_label, index, total)
        Called each time the session advances to a new label (useful for
        printing a prompt to stdout in the CLI wrapper).
    """

    def __init__(
        self,
        labels: List[str],
        language: str = "ASL",
        on_advance: Optional[Callable[[str, int, int], None]] = None,
    ):
        if not labels:
            raise ValueError("labels must be non-empty")
        self.labels = list(labels)
        self.language = language
        self.on_advance = on_advance
        self._index = 0
        self.results: List[LabelResult] = [LabelResult(l) for l in labels]
        self.aborted = False

        # Fire the initial prompt
        if self.on_advance:
            self.on_advance(self.current_label, 0, len(self.labels))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_label(self) -> Optional[str]:
        if self._index < len(self.labels):
            return self.labels[self._index]
        return None

    @property
    def current_result(self) -> Optional[LabelResult]:
        if self._index < len(self.results):
            return self.results[self._index]
        return None

    @property
    def is_complete(self) -> bool:
        return self._index >= len(self.labels) or self.aborted

    # ------------------------------------------------------------------
    # Key handler — returns True if the session should keep going
    # ------------------------------------------------------------------

    def handle_key(self, key: int, current_prediction: Optional[str] = None) -> bool:
        """
        Process a key-press from the caller.

        Returns False when the session should exit (complete or aborted).
        """
        if self.is_complete:
            return False

        if key in PASS_KEYS:
            self.current_result.mark_pass()
            self._advance()
        elif key in FAIL_KEYS:
            self.current_result.mark_fail(current_prediction)
            self._advance()
        elif key in SKIP_KEYS:
            self.current_result.mark_skip()
            self._advance()
        elif key in QUIT_KEYS:
            self.aborted = True
            return False

        return not self.is_complete

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _advance(self):
        self._index += 1
        if not self.is_complete and self.on_advance:
            self.on_advance(self.current_label, self._index, len(self.labels))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_lines(self) -> List[str]:
        """Return the summary as a list of text lines (no file I/O)."""
        tested = [r for r in self.results if r.outcome is not None]
        passed = sum(1 for r in tested if r.outcome == "PASS")
        failed = sum(1 for r in tested if r.outcome == "FAIL")
        skipped = sum(1 for r in tested if r.outcome == "SKIP")

        lines = [
            f"SignBridge — {self.language} Label Verification",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total labels: {len(self.labels)}  |  Tested: {len(tested)}"
            f"  |  PASS: {passed}  FAIL: {failed}  SKIP: {skipped}",
            "",
            f"{'Label':<20} {'Result':<6}  Misrecognised as",
            "-" * 55,
        ]
        for r in self.results:
            lines.append(r.as_row())
        lines += [
            "",
            f"PASS rate (tested): {passed}/{len(tested)}"
            f" = {(passed/len(tested)*100):.0f}%" if tested else "No labels tested.",
        ]
        if self.aborted:
            lines.append("(Session was aborted early — remaining labels not tested.)")
        return lines

    def write_results(self, output_dir: str = "verification_results") -> str:
        """Write summary to a timestamped file.  Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{self.language.lower()}_verification_{stamp}.txt"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.summary_lines()))
        return fpath
