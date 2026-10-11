"""
test_phrase_shortcuts.py
--------------------------
Phase A tests:
 - Phrase labels present in training data CSV
 - Phrase insertion into TextBuffer as a single unit
 - Backspace removes whole phrase in one press
 - SOS phrase triggers sos_triggered flag and speak_requested
 - Non-SOS phrases do not trigger sos_triggered
 - Phrase bypasses spellcheck
"""

import csv
import os
import pytest

from src.config import PHRASE_LABELS, PHRASE_GESTURES, SOS_PHRASE_LABEL, FEATURE_LENGTH
from src.text_buffer import TextBuffer


# ---------------------------------------------------------------------------
# Training data labels
# ---------------------------------------------------------------------------

def test_phrase_labels_in_asl_dataset():
    """All PHRASE_GESTURES must be rows in the ASL training CSV."""
    from src.config import dataset_csv_path
    csv_path = dataset_csv_path("ASL")
    if not os.path.exists(csv_path):
        pytest.skip("ASL dataset not generated yet")
    with open(csv_path, newline="") as f:
        labels_in_csv = {row[0] for row in csv.reader(f) if row}
    for phrase_key in PHRASE_GESTURES:
        assert phrase_key in labels_in_csv, f"{phrase_key} missing from ASL dataset"


def test_phrase_labels_in_isl_dataset():
    """All PHRASE_GESTURES must be rows in the ISL training CSV."""
    from src.config import dataset_csv_path
    csv_path = dataset_csv_path("ISL")
    if not os.path.exists(csv_path):
        pytest.skip("ISL dataset not generated yet")
    with open(csv_path, newline="") as f:
        labels_in_csv = {row[0] for row in csv.reader(f) if row}
    for phrase_key in PHRASE_GESTURES:
        assert phrase_key in labels_in_csv, f"{phrase_key} missing from ISL dataset"


# ---------------------------------------------------------------------------
# TextBuffer phrase behavior
# ---------------------------------------------------------------------------

def test_phrase_inserted_as_unit():
    buf = TextBuffer()
    buf.handle_label("PHRASE_THANK_YOU")
    assert buf.sentence == PHRASE_LABELS["PHRASE_THANK_YOU"]
    assert len(buf.finished_sentence) == 1
    assert buf._is_phrase == [True]


def test_phrase_flushes_current_word_first():
    buf = TextBuffer(auto_correct=False)
    buf.current_word = "HEL"
    buf.handle_label("PHRASE_YES")
    # "HEL" should be committed first, then phrase
    assert buf.finished_sentence[0] == "HEL"
    assert buf.finished_sentence[1] == PHRASE_LABELS["PHRASE_YES"]


def test_backspace_removes_whole_phrase():
    buf = TextBuffer()
    buf.handle_label("PHRASE_WATER")
    assert len(buf.finished_sentence) == 1
    buf.handle_label("BACKSPACE")
    assert len(buf.finished_sentence) == 0
    assert buf.current_word == ""


def test_backspace_after_phrase_then_letter():
    """After phrase, adding a letter and pressing backspace removes letter, not phrase."""
    buf = TextBuffer(auto_correct=False)
    buf.handle_label("PHRASE_NO")
    buf.handle_label("A")
    buf.handle_label("BACKSPACE")  # removes "A"
    assert buf.current_word == ""
    assert buf.finished_sentence == [PHRASE_LABELS["PHRASE_NO"]]


def test_phrase_bypasses_spellcheck():
    """Phrase text must appear verbatim — not mangled by autocorrect."""
    buf = TextBuffer(auto_correct=True)
    buf.handle_label("PHRASE_WASHROOM")
    assert buf.finished_sentence[0] == PHRASE_LABELS["PHRASE_WASHROOM"]


def test_non_sos_phrase_no_sos_flag():
    buf = TextBuffer()
    buf.handle_label("PHRASE_THANK_YOU")
    assert buf.sos_triggered is False


def test_sos_phrase_sets_sos_and_speak():
    buf = TextBuffer()
    buf.handle_label(SOS_PHRASE_LABEL)
    assert buf.sos_triggered is True
    assert buf.speak_requested is True


def test_consume_sos_resets_flag():
    buf = TextBuffer()
    buf.handle_label(SOS_PHRASE_LABEL)
    assert buf.consume_sos() is True
    assert buf.consume_sos() is False   # second call returns False


def test_clear_resets_sos():
    buf = TextBuffer()
    buf.handle_label(SOS_PHRASE_LABEL)
    buf.clear()
    assert buf.sos_triggered is False
    assert buf.last_phrase is None
