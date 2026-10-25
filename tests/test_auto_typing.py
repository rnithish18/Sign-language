"""
tests/test_auto_typing.py
-------------------------
Tests for auto-typing and hold-to-commit logic in src/auto_typer.py.

Verifies:
(a) a 1 s hold of C types one "C"
(b) a 0.3 s flicker types nothing
(c) holding C for 3 s types one "C"
(d) C, hand away, C types "CC"
(e) SPACE, CLEAR and DEL behave correctly
(f) confidence below MIN_CONF types nothing
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.auto_typer import AutoTyper
from src.config import HOLD_SECONDS, MIN_CONF, HAND_LOST_TIMEOUT_SECONDS
from src.text_buffer import TextBuffer


def test_hold_1s_types_one_c():
    """(a) a 1 s hold of C types one 'C'."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=HOLD_SECONDS, min_confidence=MIN_CONF)

    # 1.0 second hold, sampled at 0.1s intervals (11 steps: 0.0 to 1.0)
    for i in range(11):
        t = i * 0.1
        typer.process(label="C", confidence=0.95, timestamp=t)

    assert buf.sentence == "C"


def test_flicker_0_3s_types_nothing():
    """(b) a 0.3 s flicker types nothing."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=HOLD_SECONDS, min_confidence=MIN_CONF)

    # 0.3 second flicker: 0.0, 0.1, 0.2, 0.3
    for i in range(4):
        t = i * 0.1
        typer.process(label="C", confidence=0.95, timestamp=t)

    # Hand lost at 0.4s
    typer.process(label=None, confidence=0.0, timestamp=0.4)

    assert buf.sentence == ""


def test_hold_3s_types_one_c():
    """(c) holding C for 3 s types one 'C' (no repeat spam CCCCC)."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=HOLD_SECONDS, min_confidence=MIN_CONF)

    # 3.0 second hold: 0.0 to 3.0s in 0.1s steps (31 steps)
    for i in range(31):
        t = i * 0.1
        typer.process(label="C", confidence=0.95, timestamp=t)

    assert buf.sentence == "C"


def test_c_hand_away_c_types_cc():
    """(d) C, hand away (>=0.5s), C types 'CC'."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(
        text_buffer=buf,
        hold_seconds=HOLD_SECONDS,
        min_confidence=MIN_CONF,
        hand_lost_timeout=HAND_LOST_TIMEOUT_SECONDS,
    )

    # Hold C for 0.8s (0.0 to 0.8) -> types first "C"
    for i in range(9):
        t = i * 0.1
        typer.process(label="C", confidence=0.95, timestamp=t)

    assert buf.sentence == "C"

    # Hand away for 0.6s (0.9 to 1.5) -> exceeds 0.5s timeout
    for i in range(9, 16):
        t = i * 0.1
        typer.process(label=None, confidence=0.0, timestamp=t)

    # Hold C again for 0.8s (1.6 to 2.4)
    for i in range(16, 25):
        t = i * 0.1
        typer.process(label="C", confidence=0.95, timestamp=t)

    assert buf.sentence == "CC"


def test_space_clear_and_del_behave_correctly():
    """(e) SPACE, CLEAR and DEL behave correctly."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=HOLD_SECONDS, min_confidence=MIN_CONF)

    # 1. Type "A" (0.0 to 0.8s)
    for i in range(9):
        typer.process("A", 0.95, timestamp=i * 0.1)
    assert buf.sentence == "A"

    # Hand away briefly (0.9 to 1.0)
    for i in range(9, 11):
        typer.process(None, 0.0, timestamp=i * 0.1)

    # 2. Type "B" (1.1 to 1.9s)
    for i in range(11, 20):
        typer.process("B", 0.95, timestamp=i * 0.1)
    assert buf.sentence == "AB"

    # Hand away briefly (2.0 to 2.1)
    for i in range(20, 22):
        typer.process(None, 0.0, timestamp=i * 0.1)

    # 3. Type "DEL" (2.2 to 3.0s) -> deletes "B", leaves "A"
    for i in range(22, 31):
        typer.process("DEL", 0.95, timestamp=i * 0.1)
    assert buf.sentence == "A"

    # Hand away briefly (3.1 to 3.2)
    for i in range(31, 33):
        typer.process(None, 0.0, timestamp=i * 0.1)

    # 4. Type "SPACE" (3.3 to 4.1s) -> finishes word "A"
    for i in range(33, 42):
        typer.process("SPACE", 0.95, timestamp=i * 0.1)
    assert buf.current_word == ""

    # Hand away briefly (4.2 to 4.3)
    for i in range(42, 44):
        typer.process(None, 0.0, timestamp=i * 0.1)

    # 5. Type "B" (4.4 to 5.2s) -> sentence is now "A B"
    for i in range(44, 53):
        typer.process("B", 0.95, timestamp=i * 0.1)
    assert buf.sentence == "A B"

    # Hand away briefly (5.3 to 5.4)
    for i in range(53, 55):
        typer.process(None, 0.0, timestamp=i * 0.1)

    # 6. Type "CLEAR" (5.5 to 6.3s) -> wipes everything
    for i in range(55, 64):
        typer.process("CLEAR", 0.95, timestamp=i * 0.1)
    assert buf.sentence == ""


def test_confidence_below_min_conf_types_nothing():
    """(f) confidence below MIN_CONF types nothing."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=HOLD_SECONDS, min_confidence=MIN_CONF)

    # Hold C for 1.5s with confidence 0.75 (< 0.80)
    for i in range(16):
        t = i * 0.1
        typer.process(label="C", confidence=0.75, timestamp=t)

    assert buf.sentence == ""


def test_fake_frame_processing():
    """Test process_frame with mock predictor and fake landmark vector."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(text_buffer=buf, hold_seconds=0.8, min_confidence=0.80)

    mock_pred = MagicMock()
    mock_pred.raw_predict.return_value = ("H", 0.92)

    dummy_vector = np.zeros(128, dtype=np.float32)

    # 0.8s hold of fake frame
    for i in range(9):
        typer.process_frame(dummy_vector, mock_pred, timestamp=i * 0.1)

    assert buf.sentence == "H"


def test_ignore_needs_calibration():
    """Uncalibrated gestures (in needs_calibration) must not be auto-typed."""
    buf = TextBuffer(auto_correct=False)
    typer = AutoTyper(
        text_buffer=buf,
        hold_seconds=0.8,
        min_confidence=0.80,
        needs_calibration={"PHRASE_HELP", "PHRASE_WATER"},
    )

    for i in range(10):
        typer.process("PHRASE_HELP", 0.99, timestamp=i * 0.1)

    assert buf.sentence == ""
