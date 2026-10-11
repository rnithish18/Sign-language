"""
test_sign_image_mapper.py
--------------------------
Phase B tests:
 - Single letter returns image frame or placeholder (no existing images expected)
 - Space returns space marker
 - Non-alpha char returns placeholder
 - Empty string returns empty list
 - Full text sequence has correct length and types
 - Missing image logged as warning, not exception
"""

import logging
import os
import pytest
from unittest.mock import patch

from src.sign_image_mapper import text_to_sign_sequence, letter_to_frame, image_path


# ---------------------------------------------------------------------------
# letter_to_frame
# ---------------------------------------------------------------------------

def test_letter_to_frame_missing_returns_placeholder(tmp_path):
    """When image file does not exist, returns a placeholder dict."""
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        frame = letter_to_frame("ASL", "A")
    assert frame["type"] == "placeholder"
    assert frame["label"] == "A"


def test_letter_to_frame_present_returns_image(tmp_path):
    """When image file exists, returns an image dict."""
    lang_dir = tmp_path / "ASL"
    lang_dir.mkdir()
    img_file = lang_dir / "B.png"
    img_file.write_bytes(b"\x89PNG")  # fake PNG header

    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        frame = letter_to_frame("ASL", "B")
    assert frame["type"] == "image"
    assert frame["label"] == "B"
    assert frame["path"].endswith("B.png")


def test_letter_to_frame_missing_logs_warning(tmp_path, caplog):
    """Missing image emits a WARNING log, not an exception."""
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        with caplog.at_level(logging.WARNING, logger="src.sign_image_mapper"):
            frame = letter_to_frame("ASL", "Z")
    assert frame["type"] == "placeholder"
    assert any("missing" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# text_to_sign_sequence
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty():
    assert text_to_sign_sequence("", "ASL") == []


def test_space_produces_space_marker(tmp_path):
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence("A B", "ASL")
    assert seq[1]["type"] == "space"


def test_non_alpha_char_is_placeholder(tmp_path):
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence("A!", "ASL")
    assert seq[1]["type"] == "placeholder"
    assert seq[1]["label"] == "!"


def test_sequence_length_matches_chars(tmp_path):
    text = "HI"
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence(text, "ASL")
    assert len(seq) == len(text)


def test_all_letters_produce_frames(tmp_path):
    """Every letter in the input produces a frame (image or placeholder)."""
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence("HELLO", "ASL")
    assert len(seq) == 5
    for frame in seq:
        assert frame["type"] in ("image", "placeholder")


def test_lowercase_input_normalized(tmp_path):
    """Lowercase input should still produce frames for uppercase labels."""
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence("hello", "ASL")
    for frame in seq:
        assert frame["label"].isupper() or frame["label"] == " "


def test_never_raises_on_bad_input(tmp_path):
    """text_to_sign_sequence must not raise for any printable input."""
    with patch("src.sign_image_mapper.SIGN_IMAGES_DIR", str(tmp_path)):
        seq = text_to_sign_sequence("நமஸ்கார்", "ISL")  # Tamil chars
    assert isinstance(seq, list)
    # All non-alpha chars become placeholders
    for frame in seq:
        assert frame["type"] in ("image", "placeholder", "space")


def test_all_expected_assets_exist_in_data():
    """Verify that A-Z and all 6 phrase labels actually exist on disk in the real data/sign_images folder."""
    import string
    from src.config import PHRASE_LABELS
    
    expected_labels = list(string.ascii_uppercase) + list(PHRASE_LABELS.keys())
    
    for lang in ["ASL", "ISL"]:
        for label in expected_labels:
            frame = letter_to_frame(lang, label)
            assert frame["type"] == "image", f"Missing real asset for {lang} - {label}"

