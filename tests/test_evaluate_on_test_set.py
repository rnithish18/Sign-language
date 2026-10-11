"""
test_evaluate_on_test_set.py
----------------------------
Unit tests for src/evaluate_on_test_set.py.

Focuses on parse_expected_label() — the trickiest part — using a range of
mocked filenames that cover:
  • Normal A-Z letters
  • Special Kaggle classes: space → SPACE, nothing → nothing (unmapped)
  • Files that lack the _test suffix
  • Full paths (not just bare filenames)
  • Case variations

No camera, no model, no network required.
"""

import pytest
from src.evaluate_on_test_set import parse_expected_label


class TestParseExpectedLabel:
    # ── Standard letters ──────────────────────────────────────────────────

    def test_uppercase_letter(self):
        assert parse_expected_label("A_test.jpg") == "A"

    def test_each_letter_az(self):
        """All 26 letters should parse cleanly."""
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            fname = f"{ch}_test.jpg"
            assert parse_expected_label(fname) == ch, f"Failed for {fname}"

    def test_lowercase_letter_stem(self):
        """Lower-case letter stems that aren't in _STEM_MAP → uppercased."""
        # Kaggle uses uppercase stems for letters, but be defensive
        assert parse_expected_label("a_test.jpg") == "A"
        assert parse_expected_label("b_test.jpg") == "B"

    # ── Special Kaggle classes ─────────────────────────────────────────────

    def test_space_maps_to_SPACE(self):
        assert parse_expected_label("space_test.jpg") == "SPACE"

    def test_nothing_returns_nothing_sentinel(self):
        """'nothing' is not a model label; returned as-is so callers can unmapped-skip it."""
        assert parse_expected_label("nothing_test.jpg") == "nothing"

    def test_del_maps_to_DEL(self):
        assert parse_expected_label("del_test.jpg") == "DEL"

    # ── Suffix variants ────────────────────────────────────────────────────

    def test_no_test_suffix_uses_whole_stem(self):
        """If the file has no '_test' in the stem the whole stem is used (uppercased)."""
        assert parse_expected_label("A.jpg") == "A"

    def test_png_extension(self):
        assert parse_expected_label("C_test.png") == "C"

    # ── Full path support ─────────────────────────────────────────────────

    def test_full_windows_path(self):
        path = r"C:\Users\rdars\Downloads\SignBridge\archive\asl_alphabet_test\asl_alphabet_test\A_test.jpg"
        assert parse_expected_label(path) == "A"

    def test_full_posix_path(self):
        path = "/data/asl_alphabet_test/B_test.jpg"
        assert parse_expected_label(path) == "B"

    def test_space_full_path(self):
        path = r"C:\some\path\space_test.jpg"
        assert parse_expected_label(path) == "SPACE"

    def test_nothing_full_path(self):
        path = r"C:\some\path\nothing_test.jpg"
        assert parse_expected_label(path) == "nothing"

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_label_with_test_mid_stem(self):
        """A label that incidentally contains 'test' but not as the suffix."""
        # e.g. "PROTEST_test.jpg" → label part before last _test = "PROTEST"
        assert parse_expected_label("PROTEST_test.jpg") == "PROTEST"

    def test_filename_all_lowercase(self):
        assert parse_expected_label("z_test.jpg") == "Z"


class TestParseExpectedLabelKaggleFixture:
    """Replay every actual filename from the Kaggle asl_alphabet_test folder."""

    KAGGLE_FILES = [
        ("A_test.jpg",       "A"),
        ("B_test.jpg",       "B"),
        ("C_test.jpg",       "C"),
        ("D_test.jpg",       "D"),
        ("E_test.jpg",       "E"),
        ("F_test.jpg",       "F"),
        ("G_test.jpg",       "G"),
        ("H_test.jpg",       "H"),
        ("I_test.jpg",       "I"),
        ("J_test.jpg",       "J"),
        ("K_test.jpg",       "K"),
        ("L_test.jpg",       "L"),
        ("M_test.jpg",       "M"),
        ("N_test.jpg",       "N"),
        ("O_test.jpg",       "O"),
        ("P_test.jpg",       "P"),
        ("Q_test.jpg",       "Q"),
        ("R_test.jpg",       "R"),
        ("S_test.jpg",       "S"),
        ("T_test.jpg",       "T"),
        ("U_test.jpg",       "U"),
        ("V_test.jpg",       "V"),
        ("W_test.jpg",       "W"),
        ("X_test.jpg",       "X"),
        ("Y_test.jpg",       "Y"),
        ("Z_test.jpg",       "Z"),
        ("space_test.jpg",   "SPACE"),
        ("nothing_test.jpg", "nothing"),
    ]

    @pytest.mark.parametrize("filename, expected", KAGGLE_FILES)
    def test_kaggle_file(self, filename, expected):
        assert parse_expected_label(filename) == expected
