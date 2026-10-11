"""
test_predictor_metadata.py
---------------------------
Tests for:
  - metadata sidecar reading in GesturePredictor
  - is_synthetic property (true/false/missing sidecar cases)
  - train_model.py writing a correct sidecar
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

# Mock mediapipe before any src import
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.python'] = MagicMock()
sys.modules['mediapipe.python.solutions'] = MagicMock()
sys.modules['mediapipe.python.solutions.hands'] = MagicMock()
sys.modules['mediapipe.python.solutions.drawing_utils'] = MagicMock()
sys.modules['mediapipe.python.solutions.drawing_styles'] = MagicMock()

from src.predictor import GesturePredictor, ModelNotFoundError
from src.train_model import train
from src.generate_synthetic_data import generate_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(tmp_dir: str, language: str = "TST",
                     synthetic: bool = True) -> GesturePredictor:
    """Create dummy .joblib + sidecar and return a GesturePredictor."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Patch config paths to point into tmp_dir
    model_file = os.path.join(tmp_dir, f"signbridge_{language.lower()}_model.joblib")
    label_file = os.path.join(tmp_dir, f"signbridge_{language.lower()}_labels.joblib")
    meta_file = os.path.join(tmp_dir, f"signbridge_{language.lower()}_metadata.json")

    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    enc = LabelEncoder()
    X = np.random.rand(20, 128)
    y = enc.fit_transform(["A"] * 10 + ["B"] * 10)
    clf.fit(X, y)
    joblib.dump(clf, model_file)
    joblib.dump(enc, label_file)

    meta = {"synthetic": synthetic, "data_source": "synthetic" if synthetic else "real",
            "trained_at": "2026-01-01T00:00:00+00:00"}
    with open(meta_file, "w") as f:
        json.dump(meta, f)

    with patch("src.predictor.model_path", return_value=model_file), \
         patch("src.predictor.label_encoder_path", return_value=label_file), \
         patch("src.predictor.metadata_path", return_value=meta_file):
        return GesturePredictor(language)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_is_synthetic_true(tmp_path):
    pred = _make_fake_model(str(tmp_path), synthetic=True)
    assert pred.is_synthetic is True


def test_is_synthetic_false(tmp_path):
    pred = _make_fake_model(str(tmp_path), synthetic=False)
    assert pred.is_synthetic is False


def test_no_sidecar_defaults_to_not_synthetic(tmp_path):
    """Old models without a sidecar should not be flagged as synthetic."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    language = "NOSIDE"
    model_file = str(tmp_path / f"signbridge_{language.lower()}_model.joblib")
    label_file = str(tmp_path / f"signbridge_{language.lower()}_labels.joblib")
    meta_file = str(tmp_path / f"signbridge_{language.lower()}_metadata.json")

    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    enc = LabelEncoder()
    X = np.random.rand(10, 128)
    y = enc.fit_transform(["A"] * 5 + ["B"] * 5)
    clf.fit(X, y)
    joblib.dump(clf, model_file)
    joblib.dump(enc, label_file)
    # No sidecar written

    with patch("src.predictor.model_path", return_value=model_file), \
         patch("src.predictor.label_encoder_path", return_value=label_file), \
         patch("src.predictor.metadata_path", return_value=meta_file):
        pred = GesturePredictor(language)

    assert pred.is_synthetic is False


def test_train_model_writes_synthetic_sidecar(tmp_path):
    """train() with data_source='synthetic' should write sidecar with synthetic=True."""
    from src.config import FEATURE_LENGTH
    import pandas as pd

    csv_file = str(tmp_path / "tst_landmarks.csv")
    # Write a tiny CSV
    cols = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    rows = []
    for label in ["A", "B"]:
        for _ in range(20):
            rows.append([label] + list(np.random.rand(FEATURE_LENGTH)))
    pd.DataFrame(rows, columns=cols).to_csv(csv_file, index=False)

    model_file = str(tmp_path / "signbridge_tst_model.joblib")
    label_file = str(tmp_path / "signbridge_tst_labels.joblib")
    meta_file = str(tmp_path / "signbridge_tst_metadata.json")

    with patch("src.train_model.dataset_csv_path", return_value=csv_file), \
         patch("src.train_model.model_path", return_value=model_file), \
         patch("src.train_model.label_encoder_path", return_value=label_file), \
         patch("src.train_model.metadata_path", return_value=meta_file):
        train("TST", data_source="synthetic")

    assert os.path.exists(meta_file)
    with open(meta_file) as f:
        meta = json.load(f)
    assert meta["synthetic"] is True
    assert meta["data_source"] == "synthetic"
    assert "trained_at" in meta
    assert meta["num_classes"] == 2


def test_train_model_writes_real_sidecar(tmp_path):
    """train() with data_source='real' should write sidecar with synthetic=False."""
    from src.config import FEATURE_LENGTH
    import pandas as pd

    csv_file = str(tmp_path / "tst_landmarks.csv")
    cols = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    rows = []
    for label in ["A", "B"]:
        for _ in range(20):
            rows.append([label] + list(np.random.rand(FEATURE_LENGTH)))
    pd.DataFrame(rows, columns=cols).to_csv(csv_file, index=False)

    model_file = str(tmp_path / "signbridge_tst_model.joblib")
    label_file = str(tmp_path / "signbridge_tst_labels.joblib")
    meta_file = str(tmp_path / "signbridge_tst_metadata.json")

    with patch("src.train_model.dataset_csv_path", return_value=csv_file), \
         patch("src.train_model.model_path", return_value=model_file), \
         patch("src.train_model.label_encoder_path", return_value=label_file), \
         patch("src.train_model.metadata_path", return_value=meta_file):
        train("TST", data_source="real")

    with open(meta_file) as f:
        meta = json.load(f)
    assert meta["synthetic"] is False
