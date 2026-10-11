import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.predictor import GesturePredictor, ModelNotFoundError
from src.config import CONFIDENCE_THRESHOLD, HOLD_FRAMES_TO_COMMIT, PREDICTION_SMOOTHING_WINDOW

@patch("src.predictor.os.path.exists")
@patch("src.predictor.joblib.load")
def test_predictor_smoothing_and_commit(mock_load, mock_exists):
    mock_exists.return_value = True
    
    mock_clf = MagicMock()
    mock_encoder = MagicMock()
    
    # Mock joblib to return our mock objects
    mock_load.side_effect = [mock_clf, mock_encoder]
    
    # Setup mock classifier to always predict index 0 with high confidence
    mock_clf.predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_encoder.inverse_transform.return_value = ["A"]
    
    predictor = GesturePredictor("ASL")
    
    # Feed frames to test hold-to-commit logic
    dummy_feature = np.zeros(128, dtype=np.float32)
    
    for i in range(HOLD_FRAMES_TO_COMMIT - 1):
        result = predictor.update(dummy_feature)
        assert result["smoothed_label"] == "A"
        assert result["committed_label"] is None
        assert result["progress"] == round((i + 1) / HOLD_FRAMES_TO_COMMIT, 2)
        
    # Final frame to trigger commit
    result = predictor.update(dummy_feature)
    assert result["committed_label"] == "A"
    assert result["progress"] == 1.0
    
    # After commit, it should cool down and return None for committed_label
    result = predictor.update(dummy_feature)
    assert result["committed_label"] is None
