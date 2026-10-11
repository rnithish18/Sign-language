import numpy as np
import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock mediapipe before it's imported by landmark_utils
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.python'] = MagicMock()
sys.modules['mediapipe.python.solutions'] = MagicMock()
sys.modules['mediapipe.python.solutions.hands'] = MagicMock()
sys.modules['mediapipe.python.solutions.drawing_utils'] = MagicMock()
sys.modules['mediapipe.python.solutions.drawing_styles'] = MagicMock()

from src.config import MAX_NUM_HANDS, NUM_LANDMARKS
from src.landmark_utils import FrameResult, HandResult, normalize_landmarks, HandLandmarkExtractor

def test_normalize_landmarks():
    raw = [(i*10.0, i*10.0, 0.0) for i in range(21)]
    feature_vector = normalize_landmarks(raw)
    assert len(feature_vector) == 63
    assert feature_vector[0] == 0.0
    assert feature_vector[1] == 0.0
    assert feature_vector[2] == 0.0
    assert np.max(np.abs(feature_vector)) <= 1.00001

@patch("src.landmark_utils.mp_hands.Hands")
def test_extractor_process_two_hands(MockHands):
    mock_hands_instance = MagicMock()
    MockHands.return_value = mock_hands_instance
    
    # Mocking MediaPipe output
    mock_result = MagicMock()
    
    # Create two dummy hand landmarks
    class DummyLM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            
    class DummyHandLandmarks:
        def __init__(self):
            self.landmark = [DummyLM(i, i, i) for i in range(21)]
            
    class DummyClassification:
        def __init__(self, label, score):
            self.label = label
            self.score = score
            
    class DummyHandedness:
        def __init__(self, label, score):
            self.classification = [DummyClassification(label, score)]

    # We provide a Right hand first, then a Left hand.
    # The extractor should sort them into Left, then Right.
    mock_result.multi_hand_landmarks = [DummyHandLandmarks(), DummyHandLandmarks()]
    mock_result.multi_handedness = [
        DummyHandedness("Right", 0.9),
        DummyHandedness("Left", 0.8)
    ]
    
    mock_hands_instance.process.return_value = mock_result
    
    with patch("src.landmark_utils.MAX_NUM_HANDS", 2):
        extractor = HandLandmarkExtractor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_result = extractor.process(frame)
        
        assert frame_result is not None
        assert len(frame_result.hands) == 2
        # Feature vector length should be (1 + 63) * 2 = 128
        assert len(frame_result.feature_vector) == 128
        
        # Check presence flags
        # Left slot is first (idx 0), Right slot is second (idx 64)
        assert frame_result.feature_vector[0] == 1.0  # Left present
        assert frame_result.feature_vector[64] == 1.0 # Right present

@patch("src.landmark_utils.mp_hands.Hands")
def test_extractor_process_missing_hand(MockHands):
    mock_hands_instance = MagicMock()
    MockHands.return_value = mock_hands_instance
    
    mock_result = MagicMock()
    
    class DummyLM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
            
    class DummyHandLandmarks:
        def __init__(self):
            self.landmark = [DummyLM(i, i, i) for i in range(21)]
            
    class DummyClassification:
        def __init__(self, label, score):
            self.label, self.score = label, score
            
    class DummyHandedness:
        def __init__(self, label, score):
            self.classification = [DummyClassification(label, score)]

    # Only a Right hand is detected
    mock_result.multi_hand_landmarks = [DummyHandLandmarks()]
    mock_result.multi_handedness = [DummyHandedness("Right", 0.9)]
    
    mock_hands_instance.process.return_value = mock_result
    
    with patch("src.landmark_utils.MAX_NUM_HANDS", 2):
        extractor = HandLandmarkExtractor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_result = extractor.process(frame)
        
        assert frame_result is not None
        assert len(frame_result.hands) == 1
        assert len(frame_result.feature_vector) == 128
        
        # Left slot should be empty (padded with 0s)
        assert frame_result.feature_vector[0] == 0.0 # Left not present
        assert np.all(frame_result.feature_vector[1:64] == 0.0) # Left features are 0
        
        # Right slot should be present
        assert frame_result.feature_vector[64] == 1.0 # Right present
        assert np.any(frame_result.feature_vector[65:128] != 0.0) # Right features exist

