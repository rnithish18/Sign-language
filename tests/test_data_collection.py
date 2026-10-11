import os
import csv
from unittest.mock import patch, MagicMock
import pytest
from src.data_collection import collect

import numpy as np

def test_data_collection_auto_mode(tmp_path):
    """
    Tests the --auto flag state machine logic (countdown -> capture -> auto-advance).
    Uses mocked cv2 and HandLandmarkExtractor to avoid requiring a real camera.
    """
    labels = ["A", "B"]
    samples_per_label = 2
    
    # We will mock dataset_csv_path to write to a temp file
    mock_csv_path = tmp_path / "test_landmarks.csv"
    
    # Track the sequence of states we expect to see
    state_timeline = []
    
    # We need a predictable time source to trigger state transitions
    current_time = 1000.0
    
    def mock_time():
        nonlocal current_time
        # Advance time slightly on each call so the countdown completes
        current_time += 0.5
        return current_time
        
    def mock_waitkey(delay):
        # We don't press any keys, we rely on the state machine auto-advancing
        # and eventually returning. If it gets stuck, we should break it, 
        # but the logic breaks itself when all labels are done.
        return 0
        
    def mock_read():
        # Always return a valid dummy frame
        return True, "dummy_frame"
        
    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor") as mock_extractor_cls, \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.waitKey", side_effect=mock_waitkey), \
         patch("time.time", side_effect=mock_time), \
         patch("time.sleep"):
         
        # Setup mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = mock_read
        mock_vc.return_value = mock_cap
        
        # Setup mock extractor
        mock_ext_inst = MagicMock()
        # process() returns an object with a feature_vector
        mock_frame_result = MagicMock()
        mock_frame_result.feature_vector = np.zeros(63)
        mock_ext_inst.process.return_value = mock_frame_result
        mock_extractor_cls.return_value = mock_ext_inst
        
        # Run collection
        collect(
            language="ASL", 
            labels=labels, 
            samples_per_label=samples_per_label, 
            camera_index=0, 
            auto_mode=True, 
            auto_confirm=True
        )
        
    # Verify the CSV was written correctly
    assert mock_csv_path.exists()
    
    with open(mock_csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Header + 2 samples of A + 2 samples of B
    assert len(rows) == 5
    assert rows[0][0] == "label"
    assert rows[1][0] == "A"
    assert rows[2][0] == "A"
    assert rows[3][0] == "B"
    assert rows[4][0] == "B"

def test_data_collection_auto_mode_no_confirm(tmp_path):
    """
    Tests --auto mode where --no-auto-confirm is used.
    It should pause in WAITING state and require an 'n' keypress.
    """
    labels = ["A", "B"]
    samples_per_label = 1
    mock_csv_path = tmp_path / "test_landmarks.csv"
    current_time = 1000.0
    
    def mock_time():
        nonlocal current_time
        current_time += 1.0 # 1 sec per loop
        return current_time
        
    wait_calls = 0
    def mock_waitkey(delay):
        nonlocal wait_calls
        wait_calls += 1
        # It takes ~3 calls (3 secs) to finish A
        # Then it enters DONE, then 1 sec later it enters WAITING
        # If we are in WAITING (wait_calls > 5), we press 'n'
        if wait_calls == 6:
            return ord('n')
        # At wait_calls ~ 10, it finishes B and exits
        return 0
        
    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor") as mock_extractor_cls, \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.waitKey", side_effect=mock_waitkey), \
         patch("time.time", side_effect=mock_time), \
         patch("time.sleep"):
         
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "dummy_frame")
        mock_vc.return_value = mock_cap
        
        mock_ext_inst = MagicMock()
        mock_frame_result = MagicMock()
        mock_frame_result.feature_vector = np.zeros(63)
        mock_ext_inst.process.return_value = mock_frame_result
        mock_extractor_cls.return_value = mock_ext_inst
        
        collect(language="ASL", labels=labels, samples_per_label=samples_per_label, 
                camera_index=0, auto_mode=True, auto_confirm=False)
                
    with open(mock_csv_path, "r") as f:
        rows = list(csv.reader(f))
        
    assert len(rows) == 3 # Header, A, B
    assert rows[1][0] == "A"
    assert rows[2][0] == "B"
