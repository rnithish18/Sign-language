import os
import csv
from unittest.mock import patch, MagicMock
import numpy as np
from src.import_public_dataset import import_dataset

def test_import_public_dataset(tmp_path):
    """
    Tests the bulk-import logic over a dummy folder structure.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # Create two subfolders
    (dataset_dir / "A").mkdir()
    (dataset_dir / "B").mkdir()
    
    # Create dummy images (we just need the files to exist, we will mock cv2.imread)
    (dataset_dir / "A" / "1.jpg").touch()
    (dataset_dir / "A" / "2.jpg").touch()
    (dataset_dir / "B" / "1.jpg").touch()
    # B/2 is a bad file that imread fails on
    (dataset_dir / "B" / "bad.png").touch()
    
    csv_path = tmp_path / "test_imported_landmarks.csv"
    
    def mock_imread(path):
        if "bad" in path:
            return None
        return "dummy_frame"
        
    with patch("src.import_public_dataset.dataset_csv_path", return_value=str(csv_path)), \
         patch("src.import_public_dataset.HandLandmarkExtractor") as mock_extractor_cls, \
         patch("cv2.imread", side_effect=mock_imread):
         
        mock_ext_inst = MagicMock()
        mock_frame_result = MagicMock()
        mock_frame_result.feature_vector = np.zeros(63)
        # For A/1, return valid result.
        # For A/2, simulate no hand detected.
        # For B/1, return valid result.
        def mock_process(frame):
            # we can't easily distinguish frame if it's the same string,
            # so we'll just return valid result for the first 2 calls, then None for 3rd
            pass
            
        # Actually, let's just use side_effect
        mock_ext_inst.process.side_effect = [
            mock_frame_result, # A/1 works
            None,              # A/2 no hand
            mock_frame_result, # B/1 works
            # B/bad won't reach process
        ]
        mock_extractor_cls.return_value = mock_ext_inst
        
        import_dataset("ASL", str(dataset_dir))
        
    assert csv_path.exists()
    
    with open(csv_path, "r") as f:
        rows = list(csv.reader(f))
        
    # Header + A + B
    assert len(rows) == 3
    assert rows[0][0] == "label"
    assert rows[1][0] == "A"
    assert rows[2][0] == "B"
