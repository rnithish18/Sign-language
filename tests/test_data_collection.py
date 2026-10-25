import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data_collection import collect, ensure_csv_header, backup_csv


def test_data_collection_auto_mode(tmp_path):
    """
    Tests the --auto flag state machine logic (countdown -> capture -> auto-advance).
    Uses mocked cv2 and HandLandmarkExtractor to avoid requiring a real camera.
    """
    labels = ["A", "B"]
    samples_per_label = 2

    mock_csv_path = tmp_path / "test_landmarks.csv"
    current_time = 1000.0

    def mock_time():
        nonlocal current_time
        current_time += 0.5
        return current_time

    def mock_waitkey(delay):
        return 0

    def mock_read():
        return True, "dummy_frame"

    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor") as mock_extractor_cls, \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.getWindowProperty", return_value=1), \
         patch("cv2.waitKey", side_effect=mock_waitkey), \
         patch("time.time", side_effect=mock_time), \
         patch("time.sleep"):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = mock_read
        mock_vc.return_value = mock_cap

        mock_ext_inst = MagicMock()
        mock_frame_result = MagicMock()
        mock_frame_result.feature_vector = np.zeros(63)
        mock_ext_inst.process.return_value = mock_frame_result
        mock_extractor_cls.return_value = mock_ext_inst

        collect(
            language="ASL",
            labels=labels,
            samples_per_label=samples_per_label,
            camera_index=0,
            auto_mode=True,
            auto_confirm=True,
        )

    assert mock_csv_path.exists()
    with open(mock_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Header + 2 samples of A + 2 samples of B
    assert len(rows) == 5
    assert rows[0][0] == "label"
    assert rows[0][1] == "source"
    assert rows[1][0] == "A"
    assert rows[1][1] == "webcam"
    assert rows[2][0] == "A"
    assert rows[3][0] == "B"
    assert rows[3][1] == "webcam"


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
        current_time += 1.0  # 1 sec per loop
        return current_time

    wait_calls = 0

    def mock_waitkey(delay):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls >= 6:
            return ord("n")
        return 0

    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor") as mock_extractor_cls, \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.getWindowProperty", return_value=1), \
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

        collect(
            language="ASL",
            labels=labels,
            samples_per_label=samples_per_label,
            camera_index=0,
            auto_mode=True,
            auto_confirm=False,
        )

    with open(mock_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3  # Header, A, B
    assert rows[1][0] == "A"
    assert rows[2][0] == "B"


def test_data_collection_camera_open_failure(tmp_path, capsys):
    """Exit path 1: Camera cannot be opened -> exit non-zero with distinct error."""
    mock_csv_path = tmp_path / "test_landmarks.csv"
    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor"), \
         patch("cv2.VideoCapture") as mock_vc:

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc.return_value = mock_cap

        with pytest.raises(SystemExit) as exc_info:
            collect(language="ASL", labels=["A"], samples_per_label=5, camera_index=99)
        assert exc_info.value.code == 1

    out = capsys.readouterr().out
    assert "Camera open failure" in out


def test_data_collection_frame_read_failure_initial(tmp_path, capsys):
    """Exit path 2: Camera opened but initial frame read fails -> exit non-zero with 0 rows."""
    mock_csv_path = tmp_path / "test_landmarks.csv"
    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor"), \
         patch("cv2.VideoCapture") as mock_vc:

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_vc.return_value = mock_cap

        with pytest.raises(SystemExit) as exc_info:
            collect(language="ASL", labels=["A"], samples_per_label=5, camera_index=0)
        assert exc_info.value.code == 1

    out = capsys.readouterr().out
    assert "Camera frame read failure" in out
    assert "0 rows added, nothing saved" in out


def test_data_collection_user_quit_key(tmp_path, capsys):
    """Exit path 3: User presses 'q' after initial second -> quit cleanly, print 0 rows added."""
    mock_csv_path = tmp_path / "test_landmarks.csv"
    curr_t = 1000.0

    def mock_time():
        nonlocal curr_t
        curr_t += 2.0  # past 1.0s debounce
        return curr_t

    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor"), \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.getWindowProperty", return_value=1), \
         patch("cv2.waitKey", return_value=ord("q")), \
         patch("time.time", side_effect=mock_time):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "dummy_frame")
        mock_vc.return_value = mock_cap

        collect(language="ASL", labels=["A"], samples_per_label=5, auto_mode=False)

    out = capsys.readouterr().out
    assert "Quit requested by user" in out
    assert "0 rows added, nothing saved" in out


def test_data_collection_window_closed(tmp_path, capsys):
    """Exit path 4: User closes the OpenCV window via close button -> exit cleanly."""
    mock_csv_path = tmp_path / "test_landmarks.csv"
    curr_t = 1000.0

    def mock_time():
        nonlocal curr_t
        curr_t += 0.1
        return curr_t

    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor"), \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.getWindowProperty", return_value=0), \
         patch("cv2.waitKey", return_value=255), \
         patch("time.time", side_effect=mock_time):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "dummy_frame")
        mock_vc.return_value = mock_cap

        collect(language="ASL", labels=["A"], samples_per_label=5, auto_mode=False)

    out = capsys.readouterr().out
    assert "Window closed by user" in out
    assert "0 rows added, nothing saved" in out


def test_data_collection_key_debounce_first_second(tmp_path):
    """Exit path 5: Stale key press during the first second after opening is ignored."""
    mock_csv_path = tmp_path / "test_landmarks.csv"
    curr_t = 1000.0
    waitkey_calls = 0

    def mock_time():
        nonlocal curr_t
        curr_t += 0.2  # 5 calls are <= 1.0s
        return curr_t

    def mock_waitkey(delay):
        nonlocal waitkey_calls
        waitkey_calls += 1
        if waitkey_calls < 4:
            return ord("q")  # Early quit key -> must be ignored
        return 0

    with patch("src.data_collection.dataset_csv_path", return_value=str(mock_csv_path)), \
         patch("src.data_collection.HandLandmarkExtractor"), \
         patch("cv2.VideoCapture") as mock_vc, \
         patch("cv2.flip", return_value="dummy_frame"), \
         patch("cv2.imshow"), \
         patch("cv2.putText"), \
         patch("cv2.getWindowProperty", return_value=1), \
         patch("cv2.waitKey", side_effect=mock_waitkey), \
         patch("time.time", side_effect=mock_time):

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Read 5 frames successfully, then EOF
        mock_cap.read.side_effect = [(True, "dummy")] * 5 + [(False, None)]
        mock_vc.return_value = mock_cap

        collect(language="ASL", labels=["A"], samples_per_label=5, auto_mode=False)

    # Reached beyond the first 3 calls despite 'q' press
    assert waitkey_calls >= 4


def test_data_collection_backup_created(tmp_path):
    """Test that CSV backup is created before new rows are appended."""
    csv_file = tmp_path / "test_landmarks.csv"
    # Seed an existing CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        f.write("label,source,f0\nA,webcam,0.1\n")

    bak = backup_csv(str(csv_file))
    assert os.path.exists(bak)
    with open(bak, "r", encoding="utf-8") as f:
        content = f.read()
    assert "A,webcam,0.1" in content
