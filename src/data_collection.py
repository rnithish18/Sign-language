"""
data_collection.py
-------------------
Interactive webcam tool to build a labeled hand-landmark dataset.

Run:
    python -m src.data_collection --language ASL --label CLEAR --samples 200

Run multiple labels in auto timer mode:
    python -m src.data_collection --language ASL --auto --labels CLEAR SPEAK PHRASE_HELP --samples 100

Controls while the window is focused:
    c        - start/stop continuous capture for the current label
    n        - move to the next label in --labels (if a list was given)
    q / ESC  - quit and save

Each captured frame's normalized landmark vector is appended as a
row to data/processed/<language>_landmarks.csv, tagged with source='webcam'.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from typing import List, Optional

import cv2

from src.config import CONTROL_GESTURES, FEATURE_LENGTH, dataset_csv_path
from src.landmark_utils import HandLandmarkExtractor, landmarks_to_csv_row


def check_has_source(csv_path: str) -> bool:
    """Check if existing CSV header includes a 'source' column."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            return "source" in header
    return False


def ensure_csv_header(csv_path: str, feature_len: int = FEATURE_LENGTH, include_source: bool = True):
    """Ensure CSV exists with the standard header."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if not file_exists:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if include_source:
                header = ["label", "source"] + [f"f{i}" for i in range(feature_len)]
            else:
                header = ["label"] + [f"f{i}" for i in range(feature_len)]
            writer.writerow(header)


def backup_csv(csv_path: str) -> Optional[str]:
    """Create a backup of the CSV before modifying it."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        bak_path = csv_path + ".bak"
        shutil.copy2(csv_path, bak_path)
        print(f"[BACKUP] Existing dataset backed up to: {bak_path}")
        return bak_path
    return None


def collect(
    language: str,
    labels: list,
    samples_per_label: int,
    camera_index: int = 0,
    auto_mode: bool = False,
    auto_confirm: bool = True,
):
    csv_path = dataset_csv_path(language)
    extractor = HandLandmarkExtractor(static_image_mode=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Camera open failure: Could not open camera at index {camera_index}.")
        extractor.close()
        sys.exit(1)

    label_idx = 0
    current_label = labels[label_idx]
    count_for_label = 0
    total_rows_written = 0
    total_frames = 0
    backed_up = False

    # Auto-mode state
    if auto_mode:
        state = "COUNTDOWN"
        countdown_start = time.time()
        print(f"[AUTO] Preparing for first label: '{current_label}'")
    else:
        state = "MANUAL_PAUSED"

    ensure_csv_header(csv_path, feature_len=FEATURE_LENGTH, include_source=True)
    has_source = check_has_source(csv_path)

    print("=" * 60)
    print(f"Collecting data for language: {language}")
    print(f"Labels queued: {labels}")
    if auto_mode:
        print("AUTO MODE ENABLED.")
        print(f"Auto-confirm between labels: {auto_confirm}")
    else:
        print("Press 'c' to toggle capture, 'n' for next label, 'q' to quit.")
    print("=" * 60)

    window_name = "SignBridge - Data Collection"
    window_opened_time = time.time()
    exit_reason = "UNKNOWN"

    csv_file = None
    writer = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                exit_reason = "FRAME_READ_FAILURE"
                print(f"[EXIT] Camera frame read failure: Camera stopped delivering frames at frame {total_frames}.")
                if total_frames == 0:
                    print(f"[ERROR] Camera failure: Failed to read initial frame from camera {camera_index}.")
                    sys.exit(1)
                break

            total_frames += 1
            frame = cv2.flip(frame, 1)
            mp_raw = extractor.process_raw(frame)
            frame = extractor.draw(frame, mp_raw)
            frame_result = extractor.process(frame)

            now = time.time()

            # Lazy CSV open & backup on first capture
            def _write_sample(feat_vec):
                nonlocal csv_file, writer, backed_up, total_rows_written
                if not backed_up:
                    backup_csv(csv_path)
                    backed_up = True
                if csv_file is None:
                    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
                    writer = csv.writer(csv_file)
                if has_source:
                    row = landmarks_to_csv_row(feat_vec, current_label, source="webcam", session_id="webcam_session4")
                else:
                    row = landmarks_to_csv_row(feat_vec, current_label)
                writer.writerow(row)
                total_rows_written += 1

            # State Machine Logic
            if auto_mode:
                if state == "COUNTDOWN":
                    elapsed = now - countdown_start
                    if elapsed >= 3.0:
                        state = "CAPTURE"
                        print(f"[AUTO] Capturing '{current_label}'...")
                    else:
                        remaining = 3.0 - elapsed
                        cv2.putText(
                            frame,
                            f"Get ready: {current_label} in {remaining:.1f}...",
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                        )

                elif state == "CAPTURE":
                    if frame_result:
                        _write_sample(frame_result.feature_vector)
                        count_for_label += 1
                        if count_for_label >= samples_per_label:
                            print(f"[OK] Collected {count_for_label} samples for '{current_label}'.")
                            if label_idx + 1 >= len(labels):
                                exit_reason = "ALL_COMPLETED"
                                print("[AUTO] All labels completed.")
                                break
                            state = "DONE"
                            done_time = now

                elif state == "DONE":
                    elapsed = now - done_time
                    cv2.putText(frame, "Done!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if elapsed >= 1.0:
                        if auto_confirm:
                            label_idx += 1
                            current_label = labels[label_idx]
                            count_for_label = 0
                            state = "COUNTDOWN"
                            countdown_start = now
                            print(f"[AUTO] Advancing to '{current_label}'...")
                        else:
                            state = "WAITING"
                            print("[AUTO] Waiting for 'n' to advance...")

                elif state == "WAITING":
                    cv2.putText(
                        frame,
                        "Press 'n' to continue",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

            else:
                # Manual mode logic
                if state == "MANUAL_CAPTURING" and frame_result:
                    _write_sample(frame_result.feature_vector)
                    count_for_label += 1
                    if count_for_label >= samples_per_label:
                        state = "MANUAL_PAUSED"
                        print(f"[OK] Collected {count_for_label} samples for '{current_label}'.")

            # UI Overlays
            is_capturing = state in ("CAPTURE", "MANUAL_CAPTURING")
            status_color = (0, 200, 0) if is_capturing else (0, 0, 220)
            status_text = "CAPTURING" if is_capturing else state
            cv2.putText(frame, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Captured: {count_for_label}/{samples_per_label}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(
                frame,
                "[c]apture  [n]ext  [q]uit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.imshow(window_name, frame)

            # Check if user closed the window via X button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "WINDOW_CLOSED"
                    print("[EXIT] Window closed by user.")
                    break
            except cv2.error:
                pass

            raw_key = cv2.waitKey(1) & 0xFF

            # Ignore key presses for the first second after window opens
            if (now - window_opened_time) < 1.0:
                key = 255
            else:
                key = raw_key

            if key in (ord("q"), 27):
                exit_reason = "USER_QUIT_KEY"
                print("[EXIT] Quit requested by user (pressed 'q' or ESC).")
                break
            elif key == ord("c") and not auto_mode:
                state = "MANUAL_PAUSED" if state == "MANUAL_CAPTURING" else "MANUAL_CAPTURING"
            elif key == ord("n"):
                if auto_mode and state == "WAITING":
                    label_idx += 1
                    if label_idx >= len(labels):
                        exit_reason = "ALL_COMPLETED"
                        print("[AUTO] All labels completed.")
                        break
                    current_label = labels[label_idx]
                    count_for_label = 0
                    state = "COUNTDOWN"
                    countdown_start = time.time()
                elif not auto_mode:
                    label_idx = (label_idx + 1) % len(labels)
                    current_label = labels[label_idx]
                    count_for_label = 0
                    state = "MANUAL_PAUSED"
                    time.sleep(0.2)

    except Exception as exc:
        exit_reason = f"EXCEPTION: {exc}"
        print(f"[EXIT] Exception encountered during data collection: {exc}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        if csv_file is not None:
            csv_file.close()

        if total_rows_written > 0:
            print(f"[DONE] Dataset saved/updated with {total_rows_written} new samples at: {csv_path}")
        else:
            print("[DONE] 0 rows added, nothing saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand-landmark training data.")
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Space-separated list of labels to cycle through, e.g. CLEAR SPEAK PHRASE_HELP",
    )
    parser.add_argument("--label", default=None, help="Single label shortcut (overrides --labels).")
    parser.add_argument("--samples", type=int, default=150, help="Samples to capture per label.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0).")
    parser.add_argument("--auto", action="store_true", help="Enable auto-capture timer mode.")
    parser.add_argument(
        "--no-auto-confirm",
        action="store_false",
        dest="auto_confirm",
        help="Wait for 'n' between labels in auto mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.label:
        label_list = [args.label]
    elif args.labels:
        label_list = args.labels
    else:
        # Default: full alphabet + control gestures (BACKSPACE dropped)
        label_list = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + sorted(CONTROL_GESTURES)

    collect(args.language, label_list, args.samples, args.camera, args.auto, args.auto_confirm)
