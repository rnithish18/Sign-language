"""
data_collection.py
-------------------
Interactive webcam tool to build a labeled hand-landmark dataset.

Run:
    python -m src.data_collection --language ASL --label A --samples 200

Controls while the window is focused:
    c        - start/stop continuous capture for the current label
    n        - move to the next label in --labels (if a list was given)
    q / ESC  - quit and save

Each captured frame's normalized 63-dim landmark vector is appended as a
row to data/processed/<language>_landmarks.csv, with the label in column 0.
Re-running the tool for the same language appends to the existing CSV, so
you can build up your dataset across multiple sessions.
"""

import argparse
import csv
import os
import sys
import time

import cv2

from src.config import CONTROL_GESTURES, dataset_csv_path
from src.landmark_utils import HandLandmarkExtractor, landmarks_to_csv_row


def ensure_csv_header(csv_path: str, feature_len: int):
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(feature_len)]
            writer.writerow(header)


def collect(language: str, labels: list, samples_per_label: int, camera_index: int = 0, auto_mode: bool = False, auto_confirm: bool = True):
    csv_path = dataset_csv_path(language)
    extractor = HandLandmarkExtractor(static_image_mode=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        sys.exit(1)

    label_idx = 0
    current_label = labels[label_idx]
    count_for_label = 0

    # Auto-mode state
    # States: 'PREPARE' (wait for 'n' or auto-advance), 'COUNTDOWN', 'CAPTURE', 'PAUSE'
    if auto_mode:
        state = "COUNTDOWN"
        countdown_start = time.time()
        print(f"[AUTO] Preparing for first label: '{current_label}'")
    else:
        state = "MANUAL_PAUSED"

    from src.config import FEATURE_LENGTH
    ensure_csv_header(csv_path, feature_len=FEATURE_LENGTH)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    print("=" * 60)
    print(f"Collecting data for language: {language}")
    print(f"Labels queued: {labels}")
    if auto_mode:
        print("AUTO MODE ENABLED.")
        print(f"Auto-confirm between labels: {auto_confirm}")
    else:
        print("Press 'c' to toggle capture, 'n' for next label, 'q' to quit.")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            mp_raw = extractor.process_raw(frame)
            frame = extractor.draw(frame, mp_raw)
            frame_result = extractor.process(frame)

            now = time.time()

            # State Machine Logic
            if auto_mode:
                if state == "COUNTDOWN":
                    elapsed = now - countdown_start
                    if elapsed >= 3.0:
                        state = "CAPTURE"
                        print(f"[AUTO] Capturing '{current_label}'...")
                    else:
                        remaining = 3.0 - elapsed
                        cv2.putText(frame, f"Get ready: {current_label} in {remaining:.1f}...", (10, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        
                elif state == "CAPTURE":
                    if frame_result:
                        writer.writerow(landmarks_to_csv_row(frame_result.feature_vector, current_label))
                        count_for_label += 1
                        if count_for_label >= samples_per_label:
                            print(f"[OK] Collected {count_for_label} samples for '{current_label}'.")
                            if label_idx + 1 >= len(labels):
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
                    cv2.putText(frame, "Press 'n' to continue", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            else:
                # Manual mode logic
                if state == "MANUAL_CAPTURING" and frame_result:
                    writer.writerow(landmarks_to_csv_row(frame_result.feature_vector, current_label))
                    count_for_label += 1
                    if count_for_label >= samples_per_label:
                        state = "MANUAL_PAUSED"
                        print(f"[OK] Collected {count_for_label} samples for '{current_label}'.")

            # UI Overlays
            is_capturing = state in ("CAPTURE", "MANUAL_CAPTURING")
            status_color = (0, 200, 0) if is_capturing else (0, 0, 220)
            status_text = "CAPTURING" if is_capturing else state
            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Captured: {count_for_label}/{samples_per_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "[c]apture  [n]ext  [q]uit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("SignBridge - Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("c") and not auto_mode:
                state = "MANUAL_PAUSED" if state == "MANUAL_CAPTURING" else "MANUAL_CAPTURING"
            elif key == ord("n"):
                if auto_mode and state == "WAITING":
                    label_idx += 1
                    if label_idx >= len(labels):
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        csv_file.close()
        print(f"[DONE] Dataset saved/updated at: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hand-landmark training data.")
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Space-separated list of labels to cycle through, e.g. A B C SPACE BACKSPACE CLEAR",
    )
    parser.add_argument("--label", default=None, help="Single label shortcut (overrides --labels).")
    parser.add_argument("--samples", type=int, default=150, help="Samples to capture per label.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--auto", action="store_true", help="Enable auto-capture timer mode.")
    parser.add_argument("--no-auto-confirm", action="store_false", dest="auto_confirm", help="Wait for 'n' between labels in auto mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.label:
        label_list = [args.label]
    elif args.labels:
        label_list = args.labels
    else:
        # Default: full alphabet + control gestures
        label_list = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + sorted(CONTROL_GESTURES)

    collect(args.language, label_list, args.samples, args.camera, args.auto, args.auto_confirm)
