"""
cli_app.py
----------
Lightweight, dependency-minimal CLI version of SignBridge that runs
straight against your webcam with OpenCV — no browser/Streamlit needed.
Useful for quick testing after training a model, or on machines where
streamlit-webrtc's browser-based camera access is inconvenient.

Run (normal translation mode):
    python cli_app.py --language ASL

Run (per-label verification checklist — Task B):
    python cli_app.py --language ASL --verify

Verification keys:
    y / Y   - mark current label as correctly recognised → advance
    n / N   - mark current label as failed / misrecognised → advance
    s / S   - skip label (leave untested) → advance
    q / ESC - quit early and write partial results

Normal-mode keys:
    q / ESC   - quit
    s         - speak the current sentence
    SPACE     - manual space (in addition to the SPACE gesture)
"""

import argparse
import os

import cv2
import numpy as np

from src.landmark_utils import HandLandmarkExtractor
from src.predictor import GesturePredictor, ModelNotFoundError
from src.text_buffer import TextBuffer
from src.tts_engine import TTSEngine
from src.config import (
    ASL_ALPHABET, ISL_ALPHABET,
    CONTROL_GESTURES, PHRASE_LABELS,
    SUPPORTED_LANGUAGES, SUPPORTED_OUTPUT_LANGUAGES, DEFAULT_OUTPUT_LANGUAGE,
)


# ---------------------------------------------------------------------------
# Task A: live prediction overlay re-used from app.py static method, but we
# inline a trimmed version here to avoid pulling in Streamlit at import time.
# ---------------------------------------------------------------------------
def _draw_verify_hud(
    frame: np.ndarray,
    current_label: str,
    index: int,
    total: int,
    smoothed_label,
    confidence: float,
) -> None:
    """Draw the verification HUD on top of the landmark-annotated frame."""
    h, w = frame.shape[:2]

    # Dark banner at top
    cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 30), -1)

    # Expected label (large, white)
    cv2.putText(frame, f"TEST: {current_label}",
                (16, 52), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)

    # Progress counter
    cv2.putText(frame, f"{index + 1}/{total}",
                (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 1)

    # Current model prediction
    pred_str = smoothed_label or "—"
    match = (smoothed_label == current_label)
    pred_colour = (80, 220, 80) if match else (80, 80, 220)
    cv2.putText(frame, f"Prediction: {pred_str}  ({int(confidence * 100)}%)",
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_colour, 2)

    # Key legend at bottom
    legend = "y=PASS  n=FAIL  s=SKIP  q=QUIT"
    cv2.rectangle(frame, (0, h - 36), (w, h), (15, 15, 30), -1)
    cv2.putText(frame, legend, (16, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Task B: per-label verification loop
# ---------------------------------------------------------------------------
def run_verify(language: str, camera_index: int, output_dir: str):
    """
    Interactive label-by-label verification tool.
    No core prediction logic was modified; this is a standalone wrapper.
    """
    from src.verify_labels import VerificationSession

    try:
        predictor = GesturePredictor(language)
    except ModelNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    extractor = HandLandmarkExtractor()

    # Build ordered label list: alphabet + control gestures + phrase labels
    alphabet = ASL_ALPHABET if language == "ASL" else ISL_ALPHABET
    control = sorted(CONTROL_GESTURES)
    phrases = sorted(PHRASE_LABELS.keys())
    all_labels = alphabet + control + phrases

    def on_advance(label, idx, total):
        print(f"\n[{idx + 1}/{total}]  Show gesture for: {label!r}   (y=pass  n=fail  s=skip  q=quit)")

    session = VerificationSession(all_labels, language, on_advance=on_advance)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        return

    print("="*60)
    print(f"SignBridge — {language} Label Verification")
    print("Camera window: press the key *in the OpenCV window* (not terminal).")
    print("="*60)

    smoothed_label = None
    confidence = 0.0

    while not session.is_complete:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = np.ascontiguousarray(frame)

        mp_raw = extractor.process_raw(frame)
        frame = extractor.draw(frame, mp_raw)
        frame = np.ascontiguousarray(frame)

        frame_result = extractor.process_from_raw(frame, mp_raw)
        feature_vector = frame_result.feature_vector if frame_result else None

        result = predictor.update(feature_vector)
        smoothed_label = result["smoothed_label"]
        confidence = result["confidence"]

        _draw_verify_hud(
            frame,
            session.current_label,
            session._index,
            len(session.labels),
            smoothed_label,
            confidence,
        )

        cv2.imshow("SignBridge — Verification", frame)
        key = cv2.waitKey(1) & 0xFF
        session.handle_key(key, smoothed_label)

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    # Print summary
    print()
    for line in session.summary_lines():
        print(line)

    # Write to file
    fpath = session.write_results(output_dir)
    print(f"\nResults saved to: {fpath}")


# ---------------------------------------------------------------------------
# Normal translation loop (unchanged from original)
# ---------------------------------------------------------------------------
def run(language: str, camera_index: int, tts_backend: str, target_language: str):
    try:
        predictor = GesturePredictor(language)
    except ModelNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    extractor = HandLandmarkExtractor()
    buffer = TextBuffer(auto_correct=True)
    tts = TTSEngine(backend=tts_backend)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        return

    print("SignBridge CLI running. Press 'q' to quit, 's' to speak, SPACE to add a space.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        mp_raw = extractor.process_raw(frame)
        frame = extractor.draw(frame, mp_raw)
        frame_result = extractor.process(frame)
        feature_vector = frame_result.feature_vector if frame_result else None

        result = predictor.update(feature_vector)
        if result["committed_label"]:
            old_text = buffer.get_final_text()
            buffer.handle_label(result["committed_label"])
            new_text = buffer.get_final_text()

            # CLI translation to terminal (can't draw Tamil/Hindi in OpenCV easily)
            if new_text and new_text != old_text and target_language != "en":
                from src.translator import translate
                trans = translate(new_text, target_language)
                if trans["error"]:
                    print(f"\n[Translation Error] {trans['error']}")
                else:
                    print(f"\n[{target_language}] {trans['translated']}")

            if buffer.consume_sos():
                print("\n🚨🚨🚨 I NEED HELP 🚨🚨🚨\n")

            if buffer.consume_speak_request():
                text = buffer.get_final_text()
                if text.strip():
                    tts.speak(text, lang=target_language)

        h, w = frame.shape[:2]
        label_text = result["smoothed_label"] or "..."
        cv2.rectangle(frame, (0, h - 110), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"{label_text} ({int(result['confidence']*100)}%)",
                    (15, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        bar_w = int(w * result["progress"])
        cv2.rectangle(frame, (0, h - 60), (bar_w, h - 52), (0, 255, 120), -1)
        cv2.putText(frame, buffer.sentence[-60:], (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("SignBridge CLI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            text = buffer.get_final_text()
            if text.strip():
                tts.speak(text)
        elif key == 32:  # SPACE bar
            buffer.handle_label("SPACE")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


def parse_args():
    parser = argparse.ArgumentParser(description="SignBridge lightweight CLI app.")
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--tts", default="offline", choices=["offline", "online"])
    parser.add_argument("--out-lang", default=DEFAULT_OUTPUT_LANGUAGE,
                        choices=list(SUPPORTED_OUTPUT_LANGUAGES.keys()))
    parser.add_argument(
        "--verify", action="store_true",
        help="Run per-label verification checklist instead of translation mode.",
    )
    parser.add_argument(
        "--verify-out", default="verification_results",
        help="Directory for verification result files (default: verification_results/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        run_verify(args.language, args.camera, args.verify_out)
    else:
        target = SUPPORTED_OUTPUT_LANGUAGES[args.out_lang]
        run(args.language, args.camera, args.tts, target)

