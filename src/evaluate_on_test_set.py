"""
evaluate_on_test_set.py
-----------------------
Evaluate a trained SignBridge ASL/ISL classifier against a held-out image
folder (e.g. Kaggle's asl_alphabet_test set) WITHOUT touching training data.

Usage
-----
# Default: uses the saved ASL model from models/ and Kaggle test folder
python -m src.evaluate_on_test_set

# Explicit paths (works for any retrained model later)
python -m src.evaluate_on_test_set \\
    --test-dir  "C:/path/to/asl_alphabet_test/asl_alphabet_test" \\
    --model     "models/signbridge_asl_model.joblib" \\
    --encoder   "models/signbridge_asl_labels.joblib" \\
    --language  ASL

Design decisions
----------------
* Static-image mode: HandLandmarkExtractor is constructed with
  static_image_mode=True so MediaPipe does full detection on every image
  instead of tracking (which requires temporal context).
* raw_predict() from predictor.py is reused — no duplication, no
  hold-to-commit state machine (meaningless for single still images).
* Labels not known to the model (e.g. "nothing", "del") are counted as
  unmapped and reported separately — not penalised as wrong.
* Images where no hand is detected are counted as "undetected" — not wrong.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Filename → expected label
# ---------------------------------------------------------------------------

# Map Kaggle filename stems that don't directly match the model's label set.
# Keys are lowercased stems.
_STEM_MAP: Dict[str, str] = {
    "space":   "SPACE",
    "nothing": "nothing",   # model doesn't know this class; flagged as unmapped
    "del":     "DEL",
}


def parse_expected_label(filename: str) -> str:
    """
    Extract the expected label from a Kaggle-style test image filename.

    Kaggle naming convention: ``<LABEL>_test.jpg``
    Examples:
        A_test.jpg    → "A"
        space_test.jpg → "SPACE"
        nothing_test.jpg → "nothing"   (unmapped sentinel)
        B_test.jpg    → "B"

    The label is always the portion of the stem before the first ``_test``
    suffix.  If no ``_test`` suffix is present the whole stem is used as-is.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]   # "A_test"
    if "_test" in stem:
        label_part = stem[: stem.rindex("_test")]             # "A"
    else:
        label_part = stem

    lower = label_part.lower()
    if lower in _STEM_MAP:
        return _STEM_MAP[lower]

    return label_part.upper()  # A-Z, DEL, SPACE, etc.


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

class ImageEvaluator:
    """
    Wraps the MediaPipe extractor and a loaded sklearn classifier for
    static-image evaluation.  Intentionally reuses existing project code
    rather than re-implementing landmark extraction or prediction.
    """

    def __init__(self, model_path: str, encoder_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")

        self.clf = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.known_labels = set(self.encoder.classes_)

        # static_image_mode=True: full detection per image (no temporal tracking)
        from src.landmark_utils import HandLandmarkExtractor
        self._extractor = HandLandmarkExtractor(static_image_mode=True)

    def predict_image(self, image_path: str) -> Tuple[Optional[str], float]:
        """
        Run the full MediaPipe → feature-extraction → classifier pipeline on
        a single image file.

        Returns (predicted_label, confidence), or (None, 0.0) if no hand
        was detected.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        frame_result = self._extractor.process(img_bgr)
        if frame_result is None:
            return None, 0.0  # no hand detected

        fv = frame_result.feature_vector
        proba = self.clf.predict_proba([fv])[0]
        best_idx = int(np.argmax(proba))
        label = self.encoder.inverse_transform([best_idx])[0]
        confidence = float(proba[best_idx])
        return label, confidence

    def close(self):
        self._extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    results: List[dict],
    *,
    top_n_confusion: int = 10,
) -> None:
    """Print accuracy, confusion pairs, and undetected count to stdout."""

    evaluated  = [r for r in results if r["status"] == "evaluated"]
    undetected = [r for r in results if r["status"] == "undetected"]
    unmapped   = [r for r in results if r["status"] == "unmapped"]

    correct = [r for r in evaluated if r["expected"] == r["predicted"]]
    wrong   = [r for r in evaluated if r["expected"] != r["predicted"]]

    n_eval = len(evaluated)
    n_corr = len(correct)
    acc = (n_corr / n_eval * 100) if n_eval else 0.0

    print()
    print("=" * 60)
    print("  SignBridge - Held-out Test Set Evaluation")
    print("=" * 60)
    print(f"  Result        : {n_eval} of {len(results)} images detected ({n_corr} of {n_eval} correct when detected)")
    print(f"  Total images  : {len(results)}")
    print(f"  Detected      : {n_eval} of {len(results)}")
    print(f"  Undetected    : {len(undetected)} of {len(results)} (no hand detected)")
    print(f"  Unmapped      : {len(unmapped)}  "
          f"({', '.join(r['expected'] for r in unmapped) or '—'})")
    print(f"  Correct       : {n_corr} of {n_eval} detected")
    print(f"  Accuracy note : Reported strictly as '{n_eval} of {len(results)} images detected' (not headline 100%)")
    print()

    if wrong:
        print(f"  Misclassifications ({len(wrong)} total):")
        pair_counts: Counter = Counter(
            (r["expected"], r["predicted"]) for r in wrong
        )
        print(f"  {'Expected':<12} {'Predicted':<12} {'Conf':>6}  Count")
        print("  " + "-" * 45)
        for (exp, pred), cnt in pair_counts.most_common(top_n_confusion):
            # Show mean confidence for this pair
            confs = [
                r["confidence"] for r in wrong
                if r["expected"] == exp and r["predicted"] == pred
            ]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            print(f"  {exp:<12} {pred:<12} {mean_conf:>5.0%}  {cnt}")
    else:
        print("  No misclassifications!")

    print()
    print("  Per-label detail:")
    print(f"  {'Label':<12} {'Status':<12} {'Predicted':<12} {'Conf':>6}")
    print("  " + "-" * 50)
    for r in sorted(results, key=lambda x: x["filename"]):
        status = r["status"]
        pred   = r.get("predicted") or "-"
        conf   = f"{r.get('confidence', 0.0):.0%}" if r.get("confidence") else "-"
        if status in ("undetected", "unmapped"):
            mark = "."
        elif r.get("expected") == r.get("predicted"):
            mark = "OK"
        else:
            mark = "XX"
        print(f"  {r['expected']:<12} {mark:<4} {status:<10} {pred:<12} {conf}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained SignBridge model against a held-out image folder."
    )
    p.add_argument(
        "--test-dir",
        default=r"C:\Users\rdars\Downloads\SignBridge\archive\asl_alphabet_test\asl_alphabet_test",
        help="Directory containing *_test.jpg images.",
    )
    p.add_argument(
        "--language",
        default="ASL",
        choices=["ASL", "ISL"],
        help="Which language model to load (used to derive default model/encoder paths).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Explicit path to .joblib model file. "
             "Defaults to models/signbridge_<language>_model.joblib",
    )
    p.add_argument(
        "--encoder",
        default=None,
        help="Explicit path to label-encoder .joblib file. "
             "Defaults to models/signbridge_<language>_labels.joblib",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-image debug lines while processing.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve default model/encoder paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lang = args.language.lower()
    model_path  = args.model   or os.path.join(base_dir, "models", f"signbridge_{lang}_model.joblib")
    encoder_path = args.encoder or os.path.join(base_dir, "models", f"signbridge_{lang}_labels.joblib")
    test_dir    = args.test_dir

    print(f"Model  : {model_path}")
    print(f"Encoder: {encoder_path}")
    print(f"Test   : {test_dir}")

    if not os.path.isdir(test_dir):
        print(f"[ERROR] Test directory not found: {test_dir}", file=sys.stderr)
        return 1

    # Gather images
    image_files = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"[ERROR] No images found in {test_dir}", file=sys.stderr)
        return 1

    print(f"\nFound {len(image_files)} image(s). Running evaluation...\n")

    results: List[dict] = []

    with ImageEvaluator(model_path, encoder_path) as ev:
        for img_path in image_files:
            fname = os.path.basename(img_path)
            expected = parse_expected_label(fname)

            # Labels not in model → skip (unmapped, not penalised)
            if expected not in ev.known_labels:
                if args.verbose:
                    print(f"  [UNMAPPED] {fname!r}  expected={expected!r}  (not in model)")
                results.append({
                    "filename": fname,
                    "expected": expected,
                    "predicted": None,
                    "confidence": None,
                    "status": "unmapped",
                })
                continue

            try:
                predicted, confidence = ev.predict_image(img_path)
            except ValueError as e:
                print(f"  [ERROR] {fname}: {e}", file=sys.stderr)
                continue

            if predicted is None:
                if args.verbose:
                    print(f"  [UNDETECTED] {fname!r}  expected={expected!r}  (no hand)")
                results.append({
                    "filename": fname,
                    "expected": expected,
                    "predicted": None,
                    "confidence": None,
                    "status": "undetected",
                })
            else:
                ok = "OK" if predicted == expected else "XX"
                if args.verbose:
                    print(f"  [{ok}] {fname!r}  expected={expected!r}  "
                          f"predicted={predicted!r}  conf={confidence:.0%}",
                          flush=True)
                results.append({
                    "filename": fname,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "status": "evaluated",
                })

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
