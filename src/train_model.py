"""
train_model.py
---------------
Trains a gesture classifier on the landmark CSV produced by
data_collection.py / import_public_dataset.py, and saves the fitted
model + label encoder to disk.

Run (default: exclude synthetic rows):
    python -m src.train_model --language ASL

Run (include only specific sources):
    python -m src.train_model --language ASL --sources public_dataset webcam

Run (include all, including synthetic — not recommended for production):
    python -m src.train_model --language ASL --sources public_dataset webcam synthetic

Design decisions
----------------
* If the CSV has a 'source' column, rows can be filtered by source before
  training.  Pass --sources to control which sources are included.
* `synthetic` in the metadata is True if ANY synthetic rows are in the
  training set.
* Accuracy is reported SEPARATELY for labels that have real data (any
  public_dataset or webcam rows) vs. labels that are synthetic-only.
* Labels with no real rows are listed as 'needs_calibration' in metadata
  and are excluded from the reported accuracy figure.
* per_label_composition records {label: {source: count}} for full
  transparency.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    dataset_csv_path,
    label_encoder_path,
    metadata_path,
    model_path,
    ASL_ALPHABET,
    ISL_ALPHABET,
    CONTROL_GESTURES,
    PHRASE_LABELS,
)

# Sources considered "real" (human hand data, not generator noise)
REAL_SOURCES = {"public_dataset", "webcam", "real"}


def load_dataset(csv_path: str, include_sources=None, default_source=None,
                 val_session: str = None):
    """
    Load landmark CSV, optionally filtering to specific sources.

    Parameters
    ----------
    csv_path : str
    include_sources : list[str] or None
    default_source : str or None
    val_session : str or None
        If set, rows with this session_id are held out for validation only.
        Use "latest" to auto-select the most recent webcam session.

    Returns
    -------
    X, y_raw, sources, session_ids, val_mask
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No dataset found at {csv_path}. Run data_collection.py first."
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at {csv_path} is empty.")

    has_source_col = "source" in df.columns
    has_session_col = "session_id" in df.columns

    if has_source_col and include_sources is not None:
        before = len(df)
        df = df[df["source"].isin(include_sources)].reset_index(drop=True)
        print(f"[FILTER] source column present. Kept {len(df)}/{before} rows "
              f"matching sources: {include_sources}")
    elif not has_source_col:
        fallback = default_source if default_source is not None else "unknown"
        print(f"[WARN] CSV has no 'source' column — all rows treated as '{fallback}'.")
        df["source"] = fallback

    if not has_session_col:
        df["session_id"] = "unknown"

    # Resolve "latest" session alias
    resolved_val_session = None
    if val_session is not None:
        webcam_sessions = df.loc[df["source"] == "webcam", "session_id"].unique().tolist()
        if val_session == "latest":
            non_legacy = [s for s in webcam_sessions
                          if s not in ("webcam_legacy", "unknown")]
            if non_legacy:
                resolved_val_session = sorted(non_legacy)[-1]
            elif webcam_sessions:
                resolved_val_session = sorted(webcam_sessions)[-1]
            else:
                print("[WARN] --val-session latest: no webcam sessions found.")
        else:
            if val_session in df["session_id"].values:
                resolved_val_session = val_session
            else:
                print(f"[WARN] --val-session {val_session!r}: not found; "
                      "falling back to block split.")

    if resolved_val_session:
        print(f"[VAL-SESSION] Isolating session '{resolved_val_session}' as held-out.")
        val_mask = (df["session_id"] == resolved_val_session).values
        print(f"[VAL-SESSION] {val_mask.sum()} rows held out.")
    else:
        val_mask = np.zeros(len(df), dtype=bool)

    sources = df["source"].values
    session_ids = df["session_id"].values
    feature_cols = [c for c in df.columns if c not in ("label", "source", "session_id")]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values
    return X, y_raw, sources, session_ids, val_mask


def build_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=800,
            random_state=42,
        )
def block_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    Split sequentially per label by index blocks so near-duplicate adjacent
    frames are kept together and never leak across train and validation sets.
    """
    train_indices = []
    test_indices = []

    unique_labels = np.unique(y)
    for label in unique_labels:
        idx = np.where(y == label)[0]
        n_samples = len(idx)
        split_point = max(1, int(np.floor(n_samples * (1.0 - test_size))))
        train_indices.extend(idx[:split_point])
        test_indices.extend(idx[split_point:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train(language: str, model_kind: str = "rf", test_size: float = 0.2,
          include_sources=None, data_source=None, split_strategy: str = "block",
          val_session: str = None):
    """
    Train the classifier.

    Parameters
    ----------
    language : str
    model_kind : str
    test_size : float
    include_sources : list[str] or None
    data_source : str or None
    split_strategy : str  -- 'block' or 'random'
    val_session : str or None
        If set, webcam rows with this session_id go to validation only,
        never to training. Use "latest" to auto-select. When this is used,
        two accuracy numbers are reported separately:
          (a) Kaggle/block-split accuracy
          (b) Held-out webcam session accuracy
        These are NEVER merged.
    """
    csv_path = dataset_csv_path(language)
    X, y_raw, sources, session_ids, val_mask = load_dataset(
        csv_path, include_sources, default_source=data_source,
        val_session=val_session,
    )

    # --- Composition analysis -----------------------------------------------
    df_meta = pd.DataFrame({"label": y_raw, "source": sources})

    # Per-label source composition
    per_label_composition = {}
    labels_with_real = set()
    labels_synthetic_only = set()

    for label in sorted(df_meta["label"].unique()):
        sub = df_meta[df_meta["label"] == label]
        src_counts = sub["source"].value_counts().to_dict()
        per_label_composition[label] = src_counts
        real_count = sum(v for k, v in src_counts.items() if k in REAL_SOURCES)
        if real_count > 0:
            labels_with_real.add(label)
        else:
            labels_synthetic_only.add(label)

    any_synthetic = any(
        cnt > 0
        for src_counts in per_label_composition.values()
        for src, cnt in src_counts.items()
        if src == "synthetic"
    )

    print(f"\n[COMPOSITION]")
    print(f"  Labels with real data  : {sorted(labels_with_real)}")
    print(f"  Labels synthetic-only  : {sorted(labels_synthetic_only)}")
    print(f"  Any synthetic in set   : {any_synthetic}")

    # --- Train/test split ---------------------------------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # If a val_session was isolated, those rows go exclusively to validation.
    # Remaining rows are split with the chosen strategy.
    session_val_acc = None  # webcam session accuracy (reported separately)
    session_val_per_label = {}  # per-label accuracy on the session

    if val_mask.sum() > 0:
        # Session-based split: never let session rows into training
        train_idx = np.where(~val_mask)[0]
        session_idx = np.where(val_mask)[0]
        # Further split the non-session rows for Kaggle/block accuracy
        X_nonsession, y_nonsession = X[train_idx], y[train_idx]
        if split_strategy == "block":
            print(f"[SPLIT] val-session={val_mask.sum()} rows isolated; "
                  f"block-splitting remaining {len(train_idx)} rows.")
            X_train, X_block_test, y_train, y_block_test = block_train_test_split(
                X_nonsession, y_nonsession, test_size=test_size)
        else:
            X_train, X_block_test, y_train, y_block_test = train_test_split(
                X_nonsession, y_nonsession, test_size=test_size, random_state=42,
                stratify=y_nonsession if len(set(y_nonsession)) > 1 else None,
            )
        X_test, y_test = X[session_idx], y[session_idx]
        X_block_test_exists = True
    else:
        X_block_test_exists = False
        if split_strategy == "block":
            print(f"[SPLIT] Using consecutive index block split ({int((1-test_size)*100)}% train / {int(test_size*100)}% val) to prevent near-duplicate leakage.")
            X_train, X_test, y_train, y_test = block_train_test_split(X, y, test_size=test_size)
        else:
            print("[SPLIT] Using random stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if len(set(y)) > 1 else None,
            )
        X_block_test, y_block_test = X_test, y_test

    clf = build_model(model_kind)
    clf.fit(X_train, y_train)

    # (a) Block-split / Kaggle accuracy
    y_block_pred = clf.predict(X_block_test)
    acc_all = accuracy_score(y_block_test, y_block_pred)

    # (b) Webcam session accuracy (only when val_session was used)
    if val_mask.sum() > 0 and len(X_test) > 0:
        y_session_pred = clf.predict(X_test)
        session_val_acc = accuracy_score(y_test, y_session_pred)
        # Per-label breakdown, worst first
        for lbl_idx, lbl in enumerate(encoder.classes_):
            mask = y_test == lbl_idx
            if mask.sum() > 0:
                session_val_per_label[lbl] = accuracy_score(
                    y_test[mask], y_session_pred[mask])
        y_pred = y_session_pred  # for classification report on session data
    else:
        y_pred = y_block_pred

    # --- Accuracy: real-data labels only ------------------------------------
    real_label_indices = np.array([
        encoder.transform([lbl])[0]
        for lbl in labels_with_real
        if lbl in encoder.classes_
    ])
    if len(real_label_indices) > 0:
        real_mask = np.isin(y_block_test, real_label_indices)
        if real_mask.sum() > 0:
            acc_real = accuracy_score(y_block_test[real_mask], y_block_pred[real_mask])
        else:
            acc_real = None
    else:
        acc_real = None

    # --- Print results ------------------------------------------------------
    if any_synthetic:
        print(f"\n[RESULT] (a) Block-split accuracy (ALL labels incl. synthetic): {acc_all * 100:.2f}%")
        print("         NOTE: synthetic-only labels are trivially separable.")
        if acc_real is not None:
            print(f"[RESULT] (a) Block-split accuracy (real-data labels ONLY, {len(labels_with_real)} labels): "
                  f"{acc_real * 100:.2f}%")
        else:
            print("[RESULT] No real-data label test rows found.")
    else:
        print(f"\n[RESULT] (a) Block-split / Kaggle accuracy (real-data labels only): {acc_all * 100:.2f}%")
        acc_real = acc_all

    if session_val_acc is not None:
        print(f"[RESULT] (b) Held-out webcam session accuracy: {session_val_acc * 100:.2f}%")
        print("[RESULT] Per-label webcam session accuracy (worst first):")
        for lbl, lbl_acc in sorted(session_val_per_label.items(), key=lambda x: x[1]):
            print(f"    {lbl:<20} {lbl_acc * 100:6.1f}%")
    else:
        print("[RESULT] (b) No held-out webcam session (use --val-session to report this).")

    if acc_all == 1.0 and any_synthetic:
        print("\n" + "=" * 80)
        print("WARNING: 100% accuracy with synthetic data present — meaningless.")
        print("=" * 80 + "\n")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in encoder.classes_],
        zero_division=0,
    ))

    # --- Save model ---------------------------------------------------------
    joblib.dump(clf, model_path(language))
    joblib.dump(encoder, label_encoder_path(language))
    print(f"[SAVED] Model   -> {model_path(language)}")
    print(f"[SAVED] Labels  -> {label_encoder_path(language)}")

    # --- Metadata sidecar ---------------------------------------------------
    # Per-source totals
    source_totals = df_meta["source"].value_counts().to_dict()

    if data_source is not None:
        sidecar_data_source = data_source
        is_synthetic = (data_source == "synthetic") or any_synthetic
    else:
        sidecar_data_source = (
            "mixed" if len(set(sources)) > 1
            else (list(set(sources))[0] if len(set(sources)) == 1 else "unknown")
        )
        is_synthetic = any_synthetic

    meta = {
        "language": language,
        # synthetic=True if ANY synthetic rows contributed
        "synthetic": is_synthetic,
        # data_source: 'mixed' if multiple sources, otherwise the single source
        "data_source": sidecar_data_source,
        "sources_included": sorted(set(sources.tolist())),
        "model_kind": model_kind,
        "split_strategy": split_strategy,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_classes": len(encoder.classes_),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "source_totals": source_totals,
        # Accuracy on real-data labels only
        "validation_accuracy_real_labels_only": (
            round(acc_real, 4) if acc_real is not None else None
        ),
        # Overall accuracy (includes synthetic if present)
        "validation_accuracy_all_labels": round(acc_all, 4),
        # Labels with no real data — must be flagged in UI
        "needs_calibration": sorted(
            ((set(ASL_ALPHABET if language == "ASL" else ISL_ALPHABET)
              | set(CONTROL_GESTURES) | set(PHRASE_LABELS.keys())) - labels_with_real)
            | labels_synthetic_only
        ),
        "labels_with_real_data": sorted(labels_with_real),
        # Full per-label composition: {label: {source: count}}
        "per_label_composition": per_label_composition,
        # Legacy field — kept for backward compatibility with predictor.py banner logic
        "samples_per_label": {
            label: int((y_raw == label).sum())
            for label in encoder.classes_
        },
    }
    mpath = metadata_path(language)
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] Metadata -> {mpath}")

    needs_cal_list = meta["needs_calibration"]
    if needs_cal_list:
        print(f"\n[NEEDS CALIBRATION] {len(needs_cal_list)} labels have NO real data:")
        for lbl in needs_cal_list:
            print(f"    {lbl}")
        print("  These labels are excluded from the model / flagged in UI until real data is collected.")

    return clf, encoder, acc_real if acc_real is not None else acc_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SignBridge gesture classifier."
    )
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument("--model", default="rf", choices=["rf", "mlp"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        default="block",
        choices=["block", "random"],
        help="Train/validation split strategy: 'block' (default, prevents near-duplicate leakage) or 'random'.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["public_dataset", "webcam"],
        choices=["public_dataset", "webcam", "synthetic"],
        help=(
            "Which data sources to include in training. "
            "Default: public_dataset webcam (excludes synthetic). "
            "Add 'synthetic' to include generated rows."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.language,
        args.model,
        args.test_size,
        include_sources=args.sources,
        split_strategy=args.split_strategy,
    )
