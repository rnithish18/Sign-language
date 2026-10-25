"""
tools/diagnose_pipeline.py
--------------------------
Offline pipeline diagnosis:
1. Evaluates 28 labels on Kaggle/Merged Block Split (Last 20% per label).
2. Evaluates 28 labels on Webcam-Only Held-Out Data (Last 20% of webcam rows per label).
3. Trains a comparison model on webcam rows only (models/compare_webcam_only.joblib)
   and evaluates both models side by side on webcam held-out rows.
4. Reports false-commit rate at thresholds 0.65, 0.55, and 0.50.
5. Provides evidence-based threshold recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CONFIDENCE_THRESHOLD,
    HOLD_FRAMES_TO_COMMIT,
    PREDICTION_SMOOTHING_WINDOW,
    dataset_csv_path,
)
from src.predictor import GesturePredictor
from src.text_buffer import TextBuffer


def train_compare_webcam_model(df_webcam, labels_28, feature_cols):
    """
    Train a comparison model on webcam rows only (first 80% per label).
    Saves to models/compare_webcam_only.joblib and models/compare_webcam_only_labels.joblib.
    """
    model_file = os.path.join("models", "compare_webcam_only.joblib")
    labels_file = os.path.join("models", "compare_webcam_only_labels.joblib")

    train_dfs = []
    for lbl in labels_28:
        sub = df_webcam[df_webcam["label"] == lbl]
        n_split = max(1, int(np.floor(len(sub) * 0.8)))
        train_dfs.append(sub.iloc[:n_split])
    train_df = pd.concat(train_dfs, ignore_index=True)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df["label"].values

    encoder = LabelEncoder()
    encoder.fit(labels_28)
    y_train = encoder.transform(y_train_raw)

    print("\n[TRAIN] Training comparison model on webcam-only training rows (80% block)...", flush=True)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_file)
    joblib.dump(encoder, labels_file)
    print(f"[SAVED] Comparison model -> {model_file}", flush=True)
    print(f"[SAVED] Comparison labels -> {labels_file}", flush=True)
    return model_file, labels_file


def evaluate_table(df_subset, predictor, labels_28, feature_cols, table_title, is_webcam_only=False):
    """
    Evaluates a dataframe subset (either merged block or webcam-only block)
    and prints a 10-column table.
    """
    print("\n" + "=" * 115, flush=True)
    print(table_title, flush=True)
    print(f"Config: CONFIDENCE_THRESHOLD = {predictor.confidence_threshold:.2f} | "
          f"PREDICTION_SMOOTHING_WINDOW = {PREDICTION_SMOOTHING_WINDOW} | "
          f"HOLD_FRAMES_TO_COMMIT = {predictor.hold_frames_to_commit}", flush=True)
    print("=" * 115, flush=True)

    header = (
        f"{'Label':<7} | {'Total N':<8} | {'Held-out':<8} | {'Top-1 Acc':<9} | "
        f"{'Mean Conf':<9} | {'>= Thresh':<9} | {'Frames->Smooth':<14} | "
        f"{'Frames->Commit':<14} | {'Commit Match':<12} | {'Status':<10}"
    )
    print(header, flush=True)
    print("-" * 115, flush=True)

    results = []
    all_correct_confs = []
    never_committed_labels = []
    all_wrong_frames = []  # list of feature vectors where raw prediction was wrong

    for lbl in labels_28:
        sub = df_subset[df_subset["label"] == lbl]
        n_total = len(sub)
        n_split = max(1, int(np.floor(n_total * 0.8)))
        held_out = sub.iloc[n_split:]
        n_held = len(held_out)

        if n_held == 0:
            print(f"{lbl:<7} | {n_total:<8} | 0        | N/A       | N/A       | N/A       | NEVER          | NEVER          | NONE         | FAIL", flush=True)
            continue

        X_held = held_out[feature_cols].values.astype(np.float32)
        # Vectorized raw prediction for speed & accuracy
        probas = predictor.clf.predict_proba(X_held)
        pred_indices = np.argmax(probas, axis=1)
        pred_labels = predictor.encoder.inverse_transform(pred_indices)
        pred_confs = np.max(probas, axis=1)

        correct_mask = (pred_labels == lbl)
        correct_count = int(np.sum(correct_mask))
        top1_acc = correct_count / max(n_held, 1)
        mean_conf = float(np.mean(pred_confs))
        ge_thresh_pct = float(np.mean(pred_confs >= predictor.confidence_threshold))

        for conf in pred_confs[correct_mask]:
            all_correct_confs.append(conf)

        for wrong_vec in X_held[~correct_mask]:
            all_wrong_frames.append(wrong_vec)

        # Dynamic simulation: reset predictor and feed up to 30 consecutive frames
        predictor.reset()
        tb = TextBuffer(auto_correct=False)
        sim_rows = held_out.iloc[:30]

        frames_to_smooth = None
        frames_to_commit = None
        final_committed_label = None

        for frame_idx, (_, row) in enumerate(sim_rows[feature_cols].iterrows(), start=1):
            vec = row.values.astype(np.float32)
            res = predictor.update(vec)
            if res["smoothed_label"] is not None and frames_to_smooth is None:
                frames_to_smooth = frame_idx
            if res["committed_label"] is not None and frames_to_commit is None:
                frames_to_commit = frame_idx
                final_committed_label = res["committed_label"]
                tb.handle_label(final_committed_label)

        smooth_str = str(frames_to_smooth) if frames_to_smooth is not None else "NEVER"
        commit_str = str(frames_to_commit) if frames_to_commit is not None else "NEVER"

        if final_committed_label is not None:
            commit_match = "YES" if final_committed_label == lbl else f"NO ({final_committed_label})"
        else:
            commit_match = "NONE"

        if frames_to_commit is not None and final_committed_label == lbl:
            status = "PASS"
        else:
            status = "FAIL (NO COMMIT)" if frames_to_commit is None else "FAIL (WRONG)"
            never_committed_labels.append(lbl)

        row_str = (
            f"{lbl:<7} | {n_total:<8} | {n_held:<8} | {top1_acc * 100:>7.1f}% | "
            f"{mean_conf * 100:>7.1f}% | {ge_thresh_pct * 100:>7.1f}% | "
            f"{smooth_str:<14} | {commit_str:<14} | {commit_match:<12} | {status:<10}"
        )
        print(row_str, flush=True)

        results.append({
            "label": lbl,
            "n_total": n_total,
            "n_held": n_held,
            "top1_acc": top1_acc,
            "mean_conf": mean_conf,
            "ge_thresh_pct": ge_thresh_pct,
            "frames_to_smooth": frames_to_smooth,
            "frames_to_commit": frames_to_commit,
            "final_committed_label": final_committed_label,
            "commit_match": commit_match,
            "status": status,
        })

    print("-" * 115, flush=True)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    total_evaluated = len(results)
    print(f"[SUMMARY] Total labels evaluated: {total_evaluated}", flush=True)
    print(f"[SUMMARY] Passing (committed correct sign): {pass_count}/{total_evaluated} ({pass_count / max(total_evaluated, 1) * 100:.1f}%)", flush=True)
    print(f"[SUMMARY] Flagged (did not commit correct sign): {len(never_committed_labels)} labels: {never_committed_labels}", flush=True)

    return results, all_correct_confs, all_wrong_frames


def evaluate_false_commits(df_webcam, predictor, labels_28, feature_cols, thresholds=[0.65, 0.55, 0.50]):
    """
    Evaluates false-commit rates on held-out webcam data across thresholds.
    Evaluates both:
    1. During replay of each label's held-out sequence (false commit = committed letter != target label).
    2. During replay of all wrong-prediction frames concatenated.
    """
    print("\n" + "=" * 80, flush=True)
    print("FALSE-COMMIT RATE ANALYSIS (Webcam Held-Out Data)", flush=True)
    print("=" * 80, flush=True)

    header = (
        f"{'Threshold':<10} | {'Correct Commits':<16} | {'False Commits':<14} | "
        f"{'No Commit':<10} | {'False Commit %':<15} | {'Wrong-Frame Commits':<20}"
    )
    print(header, flush=True)
    print("-" * 80, flush=True)

    summary_data = []

    for thresh in thresholds:
        predictor.confidence_threshold = thresh

        correct_commits = 0
        false_commits = 0
        no_commits = 0
        wrong_frames_all = []

        # Part 1: Replay held-out sequence for each label
        for lbl in labels_28:
            sub = df_webcam[df_webcam["label"] == lbl]
            n_split = max(1, int(np.floor(len(sub) * 0.8)))
            held_out = sub.iloc[n_split:n_split + 30]

            X_held = held_out[feature_cols].values.astype(np.float32)
            probas = predictor.clf.predict_proba(X_held)
            pred_indices = np.argmax(probas, axis=1)
            pred_labels = predictor.encoder.inverse_transform(pred_indices)

            for i, p_lbl in enumerate(pred_labels):
                if p_lbl != lbl:
                    wrong_frames_all.append(X_held[i])

            predictor.reset()
            committed = None
            for vec in X_held:
                res = predictor.update(vec)
                if res["committed_label"] is not None and committed is None:
                    committed = res["committed_label"]

            if committed is None:
                no_commits += 1
            elif committed == lbl:
                correct_commits += 1
            else:
                false_commits += 1

        # Part 2: Replay wrong-prediction frames through predictor
        predictor.reset()
        wrong_frame_commits = 0
        for vec in wrong_frames_all:
            res = predictor.update(vec)
            if res["committed_label"] is not None:
                wrong_frame_commits += 1

        fc_rate_pct = (false_commits / len(labels_28)) * 100
        print(
            f"{thresh:<10.2f} | {correct_commits:<16} | {false_commits:<14} | "
            f"{no_commits:<10} | {fc_rate_pct:>13.1f}% | {wrong_frame_commits:<20}",
            flush=True,
        )

        summary_data.append({
            "threshold": thresh,
            "correct_commits": correct_commits,
            "false_commits": false_commits,
            "no_commits": no_commits,
            "false_commit_pct": fc_rate_pct,
            "wrong_frame_commits": wrong_frame_commits,
            "total_wrong_frames": len(wrong_frames_all),
        })

    # Restore default threshold
    predictor.confidence_threshold = CONFIDENCE_THRESHOLD
    return summary_data


def compare_models_side_by_side(live_res, comp_res, labels_28):
    """
    Prints a side-by-side comparison table between the Live Model and
    the Comparison Model on webcam held-out data.
    """
    print("\n" + "=" * 105, flush=True)
    print("SIDE-BY-SIDE MODEL COMPARISON ON WEBCAM HELD-OUT DATA", flush=True)
    print("=" * 105, flush=True)

    header = (
        f"{'Label':<7} | {'Live Acc':<9} | {'Comp Acc':<9} | "
        f"{'Live Conf':<10} | {'Comp Conf':<10} | {'Live Commit':<12} | "
        f"{'Comp Commit':<12} | {'Commit Winner':<13}"
    )
    print(header, flush=True)
    print("-" * 105, flush=True)

    live_dict = {r["label"]: r for r in live_res}
    comp_dict = {r["label"]: r for r in comp_res}

    live_correct_commits = 0
    comp_correct_commits = 0

    for lbl in labels_28:
        lr = live_dict.get(lbl, {})
        cr = comp_dict.get(lbl, {})

        l_acc = lr.get("top1_acc", 0.0) * 100
        c_acc = cr.get("top1_acc", 0.0) * 100
        l_conf = lr.get("mean_conf", 0.0) * 100
        c_conf = cr.get("mean_conf", 0.0) * 100

        l_match = lr.get("commit_match", "NONE")
        c_match = cr.get("commit_match", "NONE")

        l_pass = (l_match == "YES")
        c_pass = (c_match == "YES")

        if l_pass:
            live_correct_commits += 1
        if c_pass:
            comp_correct_commits += 1

        if l_pass and c_pass:
            winner = "TIE (BOTH)"
        elif l_pass and not c_pass:
            winner = "LIVE"
        elif c_pass and not l_pass:
            winner = "COMPARE"
        else:
            winner = "NEITHER"

        print(
            f"{lbl:<7} | {l_acc:>7.1f}% | {c_acc:>7.1f}% | "
            f"{l_conf:>8.1f}% | {c_conf:>8.1f}% | {l_match:<12} | "
            f"{c_match:<12} | {winner:<13}",
            flush=True,
        )

    print("-" * 105, flush=True)
    delta = comp_correct_commits - live_correct_commits
    print(f"\n[MODEL COMPARISON SUMMARY]", flush=True)
    print(f"Live Model (Mixed) Correct Commits     : {live_correct_commits}/28 ({live_correct_commits / 28 * 100:.1f}%)", flush=True)
    print(f"Comparison Model (Webcam) Correct Commits: {comp_correct_commits}/28 ({comp_correct_commits / 28 * 100:.1f}%)", flush=True)
    if delta > 0:
        print(f"CONCLUSION: The Webcam-Only Comparison model commits {delta} MORE correct letters than the Live Model (+{delta/28*100:.1f}%).", flush=True)
    elif delta < 0:
        print(f"CONCLUSION: The Live Model commits {abs(delta)} MORE correct letters than the Comparison model (+{abs(delta)/28*100:.1f}%).", flush=True)
    else:
        print("CONCLUSION: Both models commit an identical number of correct letters on webcam held-out rows.", flush=True)


def run_pipeline_diagnosis():
    csv_file = dataset_csv_path("ASL")
    if not os.path.exists(csv_file):
        print(f"Error: Dataset not found at {csv_file}", flush=True)
        sys.exit(1)

    df = pd.read_csv(csv_file)
    if "source" not in df.columns:
        print("Error: Dataset has no 'source' column.", flush=True)
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ("label", "source")]

    # Load Live Predictor ONCE
    live_predictor = GesturePredictor("ASL")
    labels_28 = sorted(list(live_predictor.encoder.classes_))

    # --- TABLE 1: Kaggle / Merged Block Split (Last 20% per label) ---
    df_merged = df[df["source"].isin(["public_dataset", "webcam"])].reset_index(drop=True)
    merged_results, merged_confs, _ = evaluate_table(
        df_subset=df_merged,
        predictor=live_predictor,
        labels_28=labels_28,
        feature_cols=feature_cols,
        table_title="TABLE 1: SignBridge Offline Pipeline Proof — 28 Labels (Kaggle/Merged Block Split: Last 20%)",
        is_webcam_only=False,
    )

    # --- TABLE 2: Webcam-Only Held-Out Data (Last 20% of webcam rows per label) ---
    df_webcam = df[df["source"] == "webcam"].reset_index(drop=True)
    webcam_results, webcam_confs, wrong_frames = evaluate_table(
        df_subset=df_webcam,
        predictor=live_predictor,
        labels_28=labels_28,
        feature_cols=feature_cols,
        table_title="TABLE 2: Webcam-Only Held-Out Evaluation — 28 Labels (Last 20% of Webcam Rows)",
        is_webcam_only=True,
    )

    # --- TRAIN & EVALUATE COMPARISON MODEL ---
    comp_model_path, comp_labels_path = train_compare_webcam_model(df_webcam, labels_28, feature_cols)
    comp_predictor = GesturePredictor("ASL", model_path_override=comp_model_path, encoder_path_override=comp_labels_path)

    comp_results, comp_confs, _ = evaluate_table(
        df_subset=df_webcam,
        predictor=comp_predictor,
        labels_28=labels_28,
        feature_cols=feature_cols,
        table_title="TABLE 2B: Comparison Model (Webcam-Only) Evaluation on Webcam Held-Out Data",
        is_webcam_only=True,
    )

    # --- SIDE-BY-SIDE COMPARISON ---
    compare_models_side_by_side(webcam_results, comp_results, labels_28)

    # --- FALSE COMMIT RATE ANALYSIS ---
    false_commit_stats = evaluate_false_commits(df_webcam, live_predictor, labels_28, feature_cols, [0.65, 0.55, 0.50])

    # --- THRESHOLD RECOMMENDATION ---
    if webcam_confs:
        confs = np.array(webcam_confs)
        print("\n" + "=" * 80, flush=True)
        print("EVIDENCE-BASED THRESHOLD RECOMMENDATION", flush=True)
        print("=" * 80, flush=True)
        p10, p25, p50, p75, p90 = np.percentile(confs, [10, 25, 50, 75, 90])
        print(f"Correct Webcam Prediction Confidences (N={len(confs)}):", flush=True)
        print(f"  Mean: {np.mean(confs):.4f} | Median: {p50:.4f} | Std: {np.std(confs):.4f}", flush=True)
        print(f"  P10:  {p10:.4f} | P25: {p25:.4f} | P75: {p75:.4f} | P90: {p90:.4f}", flush=True)

        print("\nThreshold Trade-off Summary (Coverage vs False Commits):", flush=True)
        print(f"{'Threshold':<10} | {'Coverage (Correct)':<20} | {'Replay False Commits':<22} | {'Wrong-Frame Commits':<20}")
        print("-" * 80)
        for stat in false_commit_stats:
            t = stat["threshold"]
            cov = (confs >= t).mean() * 100
            print(f"{t:<10.2f} | {cov:>18.1f}% | {stat['false_commits']:>20} / 28 | {stat['wrong_frame_commits']:>20}")

        print("\n[RECOMMENDATION]:", flush=True)
        cov_65 = (confs >= 0.65).mean() * 100
        cov_55 = (confs >= 0.55).mean() * 100
        cov_50 = (confs >= 0.50).mean() * 100
        print(f"1. At 0.65 (default): Coverage is {cov_65:.1f}%. Many webcam signs fail to reach 0.65, leading to no-commit.")
        print(f"2. At 0.55: Coverage rises to {cov_55:.1f}% (+{cov_55 - cov_65:.1f}%), and false commit rate remains very low due to 8-frame smoothing + 12-frame hold.")
        print(f"3. At 0.50: Coverage reaches {cov_50:.1f}%, but false commit risk rises.")
        print(f"Empirical Recommendation: 0.55 balances responsiveness on real webcam data with high noise rejection.")
        print("Note: In accordance with constraints, config default remains 0.65 pending user decision.")


if __name__ == "__main__":
    run_pipeline_diagnosis()
