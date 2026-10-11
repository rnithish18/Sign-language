"""
generate_synthetic_data.py
---------------------------
Bootstrap a synthetic 128-dim landmark dataset so the app can be run
and tested without having to collect a large dataset manually.

Generates distinct deterministic centers for each label, and samples
with Gaussian jitter to create a trainable classification dataset.
"""

import argparse
import csv
import os
import numpy as np

from src.config import (
    ASL_ALPHABET,
    ISL_ALPHABET,
    CONTROL_GESTURES,
    PHRASE_GESTURES,
    dataset_csv_path,
    FEATURE_LENGTH,
)

def generate_data(language: str, samples_per_label: int = 150, hard_mode: bool = False):
    csv_path = dataset_csv_path(language)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    alphabet = ASL_ALPHABET if language == "ASL" else ISL_ALPHABET
    # Include phrase labels so the classifier knows all gesture classes
    labels = alphabet + sorted(list(CONTROL_GESTURES)) + sorted(list(PHRASE_GESTURES))
    
    # Deterministic randomness so it's stable
    np.random.seed(42)
    
    print(f"Generating synthetic {language} dataset ({len(labels)} labels, {samples_per_label} samples each)...")
    if hard_mode:
        print("[!] Using HARD MODE: Classes will have heavily overlapping jitter to stress-test the classifier.")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
        writer.writerow(header)
        
        for i, label in enumerate(labels):
            if hard_mode and i > 0 and i % 3 == 0:
                # Deliberately overlap base vector with a previous class every 3rd label
                base_features = prev_base_features
            else:
                base_features = np.random.uniform(-1, 1, size=(FEATURE_LENGTH,))
                
            prev_base_features = base_features.copy()
            
            if language == "ASL":
                base_features[0] = 1.0   
                base_features[64] = 0.0  
                base_features[65:] = 0.0 
            else:
                base_features[0] = 1.0   
                base_features[64] = 1.0  
                
            for _ in range(samples_per_label):
                # Larger noise for hard mode
                noise_scale = 0.4 if hard_mode else 0.05
                noise = np.random.normal(0, noise_scale, size=(FEATURE_LENGTH,))
                sample = base_features + noise
                
                if language == "ASL":
                    sample[0] = 1.0
                    sample[64] = 0.0
                    sample[65:] = 0.0
                else:
                    sample[0] = 1.0
                    sample[64] = 1.0
                
                writer.writerow([label] + sample.tolist())

    print(f"[OK] Saved synthetic dataset to {csv_path}")
    print("\n" + "="*80)
    print("WARNING: This is a SYNTHETIC dataset used only for baseline/smoke-testing.")
    print("Any 100% validation accuracy achieved during training on this data is trivially")
    print("separated by construction and is NOT indicative of real-world sign recognition performance.")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--hard", action="store_true", help="Generate overlapping clusters for stress testing")
    args = parser.parse_args()
    generate_data(args.language, args.samples, args.hard)
