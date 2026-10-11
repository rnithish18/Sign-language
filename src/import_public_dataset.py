"""
import_public_dataset.py
------------------------
Bulk-imports a folder of labeled sign language images (e.g. from Kaggle) into the CSV dataset.
Expects a directory structure where subfolder names are the labels:
dataset_folder/
    A/
        image1.jpg
        image2.jpg
    B/
        ...

Run:
    python -m src.import_public_dataset --language ASL --dataset-dir path/to/dataset
"""

import argparse
import csv
import glob
import os
import sys

import cv2

from src.config import dataset_csv_path, FEATURE_LENGTH
from src.landmark_utils import HandLandmarkExtractor, landmarks_to_csv_row
from src.data_collection import ensure_csv_header

def import_dataset(language: str, dataset_dir: str, max_per_label: int = None):
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    csv_path = dataset_csv_path(language)
    ensure_csv_header(csv_path, feature_len=FEATURE_LENGTH)
    
    # We use static_image_mode=True for processing uncorrelated images
    extractor = HandLandmarkExtractor(static_image_mode=True)
    
    total_processed = 0
    total_skipped = 0
    stats = {}

    print("=" * 60)
    print(f"Importing public dataset to {language} CSV")
    print(f"Dataset root: {dataset_dir}")
    if max_per_label:
        print(f"Subsampling: Max {max_per_label} images per label")
    print("=" * 60)

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Iterate over subdirectories in dataset_dir
        for label_name in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
                
            label = label_name.upper()
            stats[label] = {"processed": 0, "skipped": 0}
            
            # Grab common image extensions
            image_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                image_paths.extend(glob.glob(os.path.join(label_dir, ext)))
                
            print(f"[{label}] Found {len(image_paths)} images. Processing...")
            
            for img_path in image_paths:
                if max_per_label and stats[label]["processed"] >= max_per_label:
                    break
                    
                frame = cv2.imread(img_path)
                if frame is None:
                    stats[label]["skipped"] += 1
                    total_skipped += 1
                    continue
                    
                frame_result = extractor.process(frame)
                if frame_result:
                    writer.writerow(landmarks_to_csv_row(frame_result.feature_vector, label))
                    stats[label]["processed"] += 1
                    total_processed += 1
                else:
                    stats[label]["skipped"] += 1
                    total_skipped += 1

    extractor.close()
    
    print("\n" + "=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)
    print(f"Total images with hands detected and saved: {total_processed}")
    print(f"Total images skipped (no hand/unreadable):  {total_skipped}")
    print("\nPer-label breakdown:")
    for lbl, st in sorted(stats.items()):
        print(f"  {lbl}: {st['processed']} saved, {st['skipped']} skipped")
        
    print(f"\n[DONE] Dataset updated at: {csv_path}")
    print(f"Next step: Train with 'python -m src.train_model --language {language} --source public_dataset'")

def parse_args():
    parser = argparse.ArgumentParser(description="Bulk-import a folder of labeled images into the dataset CSV.")
    parser.add_argument("--language", default="ASL", choices=["ASL", "ISL"])
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset directory (subfolders are labels).")
    parser.add_argument("--max-per-label", type=int, default=None, help="Maximum number of images to process per label (useful for fast testing).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    import_dataset(args.language, args.dataset_dir, args.max_per_label)
