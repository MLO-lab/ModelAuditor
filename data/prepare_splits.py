#!/usr/bin/env python3
"""
Split Preparation Script for ModelAuditor

Produces CSV manifests defining the exact train/ID-eval/OOD-test splits used in
the paper. Each manifest contains filepath and label columns so that a reviewer
can verify they are using identical data.

Supported datasets:
  - HAM10000:    clinic-based split (vidir_modern vs rosendahl)
  - CheXpert:    frontal AP filtered, last 5000 for eval, multi-label
  - Camelyon17:  WILDS hospital-based split
  - PolypGen:    center-based split (C1-C4 train, C5 OOD test)

Output:
  data/splits/<dataset>_<split>.csv

Usage:
    python data/prepare_splits.py --all
    python data/prepare_splits.py --dataset ham10000
    python data/prepare_splits.py --dataset chexpert
    python data/prepare_splits.py --dataset camelyon17
    python data/prepare_splits.py --dataset polypgen
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"


def prepare_ham10000():
    """
    Create CSV manifests for HAM10000 clinic-based splits.

    Split logic (from download_datasets.py):
      - vidir_modern (Vienna) = ID train/eval
      - rosendahl (Australia) = OOD test
      - Only bkl and mel diagnoses are used

    Output files:
      - ham10000_vidir_modern.csv  (ID split)
      - ham10000_rosendahl.csv     (OOD split)

    Columns: image_id, filepath, dx, label, dataset
    Label encoding: bkl=0, mel=1
    """
    print("\n=== HAM10000 ===")
    ham_dir = DATA_DIR / "ham10000"
    metadata_path = ham_dir / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        print("Download HAM10000 first. See DATASETS.md for instructions.")
        return False

    df = pd.read_csv(metadata_path)
    classes = ["bkl", "mel"]
    label_map = {"bkl": 0, "mel": 1}
    datasets = ["vidir_modern", "rosendahl"]

    for ds_name in datasets:
        subset = df[(df["dataset"] == ds_name) & (df["dx"].isin(classes))].copy()
        subset = subset.drop_duplicates(subset="image_id")
        subset["label"] = subset["dx"].map(label_map)
        subset["filepath"] = subset["image_id"].apply(
            lambda x: f"data/ham10000/{ds_name}/{subset.loc[subset['image_id'] == x, 'dx'].iloc[0]}/{x}.jpg"
        )

        # Rebuild filepath more efficiently
        rows = []
        for _, row in subset.iterrows():
            rows.append(
                {
                    "image_id": row["image_id"],
                    "filepath": f"data/ham10000/{ds_name}/{row['dx']}/{row['image_id']}.jpg",
                    "dx": row["dx"],
                    "label": label_map[row["dx"]],
                    "dataset": ds_name,
                }
            )
        manifest = pd.DataFrame(rows)

        out_path = SPLITS_DIR / f"ham10000_{ds_name}.csv"
        manifest.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} ({len(manifest)} samples)")

        # Print class distribution
        for cls in classes:
            count = (manifest["dx"] == cls).sum()
            print(f"    {cls} (label={label_map[cls]}): {count}")

    return True


def prepare_chexpert():
    """
    Create CSV manifests for CheXpert frontal AP split.

    Filtering (from main.py):
      - Frontal/Lateral == 'Frontal'
      - AP/PA == 'AP'
      - NaN and -1 mapped to 0
      - 5 pathologies: Atelectasis, Consolidation, Cardiomegaly, Pleural Effusion, Edema

    Split:
      - Full filtered set = training pool
      - Last 5000 samples = evaluation subset used by ModelAuditor

    Output files:
      - chexpert_train.csv      (all frontal AP samples)
      - chexpert_eval5k.csv     (last 5000 samples, used for auditing)

    Columns: path, Atelectasis, Consolidation, Cardiomegaly, Pleural Effusion, Edema
    """
    print("\n=== CheXpert ===")
    csv_path = DATA_DIR / "chexpert" / "train.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        print("Download CheXpert first. See DATASETS.md for instructions.")
        return False

    import numpy as np

    df = pd.read_csv(csv_path)
    pathology_cols = [
        "Atelectasis",
        "Consolidation",
        "Cardiomegaly",
        "Pleural Effusion",
        "Edema",
    ]

    # Apply filters (matches main.py:664-665)
    df = df[df["Frontal/Lateral"] == "Frontal"]
    df = df[df["AP/PA"] == "AP"]

    # Map uncertain/missing labels (matches main.py:662-663)
    df = df.replace(np.nan, 0.0)
    df = df.replace(-1.0, 0.0)

    # Clean path column (matches main.py:668)
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0-small/train/", "", regex=False)

    # Build manifest
    manifest = df[["Path"] + pathology_cols].copy()
    manifest = manifest.rename(columns={"Path": "filepath"})
    manifest["filepath"] = "data/chexpert/train/" + manifest["filepath"]

    # Convert label columns to int
    for col in pathology_cols:
        manifest[col] = manifest[col].astype(int)

    # Full training manifest
    out_train = SPLITS_DIR / "chexpert_train.csv"
    manifest.to_csv(out_train, index=False)
    print(f"  Wrote {out_train} ({len(manifest)} samples)")

    # Eval subset: last 5000 (matches main.py:689)
    eval_manifest = manifest.iloc[-5000:].copy()
    out_eval = SPLITS_DIR / "chexpert_eval5k.csv"
    eval_manifest.to_csv(out_eval, index=False)
    print(f"  Wrote {out_eval} ({len(eval_manifest)} samples)")

    # Print label distribution for eval set
    for col in pathology_cols:
        pos = (eval_manifest[col] == 1).sum()
        print(f"    {col}: {pos} positive / {len(eval_manifest) - pos} negative")

    return True


def prepare_camelyon17():
    """
    Create CSV manifest for Camelyon17 WILDS splits.

    Split logic (from CLAUDE.md and main.py):
      - WILDS train subset: 2 hospitals used for training
      - WILDS val subset: 1 hospital used for ID evaluation (loaded by main.py)
      - WILDS test subset: OOD test (held-out hospital)

    Output files:
      - camelyon17_wilds_val.csv    (the subset loaded by --dataset camelyon17)
      - camelyon17_wilds_test.csv   (OOD test)

    Columns: index, hospital, label
    """
    print("\n=== Camelyon17 ===")
    try:
        from wilds import get_dataset
    except ImportError:
        print("Error: WILDS not installed. Install with: pip install wilds")
        return False

    print("  Loading Camelyon17 metadata (not downloading images)...")
    try:
        dataset = get_dataset(dataset="camelyon17", download=False)
    except Exception:
        print("  Dataset not found locally. Downloading metadata...")
        dataset = get_dataset(dataset="camelyon17", download=True)

    for split_name in ["val", "test"]:
        subset = dataset.get_subset(split_name)
        indices = subset.indices if hasattr(subset, "indices") else range(len(subset))

        rows = []
        metadata = dataset.metadata_array
        for idx in indices:
            label = int(dataset.y_array[idx])
            hospital = int(metadata[idx, 0]) if metadata is not None else -1
            rows.append({"index": int(idx), "hospital": hospital, "label": label})

        manifest = pd.DataFrame(rows)
        out_path = SPLITS_DIR / f"camelyon17_wilds_{split_name}.csv"
        manifest.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} ({len(manifest)} samples)")

        # Print distribution
        for label_val in sorted(manifest["label"].unique()):
            count = (manifest["label"] == label_val).sum()
            label_name = "normal" if label_val == 0 else "tumor"
            print(f"    {label_name} (label={label_val}): {count}")

    return True


def prepare_polypgen():
    """
    Create CSV manifests for PolypGen center-based splits.

    Split logic (from CLAUDE.md):
      - C1, C2, C3, C4 = training centers
      - C5 = OOD test (John Radcliffe Hospital, Oxford)

    Output files (segmentation):
      - polypgen_train_seg.csv    (C1-C4 images with masks)
      - polypgen_test_seg.csv     (C5 images with masks)

    Output files (detection):
      - polypgen_train_det.csv    (C1-C4 images with bbox files)
      - polypgen_test_det.csv     (C5 images with bbox files)

    Columns: filepath, mask_path (seg) or bbox_path (det), center
    """
    print("\n=== PolypGen ===")

    train_centers = [1, 2, 3, 4]
    test_centers = [5]

    # Segmentation manifests
    for split_name, centers in [("train", train_centers), ("test", test_centers)]:
        rows = []
        for center_id in centers:
            images_dir = DATA_DIR / f"data_C{center_id}" / f"images_C{center_id}"
            masks_dir = DATA_DIR / f"data_C{center_id}" / f"masks_C{center_id}"

            if not images_dir.exists():
                print(f"  Warning: {images_dir} not found, skipping C{center_id}")
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                mask_name = img_path.stem + "_mask.jpg"
                mask_path = masks_dir / mask_name

                # Handle edge case with trailing bracket (matches main.py:855-858)
                if not mask_path.exists() and img_path.stem.endswith("]"):
                    alt_stem = img_path.stem[:-1]
                    alt_mask = masks_dir / (alt_stem + "_mask.jpg")
                    if alt_mask.exists():
                        mask_path = alt_mask

                if mask_path.exists():
                    rows.append(
                        {
                            "filepath": str(img_path.relative_to(PROJECT_ROOT)),
                            "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                            "center": center_id,
                        }
                    )

        if rows:
            manifest = pd.DataFrame(rows)
            out_path = SPLITS_DIR / f"polypgen_{split_name}_seg.csv"
            manifest.to_csv(out_path, index=False)
            print(f"  Wrote {out_path} ({len(manifest)} samples)")
            for c in sorted(manifest["center"].unique()):
                count = (manifest["center"] == c).sum()
                print(f"    C{c}: {count} images with masks")

    # Detection manifests
    for split_name, centers in [("train", train_centers), ("test", test_centers)]:
        rows = []
        for center_id in centers:
            images_dir = DATA_DIR / f"data_C{center_id}" / f"images_C{center_id}"
            bbox_dir = DATA_DIR / f"data_C{center_id}" / f"bbox_C{center_id}"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                bbox_name = img_path.stem + "_mask.txt"
                bbox_path = bbox_dir / bbox_name

                if bbox_path.exists():
                    rows.append(
                        {
                            "filepath": str(img_path.relative_to(PROJECT_ROOT)),
                            "bbox_path": str(bbox_path.relative_to(PROJECT_ROOT)),
                            "center": center_id,
                        }
                    )

        if rows:
            manifest = pd.DataFrame(rows)
            out_path = SPLITS_DIR / f"polypgen_{split_name}_det.csv"
            manifest.to_csv(out_path, index=False)
            print(f"  Wrote {out_path} ({len(manifest)} samples)")
            for c in sorted(manifest["center"].unique()):
                count = (manifest["center"] == c).sum()
                print(f"    C{c}: {count} images with bboxes")

    found_any = any(
        (DATA_DIR / f"data_C{i}" / f"images_C{i}").exists() for i in range(1, 6)
    )
    if not found_any:
        print("  No PolypGen data found. Download from Synapse first.")
        print("  See DATASETS.md for instructions.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare exact dataset splits for ModelAuditor paper reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Produces CSV manifests in data/splits/ with the exact splits used in the paper.

Examples:
    python data/prepare_splits.py --all
    python data/prepare_splits.py --dataset ham10000
    python data/prepare_splits.py --dataset chexpert
    python data/prepare_splits.py --dataset camelyon17
    python data/prepare_splits.py --dataset polypgen
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ham10000", "chexpert", "camelyon17", "polypgen"],
        help="Prepare splits for a specific dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare splits for all datasets",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    preparers = {
        "ham10000": prepare_ham10000,
        "chexpert": prepare_chexpert,
        "camelyon17": prepare_camelyon17,
        "polypgen": prepare_polypgen,
    }

    if args.all:
        results = {}
        for name, func in preparers.items():
            try:
                results[name] = func()
            except Exception as e:
                print(f"  Error preparing {name}: {e}")
                results[name] = False

        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        for name, success in results.items():
            status = "OK" if success else "SKIPPED (data not found)"
            print(f"  {name:<15} {status}")
        print(f"\nManifests saved to: {SPLITS_DIR}/")
        return

    if args.dataset:
        preparers[args.dataset]()
        print(f"\nManifest(s) saved to: {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
