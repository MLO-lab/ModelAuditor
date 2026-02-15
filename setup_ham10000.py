#!/usr/bin/env python3
"""
HAM10000 Dataset Setup Script

This script organizes the HAM10000 dataset into vidir_modern and rosendahl subsets
for use with ModelAuditor.

Prerequisites:
1. Download HAM10000 from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Extract all images (from both HAM10000_images_part_1.zip and HAM10000_images_part_2.zip)
   into data/ham10000/
3. Place HAM10000_metadata.csv in data/ham10000/

Usage:
    python setup_ham10000.py
"""

import os
import shutil
import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    exit(1)


def setup_ham10000(base_dir: str = "data/ham10000", classes: list = None):
    """
    Organize HAM10000 images into vidir_modern and rosendahl subsets.

    Args:
        base_dir: Directory containing HAM10000 images and metadata
        classes: List of diagnosis classes to include (default: ['bkl', 'mel'])
    """
    if classes is None:
        classes = ['bkl', 'mel']

    base_path = Path(base_dir)
    metadata_path = base_path / "HAM10000_metadata.csv"

    # Check for metadata file
    if not metadata_path.exists():
        print(f"Error: HAM10000_metadata.csv not found in {base_dir}")
        print("\nPlease download from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        return False

    # Load metadata
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)

    datasets = ['vidir_modern', 'rosendahl']

    # Create directories
    print("\nCreating directory structure...")
    for dataset in datasets:
        for dx in classes:
            dir_path = base_path / dataset / dx
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created {dir_path}")

    # Organize files
    print("\nOrganizing images...")
    stats = {ds: {dx: 0 for dx in classes} for ds in datasets}
    missing = []

    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]

        for dx in classes:
            image_ids = dataset_df[dataset_df['dx'] == dx]['image_id'].unique()

            for img_id in image_ids:
                src_path = base_path / f'{img_id}.jpg'
                dst_path = base_path / dataset / dx / f'{img_id}.jpg'

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    stats[dataset][dx] += 1
                else:
                    missing.append(img_id)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    for dataset in datasets:
        print(f"\n{dataset}:")
        for dx in classes:
            print(f"  {dx}: {stats[dataset][dx]} images")

    if missing:
        print(f"\nWarning: {len(missing)} images not found in source directory")
        if len(missing) <= 10:
            for img_id in missing:
                print(f"  - {img_id}.jpg")
        else:
            print(f"  First 10: {missing[:10]}")

    print("\nSetup complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Organize HAM10000 dataset into vidir_modern and rosendahl subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_ham10000.py
    python setup_ham10000.py --base-dir /path/to/ham10000
    python setup_ham10000.py --classes bkl mel nv

Download HAM10000 from:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
        """
    )

    parser.add_argument(
        "--base-dir",
        default="data/ham10000",
        help="Directory containing HAM10000 images and metadata (default: data/ham10000)"
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bkl", "mel"],
        help="Diagnosis classes to include (default: bkl mel)"
    )

    args = parser.parse_args()

    success = setup_ham10000(args.base_dir, args.classes)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
