#!/usr/bin/env python3
"""
Dataset Download Script for ModelAuditor

Downloads datasets that support programmatic access. For datasets requiring
manual download (CheXpert, ChestX-ray14, PolypGen, DDI, SLICE-3D), this script
prints instructions.

Usage:
    python data/download_datasets.py --all
    python data/download_datasets.py --dataset camelyon17
    python data/download_datasets.py --dataset ham10000 --ham10000-zip-dir /path/to/zips
    python data/download_datasets.py --list
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Resolve project root (parent of data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def setup_ham10000(base_dir: str = "data/ham10000", classes: list = None):
    """
    Organize HAM10000 images into vidir_modern and rosendahl subsets.

    Reads HAM10000_metadata.csv from base_dir and copies images into
    clinic-based subdirectories for use with ModelAuditor.

    Args:
        base_dir: Directory containing HAM10000 images and metadata
        classes: List of diagnosis classes to include (default: ['bkl', 'mel'])
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas")
        return False

    if classes is None:
        classes = ['bkl', 'mel']

    base_path = Path(base_dir)
    metadata_path = base_path / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        print(f"Error: HAM10000_metadata.csv not found in {base_dir}")
        print("\nPlease download from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        return False

    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)

    datasets = ['vidir_modern', 'rosendahl']

    print("\nCreating directory structure...")
    for dataset in datasets:
        for dx in classes:
            dir_path = base_path / dataset / dx
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created {dir_path}")

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


def download_camelyon17():
    """Download Camelyon17 via the WILDS package."""
    print("\n=== Camelyon17 (WILDS) ===")
    try:
        from wilds import get_dataset
    except ImportError:
        print("Error: WILDS not installed. Install with:")
        print("  pip install wilds")
        print("  # or: pip install -e '.[medical]'")
        return False

    print("Downloading Camelyon17 via WILDS (this may take a while, ~10 GB)...")
    get_dataset(dataset="camelyon17", download=True)
    print("Done. Dataset cached at ~/.wilds/camelyon17_v1.0/")
    return True


def download_ham10000(zip_dir=None):
    """Set up HAM10000 from pre-downloaded zip files."""
    import zipfile

    print("\n=== HAM10000 ===")
    ham_dir = DATA_DIR / "ham10000"
    ham_dir.mkdir(parents=True, exist_ok=True)

    if zip_dir is None:
        print("HAM10000 requires manual download from Harvard Dataverse:")
        print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print()
        print("Download these files:")
        print("  1. HAM10000_images_part_1.zip")
        print("  2. HAM10000_images_part_2.zip")
        print("  3. HAM10000_metadata (download as CSV)")
        print()
        print("Then run:")
        print(f"  python data/download_datasets.py --dataset ham10000 --ham10000-zip-dir /path/to/downloads")
        return False

    zip_path = Path(zip_dir)
    part1 = zip_path / "HAM10000_images_part_1.zip"
    part2 = zip_path / "HAM10000_images_part_2.zip"
    metadata_candidates = [
        zip_path / "HAM10000_metadata.csv",
        zip_path / "HAM10000_metadata.tab",
    ]

    for f in [part1, part2]:
        if not f.exists():
            print(f"Error: {f} not found.")
            return False

    metadata_src = None
    for candidate in metadata_candidates:
        if candidate.exists():
            metadata_src = candidate
            break
    if metadata_src is None:
        print(f"Error: HAM10000_metadata.csv not found in {zip_dir}")
        return False

    print(f"Extracting {part1.name}...")
    with zipfile.ZipFile(part1, "r") as z:
        z.extractall(ham_dir)

    print(f"Extracting {part2.name}...")
    with zipfile.ZipFile(part2, "r") as z:
        z.extractall(ham_dir)

    dest_csv = ham_dir / "HAM10000_metadata.csv"
    if not dest_csv.exists():
        shutil.copy2(metadata_src, dest_csv)
        print(f"Copied metadata to {dest_csv}")

    print("Creating vidir_modern/rosendahl splits...")
    success = setup_ham10000(str(ham_dir))
    if success:
        print("HAM10000 setup complete.")
    return success


def download_cifar10():
    """Download CIFAR-10 via torchvision."""
    print("\n=== CIFAR-10 ===")
    try:
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("Error: torchvision not installed.")
        return False

    print("Downloading CIFAR-10...")
    CIFAR10(root=str(DATA_DIR), train=False, download=True)
    print(f"Done. Saved to {DATA_DIR}/cifar-10-batches-py/")
    return True


def download_models():
    """Download pretrained models from HuggingFace."""
    print("\n=== Pretrained Models (HuggingFace) ===")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_files = [
        "camelyon17_resnet50_1_224.pt",
        "chexpert_resnet50_1_224.pt",
        "ham10000_resnet50_1_224.pt",
    ]

    for model_file in model_files:
        print(f"Downloading {model_file}...")
        hf_hub_download(
            "lukaskuhndkfz/ModelAuditor",
            model_file,
            local_dir=str(MODELS_DIR),
        )
        print(f"  Saved to {MODELS_DIR / model_file}")

    print("All pretrained models downloaded.")
    return True


def print_manual_instructions():
    """Print instructions for datasets requiring manual download."""
    print("\n" + "=" * 70)
    print("DATASETS REQUIRING MANUAL DOWNLOAD")
    print("=" * 70)

    print("""
--- CheXpert ---
1. Register at: https://stanfordmlgroup.github.io/competitions/chexpert/
2. Download CheXpert-v1.0-small.zip (~11 GB)
3. Extract to data/chexpert/:
   unzip CheXpert-v1.0-small.zip -d data/chexpert/

--- ChestX-ray14 ---
1. Download from: https://www.kaggle.com/datasets/nih-chest-xrays/data (~42 GB)
2. Extract to data/chestxray14/:
   Place Data_Entry_2017.csv and images/ directory.

--- PolypGen ---
1. Register at: https://www.synapse.org/Synapse:syn26376615/wiki/613312
2. Download per-center archives (C1-C5)
3. Extract so that data/data_C{N}/ contains:
   - images_C{N}/
   - masks_C{N}/
   - bbox_C{N}/

--- SIIM-ISIC ---
1. Download pretrained checkpoint from: https://zenodo.org/doi/10.5281/zenodo.10049216
2. Place model files in models/isic/
3. ISIC evaluation data from: https://challenge.isic-archive.com/data
4. Organize as: data/isic/train/{benign,malignant}/

--- DeepDerm ---
1. Download checkpoint from: https://zenodo.org/records/10049217
2. Place as models/deepderm_isic.pth

--- Fitzpatrick17k ---
1. See: https://github.com/mattgroh/fitzpatrick17k
2. Request access and download images + metadata CSV.
3. Place in data/fitzpatrick17k/


""")


def list_datasets():
    """List all datasets and their availability status."""
    print("\nDatasets used in the ModelAuditor paper:\n")
    print(f"{'Dataset':<20} {'Auto-download':<15} {'Status'}")
    print("-" * 60)

    checks = [
        ("CIFAR-10", True, (DATA_DIR / "cifar-10-batches-py").exists()),
        ("Camelyon17", True, Path.home().joinpath(".wilds/camelyon17_v1.0").exists()),
        ("HAM10000", False, (DATA_DIR / "ham10000" / "vidir_modern").exists()
            or (EXAMPLES_DIR / "ham10000" / "vidir_modern").exists()),
        ("CheXpert", False, (DATA_DIR / "chexpert" / "train.csv").exists()),
        ("ChestX-ray14", False, (DATA_DIR / "chestxray14").exists()),
        ("PolypGen", False, (DATA_DIR / "data_C5" / "images_C5").exists()),
        ("ISIC", False, (DATA_DIR / "isic" / "train").exists()),
        ("Fitzpatrick17k", False, (DATA_DIR / "fitzpatrick17k").exists()),
    ]

    for name, auto, exists in checks:
        auto_str = "Yes" if auto else "Manual"
        status = "Found" if exists else "Not found"
        print(f"  {name:<20} {auto_str:<15} {status}")

    print()
    # Check models
    print("Pretrained models:\n")
    model_files = [
        "camelyon17_resnet50_1_224.pt",
        "chexpert_resnet50_1_224.pt",
        "ham10000_resnet50_1_224.pt",
    ]
    for mf in model_files:
        status = "Found" if (MODELS_DIR / mf).exists() else "Not found"
        print(f"  {mf:<40} {status}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for ModelAuditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data/download_datasets.py --list
    python data/download_datasets.py --all
    python data/download_datasets.py --dataset camelyon17
    python data/download_datasets.py --dataset ham10000 --ham10000-zip-dir ~/Downloads
    python data/download_datasets.py --models
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["camelyon17", "ham10000", "cifar10"],
        help="Download a specific dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all auto-downloadable datasets and models",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download pretrained models from HuggingFace",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all datasets and their availability status",
    )
    parser.add_argument(
        "--ham10000-zip-dir",
        type=str,
        default=None,
        help="Directory containing HAM10000 zip files (for automated extraction)",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.list:
        list_datasets()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        download_cifar10()
        download_camelyon17()
        download_ham10000(args.ham10000_zip_dir)
        download_models()
        print_manual_instructions()
        return

    if args.models:
        download_models()
        return

    if args.dataset == "camelyon17":
        download_camelyon17()
    elif args.dataset == "ham10000":
        download_ham10000(args.ham10000_zip_dir)
    elif args.dataset == "cifar10":
        download_cifar10()


if __name__ == "__main__":
    main()
