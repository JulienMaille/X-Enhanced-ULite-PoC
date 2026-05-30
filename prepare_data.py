"""
Prepare NEU-SEG dataset in the format expected by XULite.py.

Downloads from: https://github.com/DHW-Master/NEU_Seg
Converts PNG masks to RLE-encoded CSV files.
Creates folder structure:
    NEU-seg/
    ├── TrainingData/   (train images + NEU_train.csv)
    ├── ValData/        (val images   + NEU_val.csv)
    └── TestData/       (test images  + NEU_test.csv)

Usage:  python prepare_data.py
"""
import os, shutil, urllib.request, zipfile, tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_ZIP = "https://github.com/DHW-Master/NEU_Seg/archive/refs/heads/main.zip"
IMG_SIZE = 200
VAL_SPLIT = 0.2  # 20% of training set for validation
RANDOM_STATE = 42

TASK_DIR = Path("NEU-seg")
TRAIN_DIR = TASK_DIR / "TrainingData"
VAL_DIR = TASK_DIR / "ValData"
TEST_DIR = TASK_DIR / "TestData"


def rle_encode(mask):
    """Encode a binary mask to RLE string (1-indexed, Fortran/column-major).
    
    Matches the rle_decode in XULite.py:
        starts -= 1  → stored starts are 1-indexed
        order='F'    → column-major (Fortran) flattening
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def download_and_extract(target_dir):
    """Download the NEU_Seg repo and extract to target_dir."""
    zip_path = target_dir / "repo.zip"
    print("Downloading NEU_Seg dataset from GitHub...")
    urllib.request.urlretrieve(REPO_ZIP, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    zip_path.unlink()
    return target_dir / "NEU_Seg-main"


def process_split(src_images, src_annotations, dst_dir, csv_name, split_label):
    """Process images and masks into flat image dir + RLE CSV."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    records = []

    img_files = sorted(src_images.glob("*.jpg"))
    for img_path in tqdm(img_files, desc=f"Processing {split_label}"):
        stem = img_path.stem  # e.g. "000201"
        mask_path = src_annotations / f"{stem}.png"

        if not mask_path.exists():
            continue

        # Copy image
        dst_img = dst_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # Load mask (grayscale, single channel)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (IMG_SIZE, IMG_SIZE):
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Binarize (mask may have values 0-255, defect pixels > 0)
        binary = (mask > 0).astype(np.uint8)

        rle = rle_encode(binary)
        if rle:
            records.append({"ImageId": img_path.name, "EncodedPixels": rle})

    df = pd.DataFrame(records)
    csv_path = dst_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} masks written to {csv_path}")
    return df


def main():
    repo_dir = download_and_extract(Path(tempfile.gettempdir()))

    # Source paths from the repo
    train_img_dir = repo_dir / "images" / "training"
    train_ann_dir = repo_dir / "annotations" / "training"
    test_img_dir = repo_dir / "images" / "test"
    test_ann_dir = repo_dir / "annotations" / "test"

    # Validate sources
    for d in [train_img_dir, train_ann_dir, test_img_dir, test_ann_dir]:
        assert d.exists(), f"Missing: {d}"

    print(f"\nSource: {train_img_dir} ({len(list(train_img_dir.glob('*.jpg')))} images)")
    print(f"Source: {test_img_dir} ({len(list(test_img_dir.glob('*.jpg')))} images)")

    # --- Create output directories ---
    if TASK_DIR.exists():
        print(f"\nRemoving existing {TASK_DIR}...")
        shutil.rmtree(TASK_DIR)

    # --- Split training set into train/val ---
    train_stems = sorted(set(p.stem for p in train_img_dir.glob("*.jpg")))
    train_stems_v, val_stems_v = train_test_split(
        train_stems, test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )
    print(f"\nSplitting training set: {len(train_stems_v)} train, {len(val_stems_v)} val")

    # Helper: copy images for a subset of stems
    def copy_subset(stems, src_img_dir, src_ann_dir, dst_dir, csv_name, label):
        dst_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for stem in tqdm(stems, desc=f"  {label}"):
            img_name = f"{stem}.jpg"
            mask_name = f"{stem}.png"
            shutil.copy2(src_img_dir / img_name, dst_dir / img_name)

            mask = cv2.imread(str(src_ann_dir / mask_name), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                binary = (mask > 0).astype(np.uint8)
                cv2.imwrite(str(dst_dir / mask_name), binary * 255)
                rle = rle_encode(binary)
                if rle:
                    records.append({"ImageId": img_name, "EncodedPixels": rle})

        df = pd.DataFrame(records)
        df.to_csv(dst_dir / csv_name, index=False)
        print(f"    {len(df)} records -> {dst_dir / csv_name}")
        return df

    copy_subset(train_stems_v, train_img_dir, train_ann_dir, TRAIN_DIR,
                "NEU_train.csv", "Train")
    copy_subset(val_stems_v, train_img_dir, train_ann_dir, VAL_DIR,
                "NEU_val.csv", "Val")
    copy_subset(
        sorted(set(p.stem for p in test_img_dir.glob("*.jpg"))),
        test_img_dir, test_ann_dir, TEST_DIR,
        "NEU_test.csv", "Test"
    )

    # Clean up downloaded repo
    shutil.rmtree(repo_dir)

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset ready at: {TASK_DIR.resolve()}")
    for d, name in [(TRAIN_DIR, "Train"), (VAL_DIR, "Val"), (TEST_DIR, "Test")]:
        n_img = len(list(d.glob("*.jpg")))
        n_csv = len(pd.read_csv(d / f"NEU_{name.lower()}.csv")) if (d / f"NEU_{name.lower()}.csv").exists() else 0
        print(f"  {name}: {n_img} images, {n_csv} masks in CSV")
    print(f"\nNow run the build_npz step, or use XULite.py directly.")
    print(f"  python -c \"exec(open('XULite.py').read().split('train_loader')[0]); "
          f"build_npz('NEU-seg/TrainingData', 'NEU-seg/TrainingData/NEU_train.csv', 'train.npz', 'train')\"")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
