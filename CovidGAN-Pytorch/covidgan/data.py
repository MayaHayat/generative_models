"""
Dataset assembly for CovidGAN, matching Sec. II-A of the paper:

  1. Pull COVID-CXR images from the IEEE covid-chestxray-dataset (metadata.csv
     driven, filtered to PA/AP chest views) and Normal-CXR images from a
     folder of non-COVID frontal CXRs (e.g. the Kaggle COVID-19 Radiography
     Database's "Normal" class).
  2. Merge and drop near-duplicate images with a perceptual average-hash.
  3. Stratified split into train/test, defaulting to the paper's own counts
     (331/72 COVID, 601/120 Normal) when there is enough data, else a
     fraction-based split.

Only steps that need the raw dataset folders on disk are here; everything
else (GAN training, classifier training) consumes the resulting file lists.
"""
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

COVID_LABEL = 0
NORMAL_LABEL = 1
CLASS_NAMES = ["covid", "normal"]

IMG_EXTS = {".png", ".jpg", ".jpeg"}


# --------------------------------------------------------------------------
# Source-specific collection
# --------------------------------------------------------------------------

def load_ieee_covid_chestxray(root: str, views: Sequence[str] = ("PA", "AP")) -> List[Path]:
    """Collect COVID-19 CXR image paths from the ieee8023/covid-chestxray-dataset
    layout: <root>/metadata.csv + <root>/images/<filename>.
    """
    root = Path(root)
    metadata_path = root / "metadata.csv"
    images_dir = root / "images"
    paths: List[Path] = []
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            finding = row.get("finding", "")
            view = row.get("view", "")
            if "COVID-19" not in finding:
                continue
            if views and view not in views:
                continue
            fname = row.get("filename", "")
            fpath = images_dir / fname
            if fpath.suffix.lower() in IMG_EXTS and fpath.exists():
                paths.append(fpath)
    return paths


SKIP_DIR_NAMES = {"mask", "masks", "label", "labels", "lung masks", "lung_masks"}


def load_image_folder(root: str) -> List[Path]:
    """Collect every image under a folder (e.g. a Kaggle 'Normal' class
    directory, or a manually curated COVID-CXR folder), skipping any
    subdirectory that looks like segmentation masks/labels rather than CXR
    photos -- several public CXR datasets (including the Kaggle COVID-19
    Radiography Database) ship a masks/ folder alongside images/ with the
    same filenames, which would otherwise silently double-count as images.
    """
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
        and not (SKIP_DIR_NAMES & {part.lower() for part in p.parts})
    )


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def average_hash(path: Path, hash_size: int = 8) -> int:
    """Lightweight perceptual hash (no extra dependency): downscale to a
    hash_size x hash_size grayscale thumbnail, threshold against the mean.
    Stands in for the paper's "Image Hashing method" duplicate removal.
    """
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


def dedupe(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    unique = []
    for p in paths:
        try:
            h = average_hash(p)
        except Exception:
            continue
        if h in seen:
            continue
        seen.add(h)
        unique.append(p)
    return unique


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

@dataclass
class Split:
    train: List[Tuple[Path, int]]
    test: List[Tuple[Path, int]]


def stratified_split(
    covid_paths: Sequence[Path],
    normal_paths: Sequence[Path],
    test_covid: Optional[int] = None,
    test_normal: Optional[int] = None,
    test_frac: float = 0.17,
    seed: int = 999,
) -> Split:
    rng = random.Random(seed)
    covid = list(covid_paths)
    normal = list(normal_paths)
    rng.shuffle(covid)
    rng.shuffle(normal)

    n_test_covid = test_covid if test_covid is not None else round(len(covid) * test_frac)
    n_test_normal = test_normal if test_normal is not None else round(len(normal) * test_frac)
    n_test_covid = min(n_test_covid, len(covid))
    n_test_normal = min(n_test_normal, len(normal))

    test = [(p, COVID_LABEL) for p in covid[:n_test_covid]] + \
           [(p, NORMAL_LABEL) for p in normal[:n_test_normal]]
    train = [(p, COVID_LABEL) for p in covid[n_test_covid:]] + \
            [(p, NORMAL_LABEL) for p in normal[n_test_normal:]]
    rng.shuffle(train)
    rng.shuffle(test)
    return Split(train=train, test=test)


# --------------------------------------------------------------------------
# torch Dataset
# --------------------------------------------------------------------------

class CXRDataset(Dataset):
    """Wraps a list of (path, label) pairs. `value_range` picks the pixel
    scaling: GAN training normalizes to [-1, 1] (Sec. III-B), the classifier
    normalizes to [0, 1] (Sec. II-B)."""

    def __init__(self, items: Sequence[Tuple[Path, int]], image_size: int = 112,
                 value_range: str = "tanh"):
        assert value_range in ("tanh", "unit")
        self.items = list(items)
        self.image_size = image_size
        self.value_range = value_range

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        arr = torch.from_numpy(np.array(img, dtype="float32")).permute(2, 0, 1) / 255.0
        if self.value_range == "tanh":
            arr = arr * 2.0 - 1.0
        return arr, label


def read_manifest(manifest_path: str, split: str) -> List[Tuple[Path, int]]:
    """Read the manifest.csv written by prepare_dataset.py and return the
    (path, label) items for the requested split ('train' or 'test')."""
    name_to_label = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            items.append((Path(row["path"]), name_to_label[row["label"]]))
    return items


def make_synthetic_items(root: str) -> List[Tuple[Path, int]]:
    """Read a synthetic image pool laid out as <root>/covid/*.png and
    <root>/normal/*.png (as written by generate_synthetic.py)."""
    root = Path(root)
    items = []
    for name, label in [("covid", COVID_LABEL), ("normal", NORMAL_LABEL)]:
        for p in sorted((root / name).glob("*.png")):
            items.append((p, label))
    return items
