"""
DDSM/CBIS-DDSM data pipeline: join pre-extracted ROI patches against the
CBIS-DDSM case-description CSVs to get benign/malignant pathology labels,
using each CSV's own train/test split (the standard CBIS-DDSM benchmark
protocol) rather than re-randomizing it.

Expected ROI filenames (the standard CBIS-DDSM ROI-patch naming convention):
    P_<patient_id>_<LEFT|RIGHT>_<CC|MLO>_<mass|calcification>_<abnormality_id>.png
e.g. P_01152_RIGHT_MLO_mass_3.png

Expected CSVs: the four standard CBIS-DDSM case-description files
(mass_case_description_train_set.csv, mass_case_description_test_set.csv,
calc_case_description_train_set.csv, calc_case_description_test_set.csv),
each with (among others) patient_id, left or right breast, image view,
abnormality id, abnormality type, and pathology columns.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

BENIGN_LABEL = 0
MALIGNANT_LABEL = 1
CLASS_NAMES = ["benign", "malignant"]

FILENAME_RE = re.compile(
    r"^(P_\d+)_(LEFT|RIGHT)_(CC|MLO)_(mass|calcification)_(\d+)\.png$",
    re.IGNORECASE,
)

PATHOLOGY_TO_LABEL = {
    "MALIGNANT": MALIGNANT_LABEL,
    "BENIGN": BENIGN_LABEL,
    "BENIGN_WITHOUT_CALLBACK": BENIGN_LABEL,
}


def parse_roi_filename(path: Path) -> Optional[dict]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    patient_id, side, view, abn_type, abn_id = m.groups()
    return {
        "key": f"{patient_id}_{side.upper()}_{view.upper()}_{abn_type.lower()}_{int(abn_id)}",
        "patient_id": patient_id,
        "abnormality_type": abn_type.lower(),
    }


def _normalize_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {raw: raw.strip().lower().replace(" ", "_") for raw in fieldnames}


def load_case_description_csv(csv_path: str, split: str) -> Dict[str, Tuple[int, str]]:
    """Read one CBIS-DDSM case-description CSV -> {join_key: (label, split)}
    for every row with a recognized pathology value."""
    entries = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        colmap = _normalize_columns(reader.fieldnames)
        for raw_row in reader:
            row = {colmap[k]: v for k, v in raw_row.items() if k in colmap}
            pathology = (row.get("pathology") or "").strip().upper()
            label = PATHOLOGY_TO_LABEL.get(pathology)
            if label is None:
                continue  # unrecognized/missing pathology value -- skip

            patient_id = (row.get("patient_id") or "").strip()
            side = (row.get("left_or_right_breast") or "").strip().upper()
            view = (row.get("image_view") or "").strip().upper()
            abn_type = (row.get("abnormality_type") or "").strip().lower()
            abn_id_raw = (row.get("abnormality_id") or "").strip()
            try:
                abn_id = int(float(abn_id_raw))
            except ValueError:
                continue

            key = f"{patient_id}_{side}_{view}_{abn_type}_{abn_id}"
            entries[key] = (label, split)
    return entries


def build_ddsm_manifest(
    roi_dir: str,
    mass_train_csv: Optional[str] = None,
    mass_test_csv: Optional[str] = None,
    calc_train_csv: Optional[str] = None,
    calc_test_csv: Optional[str] = None,
) -> Tuple[List[Tuple[Path, int, str]], List[Path]]:
    """Join every ROI PNG in roi_dir against the given CSVs by
    (patient, side, view, abnormality type, abnormality id). Pass only the
    mass_*/calc_* pair for the abnormality type(s) you want -- e.g. omit
    calc_train_csv/calc_test_csv entirely to restrict to mass ROIs only.
    Files of an abnormality type you didn't provide CSVs for are skipped
    (not scanned/counted as unmatched) rather than cluttering the report.
    Returns (matched [(path, label, split)], unmatched [path])."""
    allowed_types = set()
    lookup: Dict[str, Tuple[int, str]] = {}
    if mass_train_csv or mass_test_csv:
        if not (mass_train_csv and mass_test_csv):
            raise ValueError("pass both --mass-train-csv and --mass-test-csv together, or neither")
        lookup.update(load_case_description_csv(mass_train_csv, "train"))
        lookup.update(load_case_description_csv(mass_test_csv, "test"))
        allowed_types.add("mass")
    if calc_train_csv or calc_test_csv:
        if not (calc_train_csv and calc_test_csv):
            raise ValueError("pass both --calc-train-csv and --calc-test-csv together, or neither")
        lookup.update(load_case_description_csv(calc_train_csv, "train"))
        lookup.update(load_case_description_csv(calc_test_csv, "test"))
        allowed_types.add("calcification")
    if not allowed_types:
        raise ValueError("pass at least one CSV pair: --mass-train-csv/--mass-test-csv and/or "
                          "--calc-train-csv/--calc-test-csv")

    matched: List[Tuple[Path, int, str]] = []
    unmatched: List[Path] = []
    for path in sorted(Path(roi_dir).glob("*.png")):
        parsed = parse_roi_filename(path)
        if parsed is None:
            unmatched.append(path)
            continue
        if parsed["abnormality_type"] not in allowed_types:
            continue  # not an abnormality type we have labels for -- silently skip, not "unmatched"
        entry = lookup.get(parsed["key"])
        if entry is None:
            unmatched.append(path)
            continue
        label, split = entry
        matched.append((path, label, split))
    return matched, unmatched


def write_manifest(manifest_path: str, matched: Sequence[Tuple[Path, int, str]]):
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        for path, label, split in matched:
            writer.writerow([str(path), CLASS_NAMES[label], split])


def read_manifest(manifest_path: str, split: str) -> List[Tuple[Path, int]]:
    name_to_label = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    items = []
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            items.append((Path(row["path"]), name_to_label[row["label"]]))
    return items


class ROIDataset(Dataset):
    """Wraps a list of (path, label) pairs. ROI patches are grayscale PNGs of
    any native size/bit-depth (resized to `image_size` and channel-replicated
    to RGB here, regardless of source resolution -- e.g. DDSM's native
    384x384 works fine and gets downsized like anything else). `value_range`
    picks pixel scaling: GAN training normalizes to [-1, 1], the classifier
    to [0, 1]. Default `image_size=112` matches the generator/discriminator
    architecture in models.py -- don't raise it without redesigning those."""

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
        img = _load_as_uint8_grayscale(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        arr = torch.from_numpy(np.array(img, dtype="float32")).permute(2, 0, 1) / 255.0
        if self.value_range == "tanh":
            arr = arr * 2.0 - 1.0
        return arr, label


def _load_as_uint8_grayscale(path: Path) -> Image.Image:
    """CBIS-DDSM ROI patches are commonly stored as 16-bit grayscale PNGs
    (mode 'I' / 'I;16', e.g. uint16 values in [20745, 65535], not [0, 255]).
    PIL's Image.convert('RGB') does not rescale 'I'/'I;16' images -- it
    truncates to 8 bits, which for values this large collapses almost every
    pixel to near-white with sharp artifacts wherever the low byte happens
    to wrap low. Per-image min-max normalize to uint8 first so the actual
    tissue contrast survives the conversion."""
    img = Image.open(path)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N") or np.array(img).dtype != np.uint8:
        arr = np.array(img).astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    return img.convert("L")


def make_synthetic_items(root: str) -> List[Tuple[Path, int]]:
    """Read a synthetic image pool laid out as <root>/benign/*.png and
    <root>/malignant/*.png (as written by generate_synthetic.py)."""
    root = Path(root)
    items = []
    for name, label in [("benign", BENIGN_LABEL), ("malignant", MALIGNANT_LABEL)]:
        for p in sorted((root / name).glob("*.png")):
            items.append((p, label))
    return items
