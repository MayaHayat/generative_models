"""
Join pre-extracted CBIS-DDSM ROI patches against the case-description CSVs
to build data/manifest.csv (path, label, split), using each CSV's own
train/test assignment.

Pass the mass CSVs, the calc CSVs, or both -- e.g. to restrict to mass-only:

    python prepare_ddsm.py \
        --roi-dir /content/ddsm_rois/rois \
        --mass-train-csv /content/drive/MyDrive/Thesis/tabular-dataset/mass_case_description_train_set.csv \
        --mass-test-csv  /content/drive/MyDrive/Thesis/tabular-dataset/mass_case_description_test_set.csv \
        --out-dir data

(omit --calc-train-csv/--calc-test-csv entirely -- calcification ROIs are
then skipped rather than reported as unmatched.) Pass all four to include
both abnormality types.

Prints a match-rate report -- if the match rate is low, the CSV's column
names or the ROI filenames likely don't follow the standard CBIS-DDSM
convention this script assumes; check the printed sample of unmatched
filenames and keys before trusting the manifest.
"""
import argparse
from collections import Counter
from pathlib import Path

from ddsm_acgan.data import build_ddsm_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi-dir", required=True, help="Folder of P_<id>_<side>_<view>_<type>_<n>.png ROI patches.")
    ap.add_argument("--mass-train-csv", default=None)
    ap.add_argument("--mass-test-csv", default=None)
    ap.add_argument("--calc-train-csv", default=None)
    ap.add_argument("--calc-test-csv", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    try:
        matched, unmatched = build_ddsm_manifest(
            roi_dir=args.roi_dir,
            mass_train_csv=args.mass_train_csv,
            mass_test_csv=args.mass_test_csv,
            calc_train_csv=args.calc_train_csv,
            calc_test_csv=args.calc_test_csv,
        )
    except ValueError as e:
        raise SystemExit(str(e))

    total = len(matched) + len(unmatched)
    match_rate = len(matched) / total if total else 0.0
    print(f"Matched {len(matched)}/{total} ROI files to a pathology label ({match_rate:.1%}).")

    if unmatched:
        print(f"\n{len(unmatched)} unmatched files -- first 10 examples:")
        for p in unmatched[:10]:
            print(f"  {p.name}")
        if match_rate < 0.5:
            print(
                "\nMatch rate is low -- likely causes: CSV column names differ from the "
                "standard CBIS-DDSM schema (patient_id / left or right breast / image view / "
                "abnormality id / abnormality type / pathology), or ROI filenames don't follow "
                "P_<id>_<side>_<view>_<mass|calcification>_<n>.png. Inspect the CSV header and "
                "a few filenames above before trusting the manifest."
            )

    if not matched:
        raise SystemExit("No ROI files matched a pathology label -- nothing to write.")

    counts = Counter((label, split) for _, label, split in matched)
    from ddsm_acgan.data import CLASS_NAMES
    print("\nMatched breakdown:")
    for split in ("train", "test"):
        for label, name in enumerate(CLASS_NAMES):
            print(f"  {split:<6} {name:<10} {counts.get((label, split), 0)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    write_manifest(str(manifest_path), matched)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
