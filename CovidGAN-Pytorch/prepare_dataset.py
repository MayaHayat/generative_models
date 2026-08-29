"""
Build the train/test split described in Sec. II-A of the paper from raw
dataset folders, and write it to a manifest CSV that every other script
reads from.

Example (matching the layout already present in this workspace):

    python prepare_dataset.py \
        --ieee-covid-root ../covid-chestxray-dataset \
        --normal-dir /path/to/COVID-19_Radiography_Dataset/Normal \
        --out-dir data

Any of --ieee-covid-root, --extra-covid-dir, --normal-dir may be repeated /
omitted; at minimum you need some COVID-CXR source and some Normal-CXR
source. The paper used 403 COVID-CXR + 721 Normal-CXR after merging three
public sources and de-duplicating -- see README.md for where to get them.

The default multi-source collection draws COVID and Normal from *different*
datasets, which lets a classifier exploit cross-source artifacts (resolution,
borders, text, brightness) instead of lung pathology and inflates accuracy.
Pass --same-source-root instead to draw both classes from ONE dataset's
per-class subfolders (e.g. the Kaggle Radiography Database's COVID/ + Normal/),
as an A/B to measure how much of the accuracy is real signal vs. source bias:

    python prepare_dataset.py \
        --same-source-root /path/to/COVID-19_Radiography_Dataset \
        --max-covid 403 --max-normal 721 --out-dir data
"""
import argparse
import csv
import random
from pathlib import Path

from covidgan.data import (
    CLASS_NAMES,
    dedupe,
    load_ieee_covid_chestxray,
    load_image_folder,
    load_same_source,
    stratified_split,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ieee-covid-root", action="append", default=[],
                     help="Root of an ieee8023/covid-chestxray-dataset checkout (metadata.csv + images/).")
    ap.add_argument("--extra-covid-dir", action="append", default=[],
                     help="Flat folder of additional COVID-CXR images.")
    ap.add_argument("--normal-dir", action="append", default=[],
                     help="Flat folder of Normal-CXR images (e.g. Kaggle Radiography Database 'Normal' class).")
    ap.add_argument("--same-source-root", action="append", default=[],
                     help="Root of a SINGLE dataset that ships per-class subfolders (e.g. the Kaggle "
                          "COVID-19 Radiography Database, with COVID/ and Normal/ under one root). Both "
                          "classes are collected from here with one processing pipeline, removing the "
                          "cross-source shortcut that inflates accuracy. Repeatable; combines with the "
                          "other sources if you pass them too.")
    ap.add_argument("--covid-subdir", default="COVID",
                     help="Per-class COVID subfolder name under --same-source-root (default: COVID).")
    ap.add_argument("--normal-subdir", default="Normal",
                     help="Per-class Normal subfolder name under --same-source-root (default: Normal).")
    ap.add_argument("--out-dir", default="data", help="Where to write manifest.csv")
    ap.add_argument("--test-covid", type=int, default=72, help="Paper default: 72")
    ap.add_argument("--test-normal", type=int, default=120, help="Paper default: 120")
    ap.add_argument("--test-frac", type=float, default=0.17,
                     help="Used instead of --test-covid/--test-normal if the dataset is too small for those exact counts.")
    ap.add_argument("--max-covid", type=int, default=None,
                     help="Randomly subsample down to this many COVID-CXR images (after dedup) before splitting. "
                          "Pass 403 to match the paper's exact scale -- public sources like the Kaggle Radiography "
                          "Database are far larger than what the paper used, and training on all of it defeats the "
                          "paper's small-data premise.")
    ap.add_argument("--max-normal", type=int, default=None,
                     help="Randomly subsample down to this many Normal-CXR images (after dedup). Pass 721 to match "
                          "the paper's exact scale.")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    covid_paths = []
    for root in args.ieee_covid_root:
        covid_paths += load_ieee_covid_chestxray(root)
    for d in args.extra_covid_dir:
        covid_paths += load_image_folder(d)

    normal_paths = []
    for d in args.normal_dir:
        normal_paths += load_image_folder(d)

    for root in args.same_source_root:
        ss_covid, ss_normal = load_same_source(root, args.covid_subdir, args.normal_subdir)
        print(f"Same-source root {root}: {len(ss_covid)} COVID-CXR, {len(ss_normal)} Normal-CXR "
              f"(both from {args.covid_subdir}/ and {args.normal_subdir}/, one pipeline).")
        covid_paths += ss_covid
        normal_paths += ss_normal

    if not covid_paths:
        raise SystemExit("No COVID-CXR images found -- pass --ieee-covid-root and/or --extra-covid-dir.")
    if not normal_paths:
        raise SystemExit("No Normal-CXR images found -- pass --normal-dir.")

    print(f"Collected {len(covid_paths)} COVID-CXR, {len(normal_paths)} Normal-CXR before dedup.")
    covid_paths = dedupe(covid_paths)
    normal_paths = dedupe(normal_paths)
    print(f"After hash-based de-duplication: {len(covid_paths)} COVID-CXR, {len(normal_paths)} Normal-CXR.")

    rng = random.Random(args.seed)
    if args.max_covid is not None and len(covid_paths) > args.max_covid:
        covid_paths = rng.sample(covid_paths, args.max_covid)
    if args.max_normal is not None and len(normal_paths) > args.max_normal:
        normal_paths = rng.sample(normal_paths, args.max_normal)
    if args.max_covid is not None or args.max_normal is not None:
        print(f"After subsampling: {len(covid_paths)} COVID-CXR, {len(normal_paths)} Normal-CXR.")

    test_covid = args.test_covid if len(covid_paths) >= args.test_covid + 10 else None
    test_normal = args.test_normal if len(normal_paths) >= args.test_normal + 10 else None
    if test_covid is None or test_normal is None:
        print("Dataset smaller than the paper's; falling back to a "
              f"{args.test_frac:.0%} fractional split instead of exact paper counts.")

    split = stratified_split(
        covid_paths, normal_paths,
        test_covid=test_covid, test_normal=test_normal,
        test_frac=args.test_frac, seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        for path, label in split.train:
            writer.writerow([str(path), CLASS_NAMES[label], "train"])
        for path, label in split.test:
            writer.writerow([str(path), CLASS_NAMES[label], "test"])

    n_train_covid = sum(1 for _, l in split.train if CLASS_NAMES[l] == "covid")
    n_train_normal = sum(1 for _, l in split.train if CLASS_NAMES[l] == "normal")
    print(f"\nWrote {manifest_path}")
    print(f"  train: {len(split.train)}  (covid {n_train_covid}, normal {n_train_normal})")
    print(f"  test:  {len(split.test)}  (covid {len(split.test) - sum(1 for _,l in split.test if CLASS_NAMES[l]=='normal')}, "
          f"normal {sum(1 for _,l in split.test if CLASS_NAMES[l]=='normal')})")


if __name__ == "__main__":
    main()
