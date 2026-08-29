"""
Stage 1 -- multi-seed CNN-AD vs CNN-SA comparison (frozen VGG16, no BN head).

The single-run Stage 1 result reported CNN-SA as "flat" (90.62 -> 90.10), but a
single train/eval on 192 test images has ~0.5%/image quantization plus
seed-to-seed variance, so a one-off near-zero difference could be noise. This
driver retrains the *faithful Stage 1 classifier* (frozen ImageNet VGG16 + the
paper's plain head, no BatchNorm -- i.e. covidgan.models.build_classifier) from
scratch for several seeds in BOTH modes and reports mean +/- std of accuracy and
COVID recall, so "flat" can be stated as a distribution rather than one number.

    python stage1_multiseed.py --seeds 0 1 2 --synthetic-dir data/synthetic

Writes an aggregate summary to runs/stage1_multiseed/summary.json and prints a
table. This is the Stage 1 analogue of stage2/multiseed.py.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from covidgan.data import CLASS_NAMES, CXRDataset, make_synthetic_items, read_manifest
from covidgan.models import build_classifier, pick_device

COVID = CLASS_NAMES.index("covid")  # class index 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    accuracy = float((preds == labels).mean())
    covid_mask = labels == COVID
    covid_recall = float((preds[covid_mask] == COVID).mean())
    return accuracy, covid_recall


def train_one(mode: str, seed: int, args, device):
    set_seed(seed)
    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    train_ds = CXRDataset(train_items, image_size=112, value_range="unit", cache=True)
    test_ds = CXRDataset(test_items, image_size=112, value_range="unit", cache=True)

    if mode == "sa":
        synth_items = make_synthetic_items(args.synthetic_dir)
        if not synth_items:
            raise SystemExit(f"No synthetic images under {args.synthetic_dir}")
        synth_ds = CXRDataset(synth_items, image_size=112, value_range="unit", cache=True)
        train_data = ConcatDataset([train_ds, synth_ds])
    else:
        train_data = train_ds

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_classifier(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.999)
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    acc, rec = evaluate(model, test_loader, device)
    print(f"  [{mode}] seed {seed}: acc={acc:.4f}  covid_recall={rec:.4f}")
    return acc, rec


def _stats(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", default="data/synthetic")
    ap.add_argument("--out-root", default="runs/stage1_multiseed")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device("cpu" if args.cpu else args.device)
    print(f"device: {device}  seeds: {args.seeds}  epochs: {args.epochs}")

    results = {"ad": {"accuracy": [], "covid_recall": []},
               "sa": {"accuracy": [], "covid_recall": []}}
    for seed in args.seeds:
        for mode in ("ad", "sa"):
            acc, rec = train_one(mode, seed, args, device)
            results[mode]["accuracy"].append(acc)
            results[mode]["covid_recall"].append(rec)

    summary = {}
    for mode in ("ad", "sa"):
        summary[mode] = {
            "accuracy": _stats(results[mode]["accuracy"]),
            "covid_recall": _stats(results[mode]["covid_recall"]),
            "n_runs": len(args.seeds),
            "seeds": args.seeds,
        }
    summary["lift"] = {
        "accuracy": summary["sa"]["accuracy"]["mean"] - summary["ad"]["accuracy"]["mean"],
        "covid_recall": summary["sa"]["covid_recall"]["mean"] - summary["ad"]["covid_recall"]["mean"],
    }

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Stage 1 multi-seed (frozen VGG16) ===")
    for mode in ("ad", "sa"):
        a, r = summary[mode]["accuracy"], summary[mode]["covid_recall"]
        print(f"CNN-{mode.upper()}: acc {a['mean']*100:.2f}% +/- {a['std']*100:.2f}   "
              f"covid_recall {r['mean']*100:.2f}% +/- {r['std']*100:.2f}")
    print(f"augmentation lift: acc {summary['lift']['accuracy']*100:+.2f} pts   "
          f"covid_recall {summary['lift']['covid_recall']*100:+.2f} pts")
    print(f"\nwrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
