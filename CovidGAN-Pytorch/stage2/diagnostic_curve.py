"""
Stage 2 -- DIAGNOSTIC: per-epoch train vs. test accuracy curve.

Purpose. Our reported models train for a FIXED number of epochs chosen without
looking at the test set (test-blind), which is unbiased. This script is a
separate diagnostic that additionally evaluates the held-out test set after
EVERY epoch, so we can *see* the generalization curve -- when train accuracy
saturates (~100%) and whether test accuracy is still improving, plateaued, or
degrading. That tells us, retroactively, whether the fixed epoch count was
reasonable.

    !!! METHODOLOGICAL NOTE !!!
    The per-epoch test numbers here are for ANALYSIS / illustration ONLY. They
    are NOT used to pick the stopping epoch or any hyperparameter -- doing so
    would be selecting on the test set (leakage) and would bias the reported
    result. The reported models (train_stage2.py / multiseed.py) never see the
    test set during training or model selection. This script changes nothing
    about them; it just logs extra numbers for a figure.

    python -m stage2.diagnostic_curve --unfreeze-blocks 2 --epochs 25

Writes runs/stage2_diagnostic/curve_<mode>.csv (epoch, train_acc, test_acc,
test_covid_recall) for mode in {ad, sa} and prints the curves.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from covidgan.data import CLASS_NAMES, CXRDataset, make_synthetic_items, read_manifest
from covidgan.models import pick_device

from stage2.model import build_stage2_classifier
from stage2.train_stage2 import seed_everything, summarize


@torch.no_grad()
def eval_test(model, loader, device):
    model.eval()
    preds, labs = [], []
    for imgs, l in loader:
        preds.append(model(imgs.to(device)).argmax(1).cpu().numpy())
        labs.append(l.numpy())
    return summarize(np.concatenate(labs), np.concatenate(preds))


def run_mode(mode, real_ds, synth_ds, test_loader, device, args):
    seed_everything(args.seed)
    if mode == "sa":
        train_ds = ConcatDataset([real_ds, synth_ds])
    else:
        train_ds = real_ds
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model = build_stage2_classifier(num_classes=len(CLASS_NAMES),
                                    unfreeze_blocks=args.unfreeze_blocks).to(device)
    optimizer = torch.optim.Adam(model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
                                 betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_acc = correct / total
        test = eval_test(model, test_loader, device)  # DIAGNOSTIC ONLY (see module docstring)
        rows.append((epoch, train_acc, test["accuracy"], test["covid_recall"]))
        print(f"[{mode}] epoch {epoch:>2}/{args.epochs}  train={train_acc*100:6.2f}%  "
              f"test={test['accuracy']*100:6.2f}%  test_covid_recall={test['covid_recall']*100:6.2f}%")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["ad", "sa"], choices=["ad", "sa"])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="runs/stage2_diagnostic")
    ap.add_argument("--epochs", type=int, default=25,
                    help="Run PAST the reported 15 so we can see whether test degrades later.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}  (STAGE 2 DIAGNOSTIC per-epoch train/test curve; "
          f"test evals are analysis-only, not used for model selection)")

    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    real_ds = CXRDataset(train_items, image_size=112, value_range="unit", cache=True)
    synth_ds = CXRDataset(make_synthetic_items(args.synthetic_dir), image_size=112,
                          value_range="unit", cache=True)
    test_loader = DataLoader(CXRDataset(test_items, image_size=112, value_range="unit", cache=True),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        rows = run_mode(mode, real_ds, synth_ds, test_loader, device, args)
        csv_path = out_dir / f"curve_{mode}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_acc", "test_acc", "test_covid_recall"])
            w.writerows(rows)
        # Where did test accuracy actually peak? (reported for the writeup;
        # NOT used to change the reported models.)
        best = max(rows, key=lambda r: r[2])
        print(f"[{mode}] test-acc peak: {best[2]*100:.2f}% at epoch {best[0]} "
              f"(reported models use fixed 15). saved {csv_path}\n")


if __name__ == "__main__":
    main()
