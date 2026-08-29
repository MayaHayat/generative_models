"""
Stage 2 -- SYNTHETIC-ONLY TRANSFER PROBE.

THE QUESTION
============
Does a GAN's synthetic pool carry *transferable* COVID/Normal pathology, or only
a self-consistent "class fingerprint" the AC-GAN imprinted via the label
embedding? CNN-SA accuracy cannot answer this: once you also train on real data
(and especially once the encoder is unfrozen), synthetic images help simply by
adding volume, which masks whether their *class signal* is real.

THE TEST
========
Train the classifier on the synthetic pool ALONE (no real images), then evaluate
on the REAL held-out test split. If the synthetic class signal is real pathology,
accuracy transfers (well above the majority-class floor). If it is a fingerprint,
the classifier learns something that does not exist in real CXRs and lands at/below
chance. Stage 1 ran exactly this probe on the original AC-GAN pool and got 55.2%
accuracy / COVID recall 0.22 -- *below* the 62.5% all-Normal floor (FINDINGS Sec.
8.4). This script reproduces that probe for any pool so the two Stage 2 GANs
(AC-GAN improved vs. projection) are compared on the metric that actually isolates
the fingerprint.

DESIGN NOTES (kept identical across pools so the comparison is fair)
====================================================================
- Classifier: the paper's FROZEN-VGG16 detector (covidgan.models.build_classifier),
  the same feature space the 55% Stage 1 number came from. Only the ~33K-param head
  learns the synthetic class boundary; the probe then asks whether that boundary
  works on real images.
- Test set: the manifest's real `test` split (72 COVID + 120 Normal = 192).
- Majority-class floor (guess all Normal): 120/192 = 62.5%. Beating it is the bar.

USAGE
=====
    python -m stage2.synthetic_only_probe --synthetic-dir data/synth_acgan      --out-dir runs/probe_acgan
    python -m stage2.synthetic_only_probe --synthetic-dir data/synth_projection --out-dir runs/probe_projection
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from covidgan.data import CLASS_NAMES, CXRDataset, make_synthetic_items, read_manifest
from covidgan.metrics import classification_table
from covidgan.models import build_classifier, count_trainable_params, pick_device


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def run(args):
    device = pick_device("cpu" if args.cpu else args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}  seed: {args.seed}  synthetic-dir: {args.synthetic_dir}")

    synth_items = make_synthetic_items(args.synthetic_dir)
    if not synth_items:
        raise SystemExit(f"No synthetic images found under {args.synthetic_dir}")
    test_items = read_manifest(args.manifest, "test")

    train_ds = CXRDataset(synth_items, image_size=112, value_range="unit", cache=args.cache)
    test_ds = CXRDataset(test_items, image_size=112, value_range="unit", cache=args.cache)
    print(f"train (synthetic only): {len(train_ds)}   test (real): {len(test_ds)}")

    workers = 0 if args.cache else args.workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    model = build_classifier(num_classes=len(CLASS_NAMES)).to(device)
    print(f"classifier: frozen VGG16 + head, {count_trainable_params(model):,} trainable params")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        print(f"epoch {epoch+1}/{args.epochs}  loss={loss_sum/total:.4f}  synth_train_acc={correct/total:.2%}")

    preds, labels = predict(model, test_loader, device)
    real_acc = float((preds == labels).mean())
    floor = max(np.bincount(labels)) / len(labels)
    table = classification_table(labels, preds, CLASS_NAMES)

    verdict = ("ABOVE floor -> some transferable signal"
               if real_acc > floor else "AT/BELOW floor -> fingerprint, not pathology")
    summary = (f"SYNTHETIC-ONLY TRANSFER PROBE\n"
               f"synthetic-dir: {args.synthetic_dir}\n"
               f"trained on {len(train_ds)} synthetic, tested on {len(test_ds)} REAL\n"
               f"real-test accuracy: {real_acc:.4f}   majority floor: {floor:.4f}   [{verdict}]\n\n"
               + table + "\n")
    print("\n" + summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_metrics.txt").write_text(summary, encoding="utf-8")
    print(f"wrote {out_dir / 'probe_metrics.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-dir", required=True, help="pool with COVID/ and Normal/ subfolders")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs/probe")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu", action="store_true")
    run(ap.parse_args())
