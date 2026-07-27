"""
Train and evaluate the benign/malignant detection CNN.

  --mode ad   trained on real (actual) CBIS-DDSM ROIs only.
  --mode sa   trained on real + the synthetic pool from generate_synthetic.py.

Both modes evaluate on the same real, held-out CBIS-DDSM test split (its own
official split, not a re-randomized one), and report precision/recall/F1/
specificity per class plus macro/weighted averages, a confusion matrix, and
-- for --mode sa -- a PCA scatter of penultimate-layer features colored by
class and real/synthetic origin.

    python train_classifier.py --manifest data/manifest.csv --mode ad --out-dir runs/cnn_ad
    python train_classifier.py --manifest data/manifest.csv --mode sa \
        --synthetic-dir data/synthetic --out-dir runs/cnn_sa
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ddsm_acgan.data import CLASS_NAMES, ROIDataset, make_synthetic_items, read_manifest
from ddsm_acgan.metrics import classification_table, plot_confusion_matrix, plot_pca
from ddsm_acgan.models import build_classifier, count_params, count_trainable_params


@torch.no_grad()
def extract_features(model: nn.Sequential, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        f = model[:6](imgs)
        feats.append(f.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}  mode: {args.mode}")

    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    train_ds = ROIDataset(train_items, image_size=112, value_range="unit")
    test_ds = ROIDataset(test_items, image_size=112, value_range="unit")

    synth_ds = None
    if args.mode == "sa":
        if not args.synthetic_dir:
            raise SystemExit("--mode sa requires --synthetic-dir (see generate_synthetic.py)")
        synth_items = make_synthetic_items(args.synthetic_dir)
        if not synth_items:
            raise SystemExit(f"No synthetic images found under {args.synthetic_dir}")
        synth_ds = ROIDataset(synth_items, image_size=112, value_range="unit")
        combined_train = ConcatDataset([train_ds, synth_ds])
        print(f"train: {len(train_ds)} real + {len(synth_ds)} synthetic = {len(combined_train)}")
    else:
        combined_train = train_ds
        print(f"train: {len(train_ds)} real (actual-data only)")

    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_classifier(num_classes=len(CLASS_NAMES)).to(device)
    print(f"params: {count_params(model):,} total, {count_trainable_params(model):,} trainable")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.999)
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        print(f"epoch {epoch+1}/{args.epochs}  loss={running_loss/total:.4f}  train_acc={correct/total:.2%}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, labels = predict(model, test_loader, device)
    table = classification_table(labels, preds, CLASS_NAMES)
    print("\n" + table)
    (out_dir / "metrics.txt").write_text(table, encoding="utf-8")

    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title=f"CNN-{args.mode.upper()} confusion matrix",
        out_path=str(out_dir / "confusion_matrix.png"),
    )

    if synth_ds is not None:
        real_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=False)
        real_feats, real_labels = extract_features(model, real_loader, device)
        synth_feats, synth_labels = extract_features(model, synth_loader, device)

        all_feats = np.concatenate([real_feats, synth_feats])
        all_labels = np.concatenate([real_labels, synth_labels])
        sources = ["real"] * len(real_labels) + ["synthetic"] * len(synth_labels)
        plot_pca(all_feats, all_labels, sources, CLASS_NAMES, out_path=str(out_dir / "pca.png"))

    torch.save(model.state_dict(), out_dir / "classifier.pt")
    print(f"\ndone. artifacts written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--mode", choices=["ad", "sa"], required=True)
    ap.add_argument("--synthetic-dir", default=None)
    ap.add_argument("--out-dir", default="runs/cnn")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    run(ap.parse_args())
