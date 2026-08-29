"""
Stage 2 -- train & evaluate the IMPROVED COVID-19 detector.

This is the Stage 2 counterpart to the Stage 1 `train_classifier.py`. It shares
the exact same data pipeline, test split, and metrics (so the comparison is
apples-to-apples) but swaps in the improved model from `stage2/model.py`
(top VGG16 block fine-tuned + BatchNorm head) trained with discriminative
learning rates.

Two modes reproduce the paper's CNN-AD / CNN-SA comparison, now under the
improved architecture:

  --mode ad   improved detector on real data only.
  --mode sa   improved detector on real + the GAN synthetic pool.

Run both to fill the Stage 2 results table (see stage2/compare_results.py):

    python -m stage2.train_stage2 --mode ad --out-dir runs/stage2_cnn_ad
    python -m stage2.train_stage2 --mode sa --synthetic-dir data/synthetic \
        --out-dir runs/stage2_cnn_sa

BatchNorm in the head needs >1 sample per batch, so the train loader uses
drop_last=True.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from covidgan.data import CLASS_NAMES, CXRDataset, make_synthetic_items, read_manifest
from covidgan.metrics import classification_table, plot_confusion_matrix
from covidgan.models import count_params, count_trainable_params, pick_device

from stage2.model import build_stage2_classifier

COVID_IDX = CLASS_NAMES.index("covid")


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/torch so a multi-seed run is reproducible. Full MPS/CUDA
    determinism isn't guaranteed, but this pins the data shuffling and weight
    init, which is what the seed-to-seed spread mostly comes from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Compact metrics dict for the comparison table / multi-seed aggregation."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    total = int(labels.size)
    correct = int((labels == preds).sum())
    covid_mask = labels == COVID_IDX
    covid_support = int(covid_mask.sum())
    covid_tp = int((preds[covid_mask] == COVID_IDX).sum())
    return {
        "accuracy": correct / total,
        "n_correct": correct,
        "n_total": total,
        "covid_recall": covid_tp / covid_support if covid_support else 0.0,
        "covid_caught": covid_tp,
        "covid_support": covid_support,
    }


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
    seed_everything(args.seed)
    device = pick_device("cpu" if args.cpu else args.device)
    print(f"device: {device}  mode: {args.mode}  seed: {args.seed}  (STAGE 2 improved model)")

    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    train_ds = CXRDataset(train_items, image_size=112, value_range="unit", cache=args.cache)
    test_ds = CXRDataset(test_items, image_size=112, value_range="unit", cache=args.cache)

    if args.mode == "sa":
        if not args.synthetic_dir:
            raise SystemExit("--mode sa requires --synthetic-dir (see generate_synthetic.py)")
        synth_items = make_synthetic_items(args.synthetic_dir)
        if not synth_items:
            raise SystemExit(f"No synthetic images found under {args.synthetic_dir}")
        synth_ds = CXRDataset(synth_items, image_size=112, value_range="unit", cache=args.cache)
        combined_train = ConcatDataset([train_ds, synth_ds])
        print(f"train: {len(train_ds)} real + {len(synth_ds)} synthetic = {len(combined_train)}")
    else:
        combined_train = train_ds
        print(f"train: {len(train_ds)} real (actual-data only)")

    workers = 0 if args.cache else args.workers
    # drop_last: the head's BatchNorm1d needs >1 sample in a training batch.
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    model = build_stage2_classifier(
        num_classes=len(CLASS_NAMES), unfreeze_blocks=args.unfreeze_blocks,
        head_bn=not args.no_head_bn,
    ).to(device)
    print(f"params: {count_params(model):,} total, {count_trainable_params(model):,} trainable "
          f"(unfreeze_blocks={args.unfreeze_blocks}, head_bn={not args.no_head_bn})")

    # Discriminative LRs: fresh head at --lr, fine-tuned backbone at --backbone-lr.
    optimizer = torch.optim.Adam(
        model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
        betas=(0.9, 0.999),
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

    summary = summarize(labels, preds)
    summary.update(mode=args.mode, seed=args.seed, unfreeze_blocks=args.unfreeze_blocks,
                   head_bn=not args.no_head_bn)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title=f"Stage2 CNN-{args.mode.upper()} confusion matrix",
        out_path=str(out_dir / "confusion_matrix.png"),
    )

    torch.save(model.state_dict(), out_dir / "classifier.pt")
    print(f"\ndone. artifacts written to {out_dir}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--mode", choices=["ad", "sa"], required=True)
    ap.add_argument("--synthetic-dir", default=None)
    ap.add_argument("--out-dir", default="runs/stage2_cnn")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Fine-tuning overfits faster than head-only training, so the "
                         "default is lower than Stage 1's 25.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="Head (fresh layers) learning rate.")
    ap.add_argument("--backbone-lr", type=float, default=1e-5,
                    help="Learning rate for the unfrozen (pretrained) VGG16 block. Much smaller "
                         "than --lr so ImageNet filters are only gently adapted.")
    ap.add_argument("--unfreeze-blocks", type=int, default=1,
                    help="How many top VGG16 conv blocks to fine-tune (0 = Stage 1 behaviour, "
                         "1 = top block, up to 5). Default 1.")
    ap.add_argument("--no-head-bn", action="store_true",
                    help="Ablation: drop the BatchNorm added to the head.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (for multi-seed runs).")
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader worker processes (ignored when --cache is on).")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda > mps > cpu), or force cuda / mps / cpu.")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                    help="Preload+resize all images into RAM once (default on).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (shorthand for --device cpu).")
    run(ap.parse_args())
