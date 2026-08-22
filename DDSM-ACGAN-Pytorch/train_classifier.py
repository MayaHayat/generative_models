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

Stage 2: pass --unfreeze-blocks N (1-5) to fine-tune the top N VGG16 conv
blocks instead of the paper's fully-frozen backbone, and --head-bn to add
BatchNorm to the head -- see build_classifier()/Classifier in models.py for
the rationale. --real-frac subsamples the real training set (stratified,
--seed-controlled) for data-scarcity ablations; --seed also controls model
init and data shuffling, for multi-seed runs.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader

from ddsm_acgan.data import CLASS_NAMES, ROIDataset, make_synthetic_items, read_manifest
from ddsm_acgan.metrics import classification_table, plot_confusion_matrix, plot_pca
from ddsm_acgan.models import build_classifier, count_params, count_trainable_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample_stratified(items, frac: float, seed: int):
    """Keep `frac` of items per class label, shuffled deterministically by seed."""
    rng = random.Random(seed)
    by_label = {}
    for path, label in items:
        by_label.setdefault(label, []).append((path, label))
    kept = []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = max(1, round(len(group) * frac))
        kept += group[:n]
    return kept


@torch.no_grad()
def extract_features(model: nn.Module, loader, device):
    """Penultimate-layer (64-d) features: everything in the head up to and
    including its final activation, excluding Dropout and the last Linear --
    works whether or not head_bn is enabled since it just drops the head's
    last layer."""
    model.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        f = model.head[:-1](model.backbone(imgs))
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
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}  mode: {args.mode}  seed: {args.seed}")

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or f"cnn-{args.mode}",
                   config=vars(args), group=args.wandb_group)

    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    if args.real_frac < 1.0:
        before = len(train_items)
        train_items = subsample_stratified(train_items, args.real_frac, args.seed)
        print(f"real_frac={args.real_frac}: subsampled train {before} -> {len(train_items)}")
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

    model = build_classifier(num_classes=len(CLASS_NAMES), unfreeze_blocks=args.unfreeze_blocks,
                              head_bn=args.head_bn).to(device)
    print(f"params: {count_params(model):,} total, {count_trainable_params(model):,} trainable "
          f"(unfreeze_blocks={args.unfreeze_blocks}, head_bn={args.head_bn})")

    if args.unfreeze_blocks > 0:
        optimizer = torch.optim.Adam(model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
                                      betas=(0.9, 0.999))
        print(f"  differential lr: backbone={args.backbone_lr}  head={args.lr}")
    else:
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
        train_loss, train_acc = running_loss / total, correct / total
        print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.2%}")
        if wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}, step=epoch + 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, labels = predict(model, test_loader, device)
    table = classification_table(labels, preds, CLASS_NAMES)
    print("\n" + table)
    (out_dir / "metrics.txt").write_text(table, encoding="utf-8")

    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title=f"CNN-{args.mode.upper()} confusion matrix",
        out_path=str(cm_path),
    )

    if wandb:
        accuracy = float((labels == preds).mean())
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
        )
        test_metrics = {"test_accuracy": accuracy, "confusion_matrix": wandb.Image(str(cm_path))}
        for i, name in enumerate(CLASS_NAMES):
            test_metrics[f"test_precision_{name}"] = precision[i]
            test_metrics[f"test_recall_{name}"] = recall[i]
            test_metrics[f"test_f1_{name}"] = f1[i]
        wandb.log(test_metrics)

    if synth_ds is not None:
        real_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=False)
        real_feats, real_labels = extract_features(model, real_loader, device)
        synth_feats, synth_labels = extract_features(model, synth_loader, device)

        all_feats = np.concatenate([real_feats, synth_feats])
        all_labels = np.concatenate([real_labels, synth_labels])
        sources = ["real"] * len(real_labels) + ["synthetic"] * len(synth_labels)
        pca_path = out_dir / "pca.png"
        plot_pca(all_feats, all_labels, sources, CLASS_NAMES, out_path=str(pca_path))
        if wandb:
            wandb.log({"pca": wandb.Image(str(pca_path))})

    torch.save(model.state_dict(), out_dir / "classifier.pt")
    print(f"\ndone. artifacts written to {out_dir}")
    if wandb:
        wandb.finish()
    return {"accuracy": float((labels == preds).mean())}


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
    ap.add_argument("--seed", type=int, default=0, help="Controls model init and data shuffling.")
    ap.add_argument("--real-frac", type=float, default=1.0,
                     help="Stratified-subsample the real training set to this fraction (data-scarcity "
                          "ablation). Test set is never subsampled. Default 1.0 = use all real data.")
    ap.add_argument("--unfreeze-blocks", type=int, default=0,
                     help="Fine-tune the top N VGG16 conv blocks (0-5) instead of the paper's fully "
                          "frozen backbone (0). See Classifier in models.py for rationale.")
    ap.add_argument("--head-bn", action="store_true",
                     help="Add BatchNorm1d to the classifier head (paired with --unfreeze-blocks to "
                          "stabilize the larger trainable parameter set).")
    ap.add_argument("--backbone-lr", type=float, default=1e-5,
                     help="LR for unfrozen backbone blocks when --unfreeze-blocks > 0 (kept much lower "
                          "than --lr's head rate to avoid destroying pretrained features too fast).")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    ap.add_argument("--wandb-project", default="ddsm-acgan")
    ap.add_argument("--wandb-run-name", default=None, help="Defaults to 'cnn-<mode>'.")
    ap.add_argument("--wandb-group", default=None,
                     help="Group AD/SA runs together in the W&B UI, e.g. --wandb-group exp1.")
    run(ap.parse_args())
