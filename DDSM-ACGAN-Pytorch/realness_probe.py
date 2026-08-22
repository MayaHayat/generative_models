"""
Real-vs-synthetic discriminability probe -- measures how separable a synthetic
pool is from the real training data, for a classifier of a given capacity.

This is the direct measurement of the variable ANALYSIS.md Sec 4.2 proposes as
the mechanism behind the unfrozen-classifier reversal, replacing the
epoch-count proxy that Sec 7 showed does not work.

The hypothesis under test: baseline-GAN synthetic is off-manifold and so is
trivially quarantined by a high-capacity classifier (it memorizes those labels
in their own feature region without perturbing the real decision boundary),
whereas improved-GAN synthetic sits on-manifold and cannot be fit without
deforming the boundary where real test points live. If that is right, the
baseline pool should be near-perfectly separable from real data and the
improved pools measurably less so.

Method: relabel real training ROIs as class 0 and the synthetic pool as class
1, balance the two sides, hold out a stratified validation split of BOTH (so
this measures generalizable separability, not memorization), train the same
classifier architecture at the same capacity as the experiment being explained,
and report held-out accuracy and ROC-AUC.

    python realness_probe.py --manifest data/manifest.csv \
        --synthetic-dir data/synthetic --out-dir runs/probe_realness_baseline \
        --unfreeze-blocks 2 --head-bn --seed 1

Run it once per pool with everything else identical; the comparison between
pools is the result, not any single pool's absolute number. Pass
--unfreeze-blocks/--head-bn matching the classifier condition whose behavior
you are explaining (uf=2 + head_bn for the Stage 2 grid).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ddsm_acgan.data import ROIDataset, make_synthetic_items, read_manifest
from ddsm_acgan.models import build_classifier, count_trainable_params
from train_classifier import set_seed

REAL_LABEL = 0
FAKE_LABEL = 1


def balanced_split(real_items, synth_items, val_frac: float, seed: int):
    """Balance the two sides to the same count, then hold out `val_frac` of
    each. Balancing matters because accuracy is only readable against a 50%
    chance line if the classes are equal-sized."""
    rng = random.Random(seed)
    real = [(p, REAL_LABEL) for p, _ in real_items]
    fake = [(p, FAKE_LABEL) for p, _ in synth_items]
    rng.shuffle(real)
    rng.shuffle(fake)

    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    n_val = max(1, round(n * val_frac))

    train = real[n_val:] + fake[n_val:]
    val = real[:n_val] + fake[:n_val]
    rng.shuffle(train)
    return train, val, n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, FAKE_LABEL].cpu().numpy())
        labels.append(lbls.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float((preds == labels).mean()),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
    }


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    real_items = read_manifest(args.manifest, "train")
    synth_items = make_synthetic_items(args.synthetic_dir)
    if not synth_items:
        raise SystemExit(f"No synthetic images found under {args.synthetic_dir}")

    train_items, val_items, n_per_side = balanced_split(
        real_items, synth_items, args.val_frac, args.seed
    )
    print(f"device: {device}  seed: {args.seed}  pool: {args.synthetic_dir}")
    print(f"balanced to {n_per_side} per side -> train {len(train_items)}, held-out val {len(val_items)}")

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"realness-{Path(args.synthetic_dir).name}",
                   config=vars(args), group=args.wandb_group)

    train_loader = DataLoader(ROIDataset(train_items, image_size=112, value_range="unit"),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(ROIDataset(val_items, image_size=112, value_range="unit"),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_classifier(num_classes=2, unfreeze_blocks=args.unfreeze_blocks,
                             head_bn=args.head_bn).to(device)
    print(f"{count_trainable_params(model):,} trainable params "
          f"(unfreeze_blocks={args.unfreeze_blocks}, head_bn={args.head_bn})")

    if args.unfreeze_blocks > 0:
        optimizer = torch.optim.Adam(
            model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr), betas=(0.9, 0.999)
        )
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
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  loss={running_loss/total:.4f}  "
              f"train_acc={correct/total:.2%}  val_acc={val['accuracy']:.2%}  val_auc={val['auc']:.4f}")
        if wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": running_loss / total,
                       "train_acc": correct / total, "val_accuracy": val["accuracy"],
                       "val_auc": val["auc"]}, step=epoch + 1)

    final = evaluate(model, val_loader, device)
    print(f"\nheld-out separability: accuracy={final['accuracy']:.2%}  AUC={final['auc']:.4f}  "
          f"(0.5 AUC = indistinguishable from real, 1.0 = trivially separable)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"synthetic_dir": args.synthetic_dir, "seed": args.seed,
              "unfreeze_blocks": args.unfreeze_blocks, "n_per_side": n_per_side, **final}
    (out_dir / "realness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if wandb:
        wandb.log({"final_val_accuracy": final["accuracy"], "final_val_auc": final["auc"]})
        wandb.finish()
    print(f"wrote {out_dir / 'realness.json'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", required=True, help="One pool per run; compare across runs.")
    ap.add_argument("--out-dir", default="runs/probe_realness")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Fewer than the classifier's 25 -- real/fake is a much easier task and "
                         "over-training only pushes every pool toward memorizing the training half.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of EACH side held out. Held out on both sides so the metric is "
                         "generalizable separability rather than memorization.")
    ap.add_argument("--unfreeze-blocks", type=int, default=2,
                    help="Match the classifier condition being explained (2 for the Stage 2 grid).")
    ap.add_argument("--head-bn", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="ddsm-acgan")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    run(ap.parse_args())