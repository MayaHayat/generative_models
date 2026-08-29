"""
Stage 2 -- data-scarcity curve: does GAN augmentation help MORE when real data
is scarce?

Motivation. Our full-data augmentation lift is small (~+1 pt) because the real
baseline is already ~92.5% -- near ceiling on the clean modern dataset, so there
is little gap for synthetic data to fill. The paper's big +10-pt lift came from a
DATA-STARVED 85% baseline. This experiment recreates that regime: it subsamples
the REAL training set to a fraction f (stratified by class), keeps the FULL
synthetic pool fixed, and compares CNN-AD (real subset only) vs CNN-SA (real
subset + all synthetic) across f. The expected result -- the augmentation lift
grows as f shrinks -- both explains the paper and yields a much larger, honestly
motivated improvement number for the low-data points.

Everything uses the Stage 2 improved model (top VGG16 block fine-tuned + BN head).
Real and synthetic images are decoded/cached ONCE and re-used via Subset, so the
whole sweep is cheap.

    python -m stage2.data_scarcity --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2 \
        --synthetic-dir data/synthetic

Writes runs/stage2_scarcity/summary.json and prints a fraction-vs-lift table.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from covidgan.data import CLASS_NAMES, CXRDataset, make_synthetic_items, read_manifest
from covidgan.models import build_classifier, pick_device

from stage2.model import build_stage2_classifier
from stage2.train_stage2 import seed_everything, summarize


def stratified_indices(labels, frac, seed):
    """Pick a class-balanced fraction of indices into the real train set. Same
    (frac, seed) -> same subset, so CNN-AD and CNN-SA see identical real data."""
    by_label = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_label[lbl].append(i)
    rng = random.Random(seed)
    chosen = []
    for lbl, idxs in by_label.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * frac))
        chosen.extend(idxs[:k])
    return chosen


def train_eval(train_ds, test_loader, device, args, seed):
    """Fresh model, train on train_ds, evaluate on the fixed test set.

    --frozen uses the paper's EXACT detector (covidgan.models.build_classifier:
    frozen VGG16 base + ~33K-param head, no BatchNorm) so the sweep answers "does
    augmentation help under the paper's own design"; otherwise the Stage 2 improved
    classifier (top VGG block fine-tuned + BN head) is used."""
    seed_everything(seed)
    if args.frozen:
        model = build_classifier(num_classes=len(CLASS_NAMES)).to(device)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=args.lr, betas=(0.9, 0.999))
    else:
        model = build_stage2_classifier(num_classes=len(CLASS_NAMES),
                                        unfreeze_blocks=args.unfreeze_blocks).to(device)
        optimizer = torch.optim.Adam(model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
                                     betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    for _ in range(args.epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for imgs, l in test_loader:
            preds.append(model(imgs.to(device)).argmax(1).cpu().numpy())
            labs.append(l.numpy())
    return summarize(np.concatenate(labs), np.concatenate(preds))


def _stats(values):
    return {"mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="runs/stage2_scarcity")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=1)
    ap.add_argument("--frozen", action="store_true",
                    help="Use the paper's exact frozen-VGG16 detector (build_classifier) "
                         "instead of the Stage 2 fine-tuned model.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}  (STAGE 2 data-scarcity sweep)")

    # Decode + cache real train, synthetic pool, and test set ONCE.
    train_items = read_manifest(args.manifest, "train")
    test_items = read_manifest(args.manifest, "test")
    real_ds = CXRDataset(train_items, image_size=112, value_range="unit", cache=True)
    synth_items = make_synthetic_items(args.synthetic_dir)
    synth_ds = CXRDataset(synth_items, image_size=112, value_range="unit", cache=True)
    test_ds = CXRDataset(test_items, image_size=112, value_range="unit", cache=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    real_labels = [lbl for _, lbl in train_items]
    print(f"real train: {len(real_ds)}  synthetic (fixed): {len(synth_ds)}  test: {len(test_ds)}")

    results = []
    for frac in args.fractions:
        ad_acc, ad_rec, sa_acc, sa_rec, n_real = [], [], [], [], None
        for seed in args.seeds:
            idx = stratified_indices(real_labels, frac, seed)
            n_real = len(idx)
            real_sub = Subset(real_ds, idx)
            # CNN-AD: real subset only
            ad = train_eval(real_sub, test_loader, device, args, seed)
            # CNN-SA: real subset + full synthetic pool
            sa = train_eval(ConcatDataset([real_sub, synth_ds]), test_loader, device, args, seed)
            ad_acc.append(ad["accuracy"]); ad_rec.append(ad["covid_recall"])
            sa_acc.append(sa["accuracy"]); sa_rec.append(sa["covid_recall"])
            print(f"  frac={frac:<5} seed={seed}  n_real={n_real:<4} "
                  f"AD={ad['accuracy']*100:.2f}%  SA={sa['accuracy']*100:.2f}%  "
                  f"lift={ (sa['accuracy']-ad['accuracy'])*100:+.2f}")
        results.append({
            "fraction": frac, "n_real": n_real,
            "ad_accuracy": _stats(ad_acc), "sa_accuracy": _stats(sa_acc),
            "ad_covid_recall": _stats(ad_rec), "sa_covid_recall": _stats(sa_rec),
            "accuracy_lift": statistics.mean(sa_acc) - statistics.mean(ad_acc),
            "covid_recall_lift": statistics.mean(sa_rec) - statistics.mean(ad_rec),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": args.seeds, "frozen": args.frozen,
               "unfreeze_blocks": (None if args.frozen else args.unfreeze_blocks),
               "synthetic_dir": args.synthetic_dir, "points": results}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"STAGE 2 data-scarcity curve  (improved model, {len(args.seeds)} seeds/point)")
    print("=" * 78)
    print(f"{'real frac':>10}{'n_real':>8}{'CNN-AD':>12}{'CNN-SA':>12}{'acc lift':>11}{'recall lift':>13}")
    print("-" * 78)
    for r in results:
        print(f"{r['fraction']:>10}{r['n_real']:>8}"
              f"{r['ad_accuracy']['mean']*100:>11.2f}%{r['sa_accuracy']['mean']*100:>11.2f}%"
              f"{r['accuracy_lift']*100:>+10.2f}{r['covid_recall_lift']*100:>+12.2f}")
    print("-" * 78)
    print("Expectation: augmentation lift GROWS as the real fraction shrinks -- i.e. the")
    print("paper's regime (scarce real data) is where the GAN augmentation pays off most.")
    print(f"\nsaved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
