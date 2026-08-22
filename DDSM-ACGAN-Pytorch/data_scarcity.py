"""
Data-scarcity ablation: does GAN augmentation help more when the real
training set is artificially starved, matching the paper's own small-data
premise more closely than the full real dataset does? Subsamples the real
training set (stratified, seed-controlled) to each of --fractions, trains
CNN-AD and CNN-SA at each point (averaged over --seeds), and reports the
augmentation lift per fraction. Test set is never subsampled.

    python data_scarcity.py --fractions 0.1 0.25 0.5 1.0 --seeds 0 1 2 \
        --unfreeze-blocks 2 --head-bn --synthetic-dir data/synthetic \
        --out-root runs/scarcity_uf2
"""
import argparse
import json
import statistics
from pathlib import Path

from train_classifier import run as run_classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", required=True,
                     help="Full synthetic pool -- kept fixed across all fractions; only the real "
                          "training set shrinks, so the comparison isolates real-data scarcity.")
    ap.add_argument("--out-root", default="runs/scarcity")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--head-bn", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    class NS:
        pass

    points = []
    for frac in args.fractions:
        point = {"real_frac": frac, "ad": [], "sa": []}
        for mode in ("ad", "sa"):
            for seed in args.seeds:
                ns = NS()
                ns.manifest = args.manifest
                ns.mode = mode
                ns.synthetic_dir = args.synthetic_dir if mode == "sa" else None
                ns.out_dir = str(Path(args.out_root) / f"frac{frac}_{mode}_seed{seed}")
                ns.epochs = args.epochs
                ns.batch_size = args.batch_size
                ns.lr = args.lr
                ns.workers = 2
                ns.cpu = args.cpu
                ns.seed = seed
                ns.real_frac = frac
                ns.unfreeze_blocks = args.unfreeze_blocks
                ns.head_bn = args.head_bn
                ns.backbone_lr = args.backbone_lr
                ns.wandb = False
                ns.wandb_project = None
                ns.wandb_run_name = None
                ns.wandb_group = None
                ns.synthetic_only = False

                print(f"\n=== frac={frac} mode={mode} seed={seed} ===")
                result = run_classifier(ns)
                point[mode].append(result["accuracy"])
        points.append(point)

    print("\n\n=== Data-scarcity summary ===")
    print(f"{'frac':>6} {'n_seeds':>8} {'AD mean':>10} {'SA mean':>10} {'lift':>8}")
    summary = []
    for p in points:
        ad_mean = statistics.mean(p["ad"]) if p["ad"] else float("nan")
        sa_mean = statistics.mean(p["sa"]) if p["sa"] else float("nan")
        lift = sa_mean - ad_mean
        print(f"{p['real_frac']:>6.2f} {len(args.seeds):>8} {ad_mean:>10.4f} {sa_mean:>10.4f} {lift:>+8.4f}")
        summary.append({
            "real_frac": p["real_frac"],
            "ad_accuracies": p["ad"], "ad_mean": ad_mean,
            "sa_accuracies": p["sa"], "sa_mean": sa_mean,
            "lift": lift,
        })

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "scarcity_summary.json", "w") as f:
        json.dump({"args": vars(args), "points": summary}, f, indent=2)
    print(f"\nwrote {out_root / 'scarcity_summary.json'}")


if __name__ == "__main__":
    main()
