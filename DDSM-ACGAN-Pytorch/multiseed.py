"""
Run CNN-AD and CNN-SA across multiple seeds and report mean +/- std, so a
Stage 2 accuracy comparison is backed statistically rather than by single
noisy runs (classifier training here has no fixed seed by default, and
run-to-run swings of several accuracy points have been observed throughout
this project -- see FINDINGS.md Sec 3.2).

    python multiseed.py --seeds 0 1 2 3 4 --unfreeze-blocks 2 --head-bn \
        --synthetic-dir data/synthetic --out-root runs/multiseed_uf2
"""
import argparse
import json
import statistics
from pathlib import Path

from train_classifier import run as run_classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", default=None, help="Required to also run the SA arm.")
    ap.add_argument("--out-root", default="runs/multiseed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--head-bn", action="store_true")
    ap.add_argument("--real-frac", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    class NS:
        pass

    results = {"ad": [], "sa": []}
    modes = ["ad", "sa"] if args.synthetic_dir else ["ad"]
    if not args.synthetic_dir:
        print("no --synthetic-dir given: running AD only")

    for mode in modes:
        for seed in args.seeds:
            ns = NS()
            ns.manifest = args.manifest
            ns.mode = mode
            ns.synthetic_dir = args.synthetic_dir
            ns.out_dir = str(Path(args.out_root) / f"{mode}_seed{seed}")
            ns.epochs = args.epochs
            ns.batch_size = args.batch_size
            ns.lr = args.lr
            ns.workers = 2
            ns.cpu = args.cpu
            ns.seed = seed
            ns.real_frac = args.real_frac
            ns.unfreeze_blocks = args.unfreeze_blocks
            ns.head_bn = args.head_bn
            ns.backbone_lr = args.backbone_lr
            ns.wandb = False
            ns.wandb_project = None
            ns.wandb_run_name = None
            ns.wandb_group = None

            print(f"\n=== mode={mode} seed={seed} ===")
            result = run_classifier(ns)
            results[mode].append(result["accuracy"])

    summary = {}
    for mode, accs in results.items():
        if not accs:
            continue
        summary[mode] = {
            "accuracies": accs,
            "mean": statistics.mean(accs),
            "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        }
        print(f"\n{mode.upper()}: {[f'{a:.4f}' for a in accs]}  "
              f"mean={summary[mode]['mean']:.4f}  std={summary[mode]['std']:.4f}")

    if "ad" in summary and "sa" in summary:
        lift = summary["sa"]["mean"] - summary["ad"]["mean"]
        print(f"\nAugmentation lift: {lift:+.4f} ({lift*100:+.2f} pts)")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "multiseed_summary.json", "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)
    print(f"\nwrote {out_root / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
