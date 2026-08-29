"""
Stage 2 -- multi-seed CNN-AD vs CNN-SA comparison.

A single train/eval run on 192 test images has ~0.5%/image quantization and
seed-to-seed variance, so a one-off "+1.5 points" could be noise. This driver
retrains the improved model from scratch for several seeds in BOTH modes
(ad = real only, sa = real + synthetic) and reports mean +/- std of accuracy and
COVID recall, so the augmentation effect can be stated as a distribution rather
than a single number.

    python -m stage2.multiseed --seeds 0 1 2 3 4 --synthetic-dir data/synthetic

Writes per-run artifacts under runs/stage2_multiseed/<mode>_seed<k>/ and an
aggregate summary to runs/stage2_multiseed/summary.json (+ a printed table).
"""
import argparse
import json
import statistics
from argparse import Namespace
from pathlib import Path

from stage2.train_stage2 import run


def _args_for(seed: int, mode: str, out_dir: str, base: argparse.Namespace) -> Namespace:
    return Namespace(
        manifest=base.manifest, mode=mode,
        synthetic_dir=base.synthetic_dir, out_dir=out_dir,
        epochs=base.epochs, batch_size=base.batch_size,
        lr=base.lr, backbone_lr=base.backbone_lr,
        unfreeze_blocks=base.unfreeze_blocks, no_head_bn=base.no_head_bn,
        workers=base.workers, device=base.device, cache=base.cache,
        cpu=base.cpu, seed=seed,
    )


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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--synthetic-dir", default="data/synthetic")
    ap.add_argument("--out-root", default="runs/stage2_multiseed")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=1)
    ap.add_argument("--no-head-bn", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu", action="store_true")
    base = ap.parse_args()

    out_root = Path(base.out_root)
    results = {"ad": [], "sa": []}
    for mode in ("ad", "sa"):
        for seed in base.seeds:
            out_dir = out_root / f"{mode}_seed{seed}"
            print(f"\n=== STAGE 2 multiseed: mode={mode} seed={seed} -> {out_dir} ===")
            summary = run(_args_for(seed, mode, str(out_dir), base))
            results[mode].append(summary)

    agg = {}
    for mode in ("ad", "sa"):
        accs = [r["accuracy"] for r in results[mode]]
        recs = [r["covid_recall"] for r in results[mode]]
        agg[mode] = {"accuracy": _stats(accs), "covid_recall": _stats(recs),
                     "n_runs": len(results[mode]), "seeds": base.seeds}

    lift_mean = agg["sa"]["accuracy"]["mean"] - agg["ad"]["accuracy"]["mean"]
    recall_lift = agg["sa"]["covid_recall"]["mean"] - agg["ad"]["covid_recall"]["mean"]
    agg["augmentation_effect"] = {
        "accuracy_lift_mean": lift_mean,
        "covid_recall_lift_mean": recall_lift,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    def pct(s):
        return f"{s['mean']*100:.2f}% +/- {s['std']*100:.2f}  (min {s['min']*100:.2f}, max {s['max']*100:.2f})"

    print("\n" + "=" * 72)
    print(f"STAGE 2 multi-seed summary  (seeds={base.seeds}, {len(base.seeds)} runs/mode)")
    print("=" * 72)
    print(f"CNN-AD  accuracy     : {pct(agg['ad']['accuracy'])}")
    print(f"CNN-SA  accuracy     : {pct(agg['sa']['accuracy'])}")
    print(f"  -> augmentation lift: {lift_mean*100:+.2f} points (mean)")
    print(f"CNN-AD  COVID recall : {pct(agg['ad']['covid_recall'])}")
    print(f"CNN-SA  COVID recall : {pct(agg['sa']['covid_recall'])}")
    print(f"  -> COVID-recall lift: {recall_lift*100:+.2f} points (mean)")
    print(f"\nsaved: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
