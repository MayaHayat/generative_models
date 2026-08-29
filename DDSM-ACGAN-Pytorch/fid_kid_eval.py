"""
FID / KID comparison between synthetic pools and the real training data.

    python fid_kid_eval.py --manifest data/manifest.csv \
        --pool baseline:data/synthetic \
        --pool improved300:data/synthetic_improved \
        --pool improved600:data/synthetic_improved_epoch600 \
        --wandb

Four things this handles that a naive folder-glob version does not, all of
them load-bearing for the number being meaningful:

1. The real set comes from the manifest's *train* split, not an rglob of the
   ROI directory. The ROI folder holds calcification patches the mass-only
   GANs never saw plus the held-out test split; including either makes the
   comparison measure dataset composition instead of generation quality.

2. Real ROIs are loaded through ddsm_acgan.data._load_as_uint8_grayscale, not
   PIL's .convert("RGB"). CBIS-DDSM patches are 16-bit ('I;16') and convert()
   truncates them to near-white -- see FINDINGS.md Sec 2.2. Synthetic PNGs are
   already 8-bit, so the naive path corrupts one side of the comparison only.

3. Both sides pass through 112x112 before the 299x299 Inception resize. The
   generator emits 112x112 while real ROIs are 384x384 native; resizing real
   images straight to 299 leaves an upsampling signature that dominates FID.
   Routing both through 112 compares what the GAN was actually asked to model.

4. Sample count is clamped to the smallest pool rather than raising. Note that
   FID's 2048-d covariance is badly biased at n=600 -- treat the absolute
   values as meaningless and only compare pools evaluated at identical n. KID
   is unbiased at small n and is the number to trust here.
"""
import argparse
import copy
import json
import os
import random
from pathlib import Path

import certifi

# macOS python.org builds ship without a usable CA bundle, so torchmetrics'
# download of the pretrained InceptionV3 weights fails with
# "SSL: CERTIFICATE_VERIFY_FAILED". Point urllib at certifi's bundle *before*
# any weight download happens -- mirrors CovidGAN-Pytorch/evaluate_fid.py.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ddsm_acgan.data import _load_as_uint8_grayscale, make_synthetic_items, read_manifest

GAN_SIZE = 112
INCEPTION_SIZE = 299


def load_tensor(path: Path) -> torch.Tensor:
    """uint8 CHW at 299x299, routed through the GAN's native 112x112 so real
    and synthetic images carry the same resampling history."""
    img = _load_as_uint8_grayscale(path).convert("RGB")
    if img.size != (GAN_SIZE, GAN_SIZE):
        img = img.resize((GAN_SIZE, GAN_SIZE), Image.LANCZOS)
    img = img.resize((INCEPTION_SIZE, INCEPTION_SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def batches(paths, batch_size):
    for i in range(0, len(paths), batch_size):
        yield torch.stack([load_tensor(p) for p in paths[i:i + batch_size]])


def build_real_metrics(real_paths, device, batch_size, kid_subset_size, kid_subsets, n_fake):
    """Extract the real set's Inception features ONCE. Sweeping many pools
    (e.g. a checkpoint-by-checkpoint training curve) otherwise re-runs Inception
    over the same real images for every pool, which dominates the runtime."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subsets=kid_subsets,
                                  subset_size=min(kid_subset_size, len(real_paths), n_fake),
                                  normalize=False).to(device)
    for batch in batches(real_paths, batch_size):
        batch = batch.to(device)
        fid.update(batch, real=True)
        kid.update(batch, real=True)
    return fid, kid


def evaluate(real_metrics, fake_paths, device, batch_size):
    """Deep-copy the real-side state so each pool is scored against an identical
    real reference without recomputing it."""
    fid, kid = (copy.deepcopy(m) for m in real_metrics)
    for batch in batches(fake_paths, batch_size):
        batch = batch.to(device)
        fid.update(batch, real=False)
        kid.update(batch, real=False)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(kid_mean), "kid_std": float(kid_std)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv",
                    help="Real images come from this manifest's train split (what the GAN saw).")
    ap.add_argument("--pool", action="append", required=True, metavar="NAME:DIR",
                    help="Repeatable, e.g. --pool baseline:data/synthetic")
    ap.add_argument("--num-images", type=int, default=0,
                    help="0 = use the largest count every pool and the real set can supply.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--kid-subset-size", type=int, default=50)
    ap.add_argument("--kid-subsets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/fid_kid")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="ddsm-acgan")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default="fid-kid-pools")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    pools = {}
    for spec in args.pool:
        if ":" not in spec:
            raise SystemExit(f"--pool expects NAME:DIR, got {spec!r}")
        name, directory = spec.split(":", 1)
        paths = [p for p, _ in make_synthetic_items(directory)]
        if not paths:
            raise SystemExit(f"no synthetic images under {directory}")
        pools[name] = paths

    real_paths = [p for p, _ in read_manifest(args.manifest, "train")]
    n = min([len(real_paths)] + [len(v) for v in pools.values()])
    if args.num_images:
        n = min(n, args.num_images)
    print(f"real train images: {len(real_paths)}")
    for name, paths in pools.items():
        print(f"  pool {name}: {len(paths)}")
    print(f"evaluating every pool at n={n} (equal n is required for the comparison to mean anything)")

    rng = random.Random(args.seed)
    real_sample = rng.sample(real_paths, n)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("extracting real-set Inception features (once, shared across pools)...")
    real_metrics = build_real_metrics(real_sample, device, args.batch_size,
                                      args.kid_subset_size, args.kid_subsets, n)

    results = {}
    for name, paths in pools.items():
        scores = evaluate(real_metrics, random.Random(args.seed).sample(paths, n),
                          device, args.batch_size)
        results[name] = scores
        print(f"{name:14} FID={scores['fid']:8.2f}   KID={scores['kid_mean']:.5f} "
              f"+/- {scores['kid_std']:.5f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"n": n, "manifest": args.manifest, "seed": args.seed, "results": results}
    (out_dir / "fid_kid.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'fid_kid.json'}")

    if args.wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         name=args.wandb_run_name, job_type="gan-evaluation",
                         config={"n": n, "seed": args.seed, "pools": {k: v for k, v in
                                 (s.split(":", 1) for s in args.pool)}})
        wandb.log({f"{name}/{k}": v for name, s in results.items() for k, v in s.items()})
        run.finish()


if __name__ == "__main__":
    main()
