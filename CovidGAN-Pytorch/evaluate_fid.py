"""
Evaluate CovidGAN's synthetic chest X-rays with the Frechet Inception Distance
(FID) -- the generative-model quality metric studied in class.

WHAT FID IS
-----------
FID compares two *distributions* of images -- here, the real CXRs vs. the
GAN-synthesized CXRs -- rather than pairing images one-to-one. Every image is
pushed through a pretrained InceptionV3 network and summarized by the 2048-d
activations of its final pooling layer. Each set of activations is modelled as a
multivariate Gaussian N(mu, Sigma), and FID is the Frechet (a.k.a. 2-Wasserstein)
distance between the real Gaussian (mu_r, Sigma_r) and the fake one (mu_g, Sigma_g):

    FID = ||mu_r - mu_g||^2 + Tr( Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2} )

Lower is better: FID = 0 means the two feature distributions are identical, and
it rises as the synthetic images drift away from the real ones in either
appearance (mean) or variety (covariance). Because it looks at deep perceptual
features and at the whole distribution, FID captures both realism *and* mode
collapse, which is exactly why it is the standard yardstick for GANs -- and the
right metric for CovidGAN, whose only job is to produce realistic CXRs.

WHY PER-CLASS
-------------
CovidGAN is class-conditional (an AC-GAN): it generates covid and normal CXRs
from a class label. So besides the overall FID (all real vs. all synthetic) we
also report FID within each class -- real covid vs. synthetic covid, and real
normal vs. synthetic normal -- to check the generator is faithful to *both*
conditional distributions, not just the pooled one.

HOW TO RUN
----------
    # real images pulled from the manifest's test split, synthetic from the pool
    python evaluate_fid.py --real-manifest data/manifest.csv --real-split test \\
        --synthetic-dir data/synthetic

    # or point --real-dir at a folder laid out like the synthetic pool
    python evaluate_fid.py --real-dir data/real --synthetic-dir data/synthetic

CAVEAT ON SAMPLE COUNT
----------------------
FID estimates a 2048x2048 covariance, so it is biased (usually upward) and noisy
on small sets. Treat a per-class FID computed on only a few dozen images as a
smoke test, not a trustworthy score -- aim for a couple thousand images per set
(the InceptionV3 feature dimension is 2048) for a stable number.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import certifi

# macOS python.org builds ship without a usable CA bundle, so torchmetrics'
# download of the pretrained InceptionV3 weights fails with
# "SSL: CERTIFICATE_VERIFY_FAILED". Point urllib at certifi's bundle *before*
# any weight download happens -- mirrors the notebook's environment cell.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

from covidgan.data import (
    CLASS_NAMES,
    COVID_LABEL,
    NORMAL_LABEL,
    make_synthetic_items,
    read_manifest,
)
from covidgan.models import pick_device

# InceptionV3's native input; torchmetrics resizes for us, but decoding straight
# to this size keeps memory down and matches the standard FID preprocessing.
INCEPTION_SIZE = 299


def load_image_uint8(path: Path) -> torch.Tensor:
    """Load one image as an RGB uint8 CHW tensor in [0, 255].

    torchmetrics' FrechetInceptionDistance expects uint8 tensors and does its own
    normalization + resize to InceptionV3's input internally, so we must hand it
    raw [0, 255] pixels (NOT the [0,1]/[-1,1] floats the GAN/classifier use)."""
    img = Image.open(path).convert("RGB").resize((INCEPTION_SIZE, INCEPTION_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)  # HWC (np.array copies -> writable tensor)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8


def fid_device(device: torch.device) -> torch.device:
    """torchmetrics' FID accumulates its 2048x2048 covariance in float64, which
    Apple's MPS backend does not support ("MPS framework doesn't support
    float64"). CUDA and CPU both handle float64, so only MPS needs redirecting to
    CPU for the metric itself."""
    if device.type == "mps":
        return torch.device("cpu")
    return device


def compute_fid(real_paths: List[Path], fake_paths: List[Path],
                device: torch.device, batch_size: int) -> float:
    """Standard FID between two image sets using InceptionV3 features."""
    metric = FrechetInceptionDistance(normalize=False).to(device)
    metric.set_dtype(torch.float64)  # covariance accumulation is more stable in f64

    def feed(paths: List[Path], real: bool):
        batch: List[torch.Tensor] = []
        for p in paths:
            batch.append(load_image_uint8(p))
            if len(batch) == batch_size:
                metric.update(torch.stack(batch).to(device), real=real)
                batch = []
        if batch:
            metric.update(torch.stack(batch).to(device), real=real)

    feed(real_paths, real=True)
    feed(fake_paths, real=False)
    return float(metric.compute().item())


def split_by_class(items: List[Tuple[Path, int]]) -> Tuple[List[Path], List[Path]]:
    covid = [p for p, lbl in items if lbl == COVID_LABEL]
    normal = [p for p, lbl in items if lbl == NORMAL_LABEL]
    return covid, normal


def collect_real(args) -> List[Tuple[Path, int]]:
    if args.real_dir:
        items = make_synthetic_items(args.real_dir)  # same covid/ normal/ layout
        if not items:
            raise SystemExit(
                f"No real images found under {args.real_dir} "
                f"(expected {args.real_dir}/covid/*.png and {args.real_dir}/normal/*.png)"
            )
        return items
    items = read_manifest(args.real_manifest, args.real_split)
    if not items:
        raise SystemExit(
            f"No real images in {args.real_manifest} for split '{args.real_split}'"
        )
    return items


def render_table(rows: List[Tuple[str, int, int, float]]) -> str:
    header = f"{'set':<12}{'n_real':>8}{'n_synth':>9}{'FID':>12}"
    lines = [header, "-" * len(header)]
    for name, n_real, n_synth, fid in rows:
        lines.append(f"{name:<12}{n_real:>8}{n_synth:>9}{fid:>12.3f}")
    return "\n".join(lines)


def run(args):
    device = fid_device(pick_device("cpu" if args.cpu else args.device))
    print(f"device: {device}"
          + ("  (FID's float64 covariance is unsupported on MPS, using CPU)"
             if device.type == "cpu" and not args.cpu else ""))

    real_items = collect_real(args)
    synth_items = make_synthetic_items(args.synthetic_dir)
    if not synth_items:
        raise SystemExit(
            f"No synthetic images found under {args.synthetic_dir} "
            f"(expected {args.synthetic_dir}/covid/*.png and "
            f"{args.synthetic_dir}/normal/*.png -- run generate_synthetic.py first)"
        )

    real_covid, real_normal = split_by_class(real_items)
    synth_covid, synth_normal = split_by_class(synth_items)

    real_all = [p for p, _ in real_items]
    synth_all = [p for p, _ in synth_items]

    print(f"real:      {len(real_all)}  ({len(real_covid)} covid, {len(real_normal)} normal)")
    print(f"synthetic: {len(synth_all)}  ({len(synth_covid)} covid, {len(synth_normal)} normal)")
    print("computing FID (this loads InceptionV3 and may download its weights once)...\n")

    rows: List[Tuple[str, int, int, float]] = []
    rows.append(("overall", len(real_all), len(synth_all),
                 compute_fid(real_all, synth_all, device, args.batch_size)))
    if real_covid and synth_covid:
        rows.append((CLASS_NAMES[COVID_LABEL], len(real_covid), len(synth_covid),
                     compute_fid(real_covid, synth_covid, device, args.batch_size)))
    if real_normal and synth_normal:
        rows.append((CLASS_NAMES[NORMAL_LABEL], len(real_normal), len(synth_normal),
                     compute_fid(real_normal, synth_normal, device, args.batch_size)))

    print(render_table(rows))

    min_n = min(min(n_r, n_s) for _, n_r, n_s, _ in rows)
    if min_n < 2048:
        print(
            f"\nnote: smallest set has {min_n} images (< 2048, InceptionV3's feature "
            "dim). FID is biased/noisy on small sets -- treat this as indicative, "
            "not a definitive score."
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    real_src = ap.add_argument_group("real image source (use one)")
    real_src.add_argument("--real-manifest", default="data/manifest.csv",
                          help="manifest.csv to pull real images from (used unless --real-dir is given).")
    real_src.add_argument("--real-split", default="test", choices=["train", "test"],
                          help="which manifest split to use as the real set (default: test).")
    real_src.add_argument("--real-dir", default=None,
                          help="alternative: a folder with covid/ and normal/ subdirs of real PNGs "
                               "(overrides --real-manifest).")
    ap.add_argument("--synthetic-dir", default="data/synthetic",
                    help="synthetic pool laid out as covid/ and normal/ subdirs (from generate_synthetic.py).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    help="auto (cuda > mps > cpu), or force cuda / mps / cpu.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (shorthand for --device cpu).")
    run(ap.parse_args())
