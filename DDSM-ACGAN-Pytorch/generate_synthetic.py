"""
Sample a trained ACGAN generator to build a synthetic benign/malignant ROI
pool for augmenting the classifier's training set.

    python generate_synthetic.py --checkpoint runs/gan/checkpoints/ddsm_acgan_final.pt \
        --out-dir data/synthetic --n-benign 600 --n-malignant 600
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from ddsm_acgan.data import BENIGN_LABEL, MALIGNANT_LABEL
from ddsm_acgan.models import Generator
from ddsm_acgan.models_improved import ImprovedGenerator


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    netG = (ImprovedGenerator() if args.improved else Generator()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(ckpt["generator"])
    netG.eval()

    out_dir = Path(args.out_dir)
    for name, label, n in [("benign", BENIGN_LABEL, args.n_benign), ("malignant", MALIGNANT_LABEL, args.n_malignant)]:
        class_dir = out_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        with torch.no_grad():
            while generated < n:
                bs = min(args.batch_size, n - generated)
                z = netG.sample_z(bs, device=device)
                labels = torch.full((bs,), label, device=device, dtype=torch.long)
                imgs = netG(labels, z)
                for i in range(bs):
                    vutils.save_image(imgs[i], class_dir / f"{name}_{generated + i:05d}.png",
                                       normalize=True, value_range=(-1, 1))
                generated += bs
        print(f"wrote {generated} synthetic {name} images to {class_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--n-benign", type=int, default=600)
    ap.add_argument("--n-malignant", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--improved", action="store_true",
                     help="Load the Stage 2 improved generator architecture -- must match how the "
                          "checkpoint was trained (--improved on train_gan.py).")
    generate(ap.parse_args())
