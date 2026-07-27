"""
Sample a trained CovidGAN generator to build the synthetic augmentation
pool, Sec. III-B and Fig. 4B. The paper generated 1,669 synthetic COVID-CXR
and 1,399 synthetic Normal-CXR images; those counts are the defaults here.

    python generate_synthetic.py --checkpoint runs/gan/checkpoints/covidgan_final.pt \
        --out-dir data/synthetic
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from covidgan.data import COVID_LABEL, NORMAL_LABEL
from covidgan.models import Generator


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    netG = Generator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(ckpt["generator"])
    netG.eval()

    out_dir = Path(args.out_dir)
    for name, label, n in [("covid", COVID_LABEL, args.n_covid), ("normal", NORMAL_LABEL, args.n_normal)]:
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
    ap.add_argument("--n-covid", type=int, default=1669)
    ap.add_argument("--n-normal", type=int, default=1399)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    generate(ap.parse_args())
