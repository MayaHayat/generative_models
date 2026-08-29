"""
Sample a trained IMPROVED CovidGAN generator into a synthetic pool.

Mirror of generate_synthetic.py but rebuilds the ImprovedGenerator (which uses
the corrected noise scale). Same output layout (COVID/ and Normal/ subfolders),
so evaluate_fid.py and the Stage 2 classifier consume it unchanged.

    python -m stage2.generate_improved \
        --checkpoint runs/gan_improved/checkpoints/covidgan_final.pt \
        --out-dir data/synthetic_improved
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from covidgan.data import CLASS_NAMES
from covidgan.models import pick_device
from stage2.gan_improved import ImprovedGenerator


def generate(args):
    device = pick_device("cpu" if args.cpu else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    noise_std = float(ckpt.get("noise_std", args.noise_std))
    netG = ImprovedGenerator(noise_std=noise_std).to(device)
    netG.load_state_dict(ckpt["generator"])
    netG.eval()

    counts = {"covid": args.n_covid, "normal": args.n_normal}
    for name, n in counts.items():
        label = CLASS_NAMES.index(name)
        class_dir = Path(args.out_dir) / name
        class_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        while generated < n:
            bs = min(args.batch_size, n - generated)
            labels = torch.full((bs,), label, device=device, dtype=torch.long)
            with torch.no_grad():
                z = netG.sample_z(bs, device=device)
                imgs = netG(labels, z)
            for i in range(bs):
                vutils.save_image(imgs[i], class_dir / f"{name}_{generated + i:05d}.png",
                                  normalize=True, value_range=(-1, 1))
            generated += bs
        print(f"{name}: wrote {generated} images to {class_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="data/synthetic_improved")
    ap.add_argument("--n-covid", type=int, default=1669)
    ap.add_argument("--n-normal", type=int, default=1399)
    ap.add_argument("--noise-std", type=float, default=1.0,
                    help="fallback if the checkpoint has no noise_std")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu", action="store_true")
    generate(ap.parse_args())
