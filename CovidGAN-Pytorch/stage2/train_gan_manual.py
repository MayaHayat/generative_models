"""
Stage 2 -- MANUAL GAN trainer with a live per-epoch progress display.

Run this yourself in a terminal to watch training progress. It trains EITHER of
the two Stage 2 GANs against the same recipe, selected with --disc, so they are
directly comparable:

  --disc acgan        the AC-GAN improved variant (twin validity+class heads,
                      BCE + cross-entropy, one-sided label smoothing).
                      == stage2/gan_improved.py:ImprovedDiscriminator.
  --disc projection   the PROJECTION-discriminator variant (single conditional
                      score, hinge loss, no auxiliary classifier).
                      == stage2/gan_projection.py:ProjectionDiscriminator.
                      This is the headline Stage 2 GAN change (see that file).

Everything else is held fixed between the two: the ImprovedGenerator
(noise_std=1.0), spectral-norm conv trunk, DiffAugment, DCGAN init, batch size,
LR. So any downstream difference is attributable to the conditioning mechanism.

WHAT YOU SEE
============
A tqdm bar per epoch showing batches/s and the running D/G loss, then a one-line
epoch summary (losses + elapsed + ETA to the final epoch). Samples are written
to <out-dir>/samples every --sample-every epochs and checkpoints to
<out-dir>/checkpoints every --checkpoint-every epochs (plus a final one). All
checkpoints carry a "generator" key, so generate_improved.py samples them
unchanged.

EXAMPLES
========
    # AC-GAN improved (baseline Stage 2 GAN)
    python -m stage2.train_gan_manual --disc acgan \
        --out-dir runs/gan_acgan --epochs 400

    # Projection discriminator (headline Stage 2 GAN)
    python -m stage2.train_gan_manual --disc projection \
        --out-dir runs/gan_projection --epochs 400

Resume either with --resume <checkpoint.pt>.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback: no bar, still prints epoch summaries
    def tqdm(iterable, **kwargs):
        return iterable

from covidgan.data import CXRDataset, read_manifest
from covidgan.models import count_params, pick_device
from stage2.diffaugment import diff_augment
from stage2.gan_improved import ImprovedDiscriminator, ImprovedGenerator
from stage2.gan_projection import ProjectionDiscriminator


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def train(args):
    device = pick_device("cpu" if args.cpu else args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}  disc: {args.disc}  seed: {args.seed}  "
          f"policy: '{args.diffaugment}'  noise_std: {args.noise_std}")

    train_items = read_manifest(args.manifest, "train")
    dataset = CXRDataset(train_items, image_size=112, value_range="tanh", cache=args.cache)
    workers = 0 if args.cache else args.workers
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=workers, drop_last=True)
    print(f"training images: {len(dataset)}  batches/epoch: {len(loader)}")

    netG = ImprovedGenerator(noise_std=args.noise_std).to(device)
    if args.disc == "acgan":
        netD = ImprovedDiscriminator().to(device)
    else:
        netD = ProjectionDiscriminator().to(device)
    print(f"G params: {count_params(netG):,}  D params: {count_params(netD):,}")

    g_lr = args.g_lr if args.g_lr is not None else args.lr
    d_lr = args.d_lr if args.d_lr is not None else args.lr
    opt_g = torch.optim.Adam(netG.parameters(), lr=g_lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(netD.parameters(), lr=d_lr, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    def augment(x):
        return diff_augment(x, policy=args.diffaugment)

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        netG.load_state_dict(ckpt["generator"])
        netD.load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt and "opt_d" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    def save_checkpoint(path, epoch):
        torch.save({"generator": netG.state_dict(), "discriminator": netD.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "epoch": epoch, "noise_std": args.noise_std, "disc": args.disc}, path)

    eval_z = netG.sample_z(16, device=device)
    eval_labels = torch.arange(16, device=device) % 2

    def d_step(real_imgs, real_labels, bs):
        """One discriminator update; returns (d_loss_value, fake_imgs) so the
        G step can reuse the fakes. Branches on the objective."""
        opt_d.zero_grad()
        z = netG.sample_z(bs, device=device)
        fake_labels = torch.randint(0, 2, (bs,), device=device)
        fake_imgs = netG(fake_labels, z)

        if args.disc == "acgan":
            real_target = torch.full((bs, 1), 0.9, device=device)  # one-sided smoothing
            fake_target = torch.zeros((bs, 1), device=device)
            real_validity, real_class = netD(augment(real_imgs))
            d_loss_real = bce(real_validity, real_target) + ce(real_class, real_labels)
            fake_validity, fake_class = netD(augment(fake_imgs.detach()))
            d_loss_fake = bce(fake_validity, fake_target) + ce(fake_class, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        else:  # projection + hinge
            d_real = netD(augment(real_imgs), real_labels)
            d_fake = netD(augment(fake_imgs.detach()), fake_labels)
            d_loss = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

        d_loss.backward()
        opt_d.step()
        return d_loss.item(), fake_imgs, fake_labels

    def g_step(fake_imgs, fake_labels, bs):
        opt_g.zero_grad()
        if args.disc == "acgan":
            validity, pred_class = netD(augment(fake_imgs))
            g_loss = bce(validity, torch.ones((bs, 1), device=device)) + ce(pred_class, fake_labels)
        else:  # projection + hinge
            g_loss = -netD(augment(fake_imgs), fake_labels).mean()
        g_loss.backward()
        opt_g.step()
        return g_loss.item()

    n_batches = len(loader)
    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        g_loss_sum = d_loss_sum = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False,
                   unit="batch", dynamic_ncols=True)
        for real_imgs, real_labels in bar:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            bs = real_imgs.size(0)

            d_loss_val, fake_imgs, fake_labels = d_step(real_imgs, real_labels, bs)
            g_loss_val = g_step(fake_imgs, fake_labels, bs)

            d_loss_sum += d_loss_val
            g_loss_sum += g_loss_val
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(D=f"{d_loss_val:.3f}", G=f"{g_loss_val:.3f}")

        d_avg, g_avg = d_loss_sum / n_batches, g_loss_sum / n_batches
        done = epoch + 1 - start_epoch
        total = args.epochs - start_epoch
        elapsed = time.time() - run_start
        eta = elapsed / done * (total - done)
        print(f"epoch {epoch+1:4d}/{args.epochs}  D_loss={d_avg:.4f}  G_loss={g_avg:.4f}  "
              f"[{_fmt_hms(elapsed)} elapsed, ETA {_fmt_hms(eta)}]", flush=True)

        if device.type == "mps":
            torch.mps.empty_cache()  # guard against MPS memory growth over a long run

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            netG.eval()
            with torch.no_grad():
                samples = netG(eval_labels, eval_z)
            netG.train()
            vutils.save_image(samples, out_dir / "samples" / f"epoch_{epoch+1:04d}.png",
                              nrow=4, normalize=True, value_range=(-1, 1))

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(out_dir / "checkpoints" / f"covidgan_epoch{epoch+1:04d}.pt", epoch + 1)

    save_checkpoint(out_dir / "checkpoints" / "covidgan_final.pt", args.epochs)
    print(f"done. final checkpoint: {out_dir / 'checkpoints' / 'covidgan_final.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disc", choices=["acgan", "projection"], default="projection",
                    help="Which Stage 2 discriminator/objective to train.")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs/gan_projection")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4, help="base lr (used if --g-lr/--d-lr unset)")
    ap.add_argument("--g-lr", type=float, default=None, help="generator lr (TTUR)")
    ap.add_argument("--d-lr", type=float, default=None, help="discriminator lr (TTUR)")
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--noise-std", type=float, default=1.0,
                    help="std of z ~ N(0, std). Stage 1 used 0.02 (near-off); 1.0 is standard.")
    ap.add_argument("--diffaugment", default="color,translation,cutout",
                    help="DiffAugment policy; empty string disables it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", default=None)
    train(ap.parse_args())
