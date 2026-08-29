"""
Stage 2 -- train the IMPROVED CovidGAN.

Same AC-GAN objective and (nearly) the same architecture as Stage 1's
train_gan.py, but with the fixes motivated in stage2/gan_improved.py:

  * noise z ~ N(0, 1) instead of N(0, 0.02)           (diversity)
  * DiffAugment on the discriminator's real+fake inputs (small-data overfitting)
  * spectral-normalised discriminator, DCGAN weight init (stability)
  * optional TTUR (separate G/D learning rates)

    python -m stage2.train_gan_improved --out-dir runs/gan_improved --epochs 300

Checkpoints are drop-in compatible with generate_synthetic.py's format (a dict
with a "generator" key), so the trained model can be sampled the usual way -- or
with stage2/generate_improved.py, which rebuilds the ImprovedGenerator.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from covidgan.data import CXRDataset, read_manifest
from covidgan.models import count_params, pick_device
from stage2.diffaugment import diff_augment
from stage2.gan_improved import ImprovedDiscriminator, ImprovedGenerator


def train(args):
    device = pick_device("cpu" if args.cpu else args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}  seed: {args.seed}  policy: '{args.diffaugment}'  noise_std: {args.noise_std}")

    train_items = read_manifest(args.manifest, "train")
    dataset = CXRDataset(train_items, image_size=112, value_range="tanh", cache=args.cache)
    workers = 0 if args.cache else args.workers
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=workers, drop_last=True)
    print(f"training images: {len(dataset)}")

    netG = ImprovedGenerator(noise_std=args.noise_std).to(device)
    netD = ImprovedDiscriminator().to(device)
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
                    "epoch": epoch, "noise_std": args.noise_std}, path)

    eval_z = netG.sample_z(16, device=device)
    eval_labels = torch.arange(16, device=device) % 2

    for epoch in range(start_epoch, args.epochs):
        g_loss_sum = d_loss_sum = 0.0
        for real_imgs, real_labels in loader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            bs = real_imgs.size(0)
            real_target = torch.full((bs, 1), 0.9, device=device)  # one-sided label smoothing
            fake_target = torch.zeros((bs, 1), device=device)

            # --- Discriminator step (DiffAugment applied to real AND fake) ---
            opt_d.zero_grad()
            real_validity, real_class = netD(augment(real_imgs))
            d_loss_real = bce(real_validity, real_target) + ce(real_class, real_labels)

            z = netG.sample_z(bs, device=device)
            fake_labels = torch.randint(0, 2, (bs,), device=device)
            fake_imgs = netG(fake_labels, z)
            fake_validity, fake_class = netD(augment(fake_imgs.detach()))
            d_loss_fake = bce(fake_validity, fake_target) + ce(fake_class, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # --- Generator step (same augmentation on the fakes it is graded on) ---
            opt_g.zero_grad()
            validity, pred_class = netD(augment(fake_imgs))
            g_loss = bce(validity, torch.ones((bs, 1), device=device)) + ce(pred_class, fake_labels)
            g_loss.backward()
            opt_g.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()

        n_batches = len(loader)
        print(f"epoch {epoch+1}/{args.epochs}  D_loss={d_loss_sum/n_batches:.4f}  G_loss={g_loss_sum/n_batches:.4f}",
              flush=True)
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
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs/gan_improved")
    ap.add_argument("--epochs", type=int, default=300)
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
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", default=None)
    train(ap.parse_args())
