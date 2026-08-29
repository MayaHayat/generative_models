"""
Train CovidGAN (the AC-GAN generator/discriminator pair), Sec. III-B.3.

    python train_gan.py --manifest data/manifest.csv --out-dir runs/gan

Hyperparameters default to the paper's: batch 64, lr 2e-4, Adam beta1 0.5,
2000 epochs, BCE-with-logits for the real/fake head + cross-entropy for the
class head. On CPU this is impractically slow for the full 2000 epochs
(the paper reports ~5h on an RTX 2060) -- use --epochs to cut it down for a
smoke test, or run on a CUDA machine for a full reproduction.

Long runs checkpoint every --checkpoint-every epochs (generator, discriminator
and optimizer state). Pass --resume <checkpoint> to continue an interrupted run
from where it stopped, e.g.:

    python train_gan.py --out-dir runs/gan --resume runs/gan/checkpoints/covidgan_epoch1000.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from covidgan.data import CXRDataset, read_manifest
from covidgan.models import Discriminator, Generator, count_params, pick_device


def train(args):
    device = pick_device("cpu" if args.cpu else args.device)
    print(f"device: {device}")

    train_items = read_manifest(args.manifest, "train")
    dataset = CXRDataset(train_items, image_size=112, value_range="tanh", cache=args.cache)
    # With the whole dataset already decoded in RAM, worker processes only add
    # inter-process copying overhead, so load in-process when cached.
    workers = 0 if args.cache else args.workers
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=workers, drop_last=True)
    print(f"training images: {len(dataset)}"
          + (" (cached in RAM)" if args.cache else ""))

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    print(f"G params: {count_params(netG):,}  D params: {count_params(netD):,}")

    opt_g = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Optionally resume: restore generator, discriminator and (if present) the
    # optimizer state, then continue from the saved epoch. Lets a long 2000-epoch
    # run survive interruptions (e.g. a free-Colab session drop). Checkpoints
    # written before this feature have no optimizer state; those still load
    # (weights only) and just restart Adam's momentum.
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        netG.load_state_dict(ckpt["generator"])
        netD.load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt and "opt_d" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
        else:
            print("resume: checkpoint has no optimizer state; restarting Adam momentum.")
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    def save_checkpoint(path, epoch):
        torch.save({"generator": netG.state_dict(), "discriminator": netD.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "epoch": epoch}, path)

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

            # --- Discriminator step: real batch + fake batch ---
            opt_d.zero_grad()
            real_validity, real_class = netD(real_imgs)
            d_loss_real = bce(real_validity, real_target) + ce(real_class, real_labels)

            z = netG.sample_z(bs, device=device)
            fake_labels = torch.randint(0, 2, (bs,), device=device)
            fake_imgs = netG(fake_labels, z)
            fake_validity, fake_class = netD(fake_imgs.detach())
            d_loss_fake = bce(fake_validity, fake_target) + ce(fake_class, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # --- Generator step: fool D into predicting real + correct class ---
            opt_g.zero_grad()
            validity, pred_class = netD(fake_imgs)
            g_loss = bce(validity, torch.ones((bs, 1), device=device)) + ce(pred_class, fake_labels)
            g_loss.backward()
            opt_g.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()

        n_batches = len(loader)
        print(f"epoch {epoch+1}/{args.epochs}  D_loss={d_loss_sum/n_batches:.4f}  G_loss={g_loss_sum/n_batches:.4f}")

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
    ap.add_argument("--out-dir", default="runs/gan")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2,
                     help="DataLoader worker processes (ignored when --cache is on).")
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--device", default="auto",
                     help="auto (cuda > mps > cpu), or force cuda / mps / cpu. "
                          "'mps' uses the Apple-silicon GPU on M-series Macs.")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                     help="Preload+resize all images into RAM once (default on; the dataset is "
                          "tiny). Use --no-cache to decode from disk every epoch.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (shorthand for --device cpu).")
    ap.add_argument("--resume", default=None,
                     help="Path to a checkpoint (e.g. runs/gan/checkpoints/covidgan_epoch0100.pt) to "
                          "resume from: restores generator/discriminator + optimizer state and continues "
                          "from the saved epoch. Use to recover an interrupted long run.")
    train(ap.parse_args())
