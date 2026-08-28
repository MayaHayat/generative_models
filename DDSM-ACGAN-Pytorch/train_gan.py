"""
Train the DDSM ACGAN (generator/discriminator pair) on benign/malignant ROI
patches.

    python train_gan.py --manifest data/manifest.csv --out-dir runs/gan

Hyperparameters default to the same values CovidGAN used: batch 64, lr 2e-4,
Adam beta1 0.5, 2000 epochs, BCE-with-logits for the real/fake head + cross-
entropy for the class head.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from ddsm_acgan.data import ROIDataset, read_manifest
from ddsm_acgan.diffaugment import DiffAugment
from ddsm_acgan.models import Discriminator, Generator, count_params
from ddsm_acgan.models_improved import ImprovedDiscriminator, ImprovedGenerator


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"device: {device}  architecture: {'improved' if args.improved else 'baseline'}  seed: {args.seed}")

    train_items = read_manifest(args.manifest, "train")
    dataset = ROIDataset(train_items, image_size=112, value_range="tanh")
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, drop_last=True, generator=shuffle_generator)
    print(f"training images: {len(dataset)}")

    if args.improved:
        netG = ImprovedGenerator(noise_std=args.noise_std).to(device)
        netD = ImprovedDiscriminator().to(device)
    else:
        netG = Generator(noise_std=args.noise_std).to(device)
        netD = Discriminator().to(device)
    print(f"DiffAugment policy: {args.diffaugment or '(off)'}")
    print(f"latent noise std: {args.noise_std}"
          + ("  (paper default -- noise effectively off)" if args.noise_std <= 0.05 else ""))
    print(f"G params: {count_params(netG):,}  D params: {count_params(netD):,}")

    opt_g = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        netG.load_state_dict(ckpt["generator"])
        netD.load_state_dict(ckpt["discriminator"])
        if "opt_g" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
        else:
            print("warning: checkpoint has no optimizer state (pre-resume-support checkpoint) -- "
                  "Adam state restarts fresh, weights still resume correctly.")
        ckpt_std = ckpt.get("noise_std", 0.02)
        if abs(ckpt_std - args.noise_std) > 1e-9:
            raise SystemExit(
                f"--noise-std {args.noise_std} does not match the checkpoint's {ckpt_std}. "
                f"The generator's first dense layer is fitted to a specific noise scale; "
                f"resuming at a different one silently corrupts training. Retrain from scratch "
                f"(drop --resume) or pass --noise-std {ckpt_std}."
            )
        start_epoch = ckpt["epoch"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")
        if start_epoch >= args.epochs:
            raise SystemExit(f"--epochs {args.epochs} <= checkpoint's epoch {start_epoch}; "
                              f"nothing to do, pass a larger --epochs to keep training.")

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args), resume="allow" if args.resume else None)

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    eval_z = netG.sample_z(16, device=device)
    eval_labels = torch.arange(16, device=device) % 2

    for epoch in range(start_epoch, args.epochs):
        g_loss_sum = d_loss_sum = 0.0
        for real_imgs, real_labels in loader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            bs = real_imgs.size(0)
            real_target = torch.full((bs, 1), 0.9, device=device)
            fake_target = torch.zeros((bs, 1), device=device)

            opt_d.zero_grad()
            real_validity, real_class = netD(DiffAugment(real_imgs, args.diffaugment))
            d_loss_real = bce(real_validity, real_target) + ce(real_class, real_labels)

            z = netG.sample_z(bs, device=device)
            fake_labels = torch.randint(0, 2, (bs,), device=device)
            fake_imgs = netG(fake_labels, z)
            fake_validity, fake_class = netD(DiffAugment(fake_imgs.detach(), args.diffaugment))
            d_loss_fake = bce(fake_validity, fake_target) + ce(fake_class, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            validity, pred_class = netD(DiffAugment(fake_imgs, args.diffaugment))
            g_loss = bce(validity, torch.ones((bs, 1), device=device)) + ce(pred_class, fake_labels)
            g_loss.backward()
            opt_g.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()

        n_batches = len(loader)
        d_loss_avg = d_loss_sum / n_batches
        g_loss_avg = g_loss_sum / n_batches
        print(f"epoch {epoch+1}/{args.epochs}  D_loss={d_loss_avg:.4f}  G_loss={g_loss_avg:.4f}")
        if wandb:
            wandb.log({"epoch": epoch + 1, "d_loss": d_loss_avg, "g_loss": g_loss_avg}, step=epoch + 1)

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            netG.eval()
            with torch.no_grad():
                samples = netG(eval_labels, eval_z)
            netG.train()
            sample_path = out_dir / "samples" / f"epoch_{epoch+1:04d}.png"
            vutils.save_image(samples, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
            if wandb:
                wandb.log({"samples": wandb.Image(str(sample_path))}, step=epoch + 1)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(
                {"generator": netG.state_dict(), "discriminator": netD.state_dict(),
                 "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "epoch": epoch + 1,
                 "noise_std": args.noise_std, "improved": args.improved,
                 "diffaugment": args.diffaugment},
                out_dir / "checkpoints" / f"ddsm_acgan_epoch{epoch+1:04d}.pt",
            )

    torch.save({"generator": netG.state_dict(), "discriminator": netD.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "epoch": args.epochs,
                "noise_std": args.noise_std, "improved": args.improved,
                "diffaugment": args.diffaugment},
               out_dir / "checkpoints" / "ddsm_acgan_final.pt")
    print(f"done. final checkpoint: {out_dir / 'checkpoints' / 'ddsm_acgan_final.pt'}")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="runs/gan")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--resume", default=None,
                     help="Path to a checkpoint (e.g. runs/gan/checkpoints/ddsm_acgan_epoch0050.pt) "
                          "to continue training from. --epochs is the new *total* epoch target, "
                          "not additional epochs on top. Must match --improved (baseline and "
                          "improved checkpoints are not interchangeable -- different layer shapes).")
    ap.add_argument("--improved", action="store_true",
                     help="Use the Stage 2 architecture (ddsm_acgan/models_improved.py): spectral-norm "
                          "discriminator + kernel=4/stride=2 generator upsampling, instead of the "
                          "baseline paper-faithful architecture. Use a separate --out-dir from your "
                          "baseline runs so checkpoints/samples don't mix.")
    ap.add_argument("--diffaugment", default="",
                     help="Comma-separated DiffAugment policy applied identically to real and fake "
                          "batches before the discriminator (Zhao et al. 2020). Empty = off. "
                          "Typical: color,translation,cutout. Targets discriminator memorisation on "
                          "small datasets without capping D's capacity the way spectral norm does.")
    ap.add_argument("--noise-std", type=float, default=0.02,
                     help="Std of the latent noise z. The paper uses 0.02, which is small enough "
                          "that z barely varies and the output is driven by the class label alone "
                          "-- the condition behind mode collapse and the class fingerprint. Pass 1.0 "
                          "to switch the noise back on. Checkpoints record this value and "
                          "generate_synthetic.py reads it back, since sampling at a different scale "
                          "than the model was trained at produces garbage.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                     help="Seeds weight init, noise sampling, and data shuffling. Note: matching "
                          "seeds does not guarantee bit-identical results on GPU -- cuDNN's default "
                          "algorithm selection and some CUDA ops are non-deterministic by default "
                          "regardless of seeding (would need torch.use_deterministic_algorithms(True), "
                          "at a real performance cost, for a full GPU guarantee).")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    ap.add_argument("--wandb-project", default="ddsm-acgan")
    ap.add_argument("--wandb-run-name", default=None)
    train(ap.parse_args())
