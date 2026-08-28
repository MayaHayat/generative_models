"""
DiffAugment -- Differentiable Augmentation for Data-Efficient GAN Training
(Zhao, Liu, Lin, Zhu & Han, NeurIPS 2020).

Why this exists here: with only 1,318 real training ROIs the discriminator can
memorize the training set outright, at which point it stops teaching the
generator anything -- the failure visible in this project as the baseline's
epoch-684 divergence (D_loss 1.17 -> 0.38 while G_loss rose 2.32 -> 7.81) and
as a real-vs-synthetic probe separating every pool at 100% held-out accuracy.

The naive fix -- augmenting only the real images -- backfires in a GAN: the
generator is trained to fool D, so if every "real" example has a cutout hole,
G learns to draw holes. DiffAugment applies the *same* augmentation policy to
real and fake batches alike and keeps every operation differentiable, so
gradients flow back through the augmentation into G. D sees fresh views and
cannot memorize; G is not pushed toward artifacts, because its own output is
augmented on the same terms.

Contrast with spectral normalization, the other stabilizer in this codebase:
spectral norm buys stability by *weakening* D (bounding its Lipschitz constant),
which cost peak generation quality here. DiffAugment instead makes D's task
harder without capping its capacity.

    from ddsm_acgan.diffaugment import DiffAugment
    real_validity, real_class = netD(DiffAugment(real_imgs, policy))
    fake_validity, fake_class = netD(DiffAugment(fake_imgs.detach(), policy))

Note for this dataset: ROI patches are grayscale replicated across three
channels, so R=G=B and `rand_saturation` is a no-op on them (it scales the
per-pixel deviation from the channel mean, which is identically zero here).
`rand_brightness` and `rand_contrast` still apply, as do translation and
cutout. Kept in the 'color' policy anyway so the recipe matches the reference
implementation rather than silently diverging from it.

Images are expected in the generator's tanh range [-1, 1], which is what
train_gan.py uses.
"""
import torch
import torch.nn.functional as F

DEFAULT_POLICY = "color,translation,cutout"


def DiffAugment(x: torch.Tensor, policy: str = "") -> torch.Tensor:
    """Apply the comma-separated `policy` to a batch. Empty policy = identity,
    so callers can pass the flag through unconditionally."""
    if not policy:
        return x
    for p in policy.split(","):
        p = p.strip()
        if p not in AUGMENT_FNS:
            raise ValueError(f"unknown DiffAugment policy {p!r}; "
                             f"choose from {sorted(AUGMENT_FNS)}")
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - x_mean) * factor + x_mean


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * factor + x_mean


def rand_translation(x, ratio: float = 0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return (x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2).contiguous())


def rand_cutout(x, ratio: float = 0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2),
                             size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
