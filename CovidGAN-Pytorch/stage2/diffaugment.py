"""
Differentiable Augmentation for Data-Efficient GAN Training (Zhao et al., 2020).

CovidGAN trains on only ~932 images (403 COVID). A discriminator that small
overfits its training set within a few epochs, which starves the generator of a
useful gradient and caps sample diversity/quality. DiffAugment applies the *same*
differentiable augmentation to the discriminator's real and fake inputs on every
step, so D cannot simply memorise the real set, without the augmentation leaking
into the generated distribution (unlike augmenting the dataset directly).

Policy used: color + translation + cutout (the paper's default for limited data).
Applied to both real and fake batches in the D step and to fakes in the G step.

Reference: https://arxiv.org/abs/2006.10738
"""
import torch
import torch.nn.functional as F


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    """Random per-sample integer-ish translation via grid_sample (MPS-safe).

    The upstream DiffAugment used F.pad on 4 dims + fancy indexing, which falls
    back to an unsupported/very slow path on Apple's MPS backend (it hangs). This
    grid_sample formulation is equivalent in spirit (random shift, zero fill) and
    uses only ops MPS supports natively.
    """
    N, C, H, W = x.shape
    # random shift in normalized [-1, 1] grid units, magnitude up to `ratio` of the image
    tx = (torch.rand(N, device=x.device, dtype=x.dtype) * 2 - 1) * (2 * ratio)
    ty = (torch.rand(N, device=x.device, dtype=x.dtype) * 2 - 1) * (2 * ratio)
    theta = torch.zeros(N, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def rand_cutout(x, ratio=0.5):
    """Random per-sample square cutout via a broadcast mask (MPS-safe).

    Avoids the fancy-index scatter of the upstream version (which hangs on MPS);
    builds the mask with pure broadcasting comparisons instead.
    """
    N, C, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    cy = torch.randint(0, H, (N, 1, 1), device=x.device)
    cx = torch.randint(0, W, (N, 1, 1), device=x.device)
    ys = torch.arange(H, device=x.device).view(1, H, 1)
    xs = torch.arange(W, device=x.device).view(1, 1, W)
    inside = ((ys >= cy - ch // 2) & (ys < cy - ch // 2 + ch) &
              (xs >= cx - cw // 2) & (xs < cx - cw // 2 + cw))
    mask = (~inside).to(x.dtype).unsqueeze(1)  # (N,1,H,W): 0 inside the cutout box
    return x * mask


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy="color,translation,cutout"):
    """Apply the DiffAugment policy to a batch of images in [-1, 1]."""
    if not policy:
        return x
    for p in policy.split(","):
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()
