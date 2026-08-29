"""
Stage 2 -- improved CovidGAN generator / discriminator.

The Stage 1 reconstruction (covidgan/models.py) faithfully reproduces the paper,
but its analysis (FINDINGS.md / report Test 4-5) found the synthetic images carry
no transferable pathology and FID stays high (~273). Reading the code turned up
concrete, likely causes:

  1. NOISE IS NEARLY OFF. Stage 1 samples z ~ N(0, 0.02): the noise vector barely
     varies, so the generator's output is driven almost entirely by the class
     label. That crushes sample diversity (high FID) and turns the class signal
     into an easy "label fingerprint" rather than real pathology. 0.02 is the
     classic DCGAN *weight-init* std, so this looks like a reconstruction slip
     (weight-init std applied to the noise). Fix: z ~ N(0, noise_std) with
     noise_std=1.0, the standard choice.
  2. NO SMALL-DATA REGULARISATION. 403 COVID images overfit a discriminator fast
     -> handled by DiffAugment in the trainer (stage2/diffaugment.py).
  3. UNSTABLE D. We add spectral normalisation to the discriminator (SN-GAN) and
     drop the unusual Dropout, for a smoother D and more stable training.

Weights are initialised DCGAN-style (N(0, 0.02)) -- the *correct* place for the
0.02. The architecture (layer shapes, param counts) is otherwise unchanged from
Stage 1, so improvements are attributable to the training recipe, not a bigger
model. Checkpoints are drop-in compatible with generate_* (dict key "generator").
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from covidgan.models import Generator, NUM_CLASSES, Z_DIM


class ImprovedGenerator(Generator):
    """Same architecture as the Stage 1 Generator, but with a usable noise scale.

    The only behavioural change is sample_z: Stage 1 hard-codes a 0.02 std, which
    nearly zeroes the noise; here noise_std defaults to 1.0 (standard normal).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, z_dim: int = Z_DIM,
                 noise_std: float = 1.0):
        super().__init__(num_classes=num_classes, z_dim=z_dim)
        self.noise_std = noise_std
        self.apply(_dcgan_init)

    def sample_z(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device) * self.noise_std


class ImprovedDiscriminator(nn.Module):
    """AC-GAN discriminator with spectral normalisation and no dropout.

    Two heads (validity + class) exactly as Stage 1, so the AC-GAN objective is
    unchanged; only the layer conditioning (spectral norm) differs.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()

        def down_block(in_c, out_c, stride):
            return [
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.features = nn.Sequential(
            *down_block(in_ch, 32, stride=1),
            *down_block(32, 64, stride=2),
            *down_block(64, 128, stride=2),
            *down_block(128, 256, stride=2),
            *down_block(256, 512, stride=2),
            nn.Flatten(),
        )
        flat_dim = 7 * 7 * 512
        self.validity_head = spectral_norm(nn.Linear(flat_dim, 1))
        self.class_head = spectral_norm(nn.Linear(flat_dim, num_classes))
        self.apply(_dcgan_init)

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        return self.validity_head(feats), self.class_head(feats)


def _dcgan_init(m):
    """DCGAN-style init: conv/linear ~ N(0, 0.02); BatchNorm gamma ~ N(1, 0.02)."""
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
