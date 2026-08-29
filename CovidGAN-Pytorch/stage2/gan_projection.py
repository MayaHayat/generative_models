"""
Stage 2 -- PROJECTION-discriminator CovidGAN (the headline Stage 2 GAN change).

WHY THIS EXISTS
===============
Stage 1's analysis (FINDINGS.md Sec. 8.4) found the decisive GAN failure: a
classifier trained on the synthetic pool ALONE scored 55% on the real test set
(COVID recall 0.22) -- *below* the majority-class floor. The synthetic class
signal carries almost no transferable pathology. The mechanism is the AC-GAN
objective itself: its discriminator has a separate *auxiliary classifier head*,
and the generator is rewarded whenever that head can tell synthetic COVID from
synthetic Normal. It satisfies that cheaply, by imprinting an easy, label-driven
"fingerprint" (via the generator's class embedding) instead of learning the real
per-class pathology. Lowering FID does not fix this -- it is structural to
AC-GAN.

THE FIX -- PROJECTION DISCRIMINATOR (Miyato & Koyama, ICLR 2018,
"cGANs with Projection Discriminator", https://arxiv.org/abs/1802.05637)
========================================================================
Drop the auxiliary classifier. Condition on the label instead with an
inner-product ("projection") term:

    D(x, y) = w . phi(x) + b   +   <embed(y), phi(x)>

where phi(x) is the shared conv feature vector, w.phi + b is the usual
unconditional real/fake score, and <embed(y), phi(x)> is the projection of the
features onto a learned per-class direction. There is NO classifier head to
game, so the generator can only raise D(x, y) by making class y's *features*
genuinely match real class-y images -- exactly the transferable signal AC-GAN
skipped. This is a "modify the loss / improve the discriminator" change on the
assignment's list, and it targets the specific failure we measured.

WHAT IS HELD FIXED (so the comparison is clean)
===============================================
The generator is UNCHANGED -- we reuse stage2.gan_improved.ImprovedGenerator
(noise_std=1.0). The conv trunk, spectral normalisation, DiffAugment and DCGAN
init all match the AC-GAN improved variant. The ONLY difference between the two
Stage 2 GANs is AC-GAN's twin-head classifier vs. this projection head, so any
downstream difference is attributable to the conditioning mechanism, not a
bigger/different model. Checkpoints keep the "generator" key, so
generate_improved.py samples this GAN with no changes.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from covidgan.models import NUM_CLASSES


class ProjectionDiscriminator(nn.Module):
    """Conditional discriminator with projection-based label conditioning.

    Shares the AC-GAN improved trunk (five spectral-normalised 3x3 conv blocks,
    112 -> 7x7x512) but replaces the two-head (validity + class) output with a
    single conditional score:

        forward(x, y) -> (B, 1) logit  =  w.h + b + <embed(y), h>

    where h = ReLU-then-global-sum-pool of the 7x7x512 feature map (a 512-d
    vector, the standard SNGAN-projection pooling). Trained with the hinge loss
    (canonical pairing for spectral-norm + projection). Returns a raw logit.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3, feat_dim: int = 512):
        super().__init__()

        def down_block(in_c, out_c, stride):
            return [
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # 112x112x3 -> 112x112x32 -> 56x56x64 -> 28x28x128 -> 14x14x256 -> 7x7x512
        self.features = nn.Sequential(
            *down_block(in_ch, 32, stride=1),
            *down_block(32, 64, stride=2),
            *down_block(64, 128, stride=2),
            *down_block(128, 256, stride=2),
            *down_block(256, feat_dim, stride=2),
        )
        # Unconditional real/fake score on the pooled feature vector.
        self.validity = spectral_norm(nn.Linear(feat_dim, 1))
        # Per-class direction; the projection term is <embed(y), h>.
        self.label_embed = spectral_norm(nn.Embedding(num_classes, feat_dim))

        self.apply(_proj_init)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)                     # (B, 512, 7, 7)
        h = torch.relu(feats).sum(dim=[2, 3])        # (B, 512) global sum pool
        out = self.validity(h)                       # (B, 1) unconditional score
        proj = (self.label_embed(labels) * h).sum(dim=1, keepdim=True)  # (B, 1)
        return out + proj


def _proj_init(m):
    """DCGAN-style init for conv/linear; Xavier for the class embedding
    (Miyato's projection-GAN default -- the embedding is a projection matrix,
    not a 0.02-std weight)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
