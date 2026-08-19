"""
Improved DDSM ACGAN generator/discriminator. Two changes vs.
ddsm_acgan/models.py, both targeting generation quality (evaluated via FID
/ visual inspection, not classifier accuracy directly):

1. Spectral normalization on every discriminator layer (replacing plain
   BatchNorm), following Miyato et al. 2018 ("Spectral Normalization for
   GANs"). Constrains the discriminator's Lipschitz constant, stabilizing
   AC-GAN adversarial training -- targets the D/G loss oscillation seen
   across every DDSM training run so far.

2. kernel_size=4, stride=2 transpose convolutions in the generator
   (baseline uses kernel_size=5, stride=2). 4 is evenly divisible by the
   stride 2; 5 is not -- exactly the condition that produces checkerboard
   artifacts in transposed-conv upsampling (Odena et al. 2016,
   "Deconvolution and Checkerboard Artifacts"). This is the same faint grid
   pattern directly visible in this project's own DDSM sample grids.

Same overall topology, channel widths, and roughly the same parameter count
as the baseline in models.py, so comparisons between the two are
apples-to-apples -- only these two mechanisms differ.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

Z_DIM = 100
EMBED_DIM = 50
IMAGE_SIZE = 112
NUM_CLASSES = 2  # benign, malignant


class ImprovedGenerator(nn.Module):
    """Same label/noise conditioning as the baseline Generator; upsampling
    uses kernel=4/stride=2 transpose convs (checkerboard-artifact-free)
    instead of kernel=5/stride=2."""

    def __init__(self, num_classes: int = NUM_CLASSES, z_dim: int = Z_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.z_dim = z_dim

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_dense = nn.Linear(embed_dim, 7 * 7 * 1, bias=False)

        self.noise_dense = nn.Sequential(
            nn.Linear(z_dim, 1024 * 7 * 7, bias=False),
            nn.ReLU(inplace=True),
        )

        def up_block(in_ch, out_ch, final=False):
            # kernel=4, stride=2, padding=1 doubles H/W exactly (7->14->28->56->112)
            # with kernel evenly divisible by stride -- no checkerboard overlap.
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                                    padding=1, bias=False)
            ]
            if final:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            return layers

        self.upsample = nn.Sequential(
            *up_block(1024 + 1, 512),
            *up_block(512, 256),
            *up_block(256, 128),
            *up_block(128, 3, final=True),
        )

    def forward(self, labels: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        label_map = self.label_dense(self.label_embed(labels)).view(-1, 1, 7, 7)
        noise_map = self.noise_dense(z).view(-1, 1024, 7, 7)
        combined = torch.cat([noise_map, label_map], dim=1)  # 7x7x1025
        return self.upsample(combined)

    def sample_z(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device) * 0.02


class ImprovedDiscriminator(nn.Module):
    """Same conv topology as the baseline Discriminator, but every conv and
    the two output heads are spectral-normalized instead of using
    BatchNorm -- SNGAN-style stabilization. Outputs raw logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()

        def down_block(in_c, out_c, stride):
            return [
                spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
            ]

        # 112x112x3 -> 112x112x32 -> 56x56x64 -> 28x28x128 -> 14x14x256 -> 7x7x512
        self.features = nn.Sequential(
            *down_block(in_ch, 32, stride=1),
            *down_block(32, 64, stride=2),
            *down_block(64, 128, stride=2),
            *down_block(128, 256, stride=2),
            *down_block(256, 512, stride=2),
            nn.Flatten(),
        )
        flat_dim = 7 * 7 * 512
        self.validity_head = spectral_norm(nn.Linear(flat_dim, 1, bias=False))
        self.class_head = spectral_norm(nn.Linear(flat_dim, num_classes, bias=False))

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        return self.validity_head(feats), self.class_head(feats)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
