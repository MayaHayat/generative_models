"""
ACGAN generator/discriminator + classifier for DDSM mammography ROI patches
(benign vs malignant), reusing the CovidGAN architecture (Waheed et al. 2020,
Figs. 1-3) unchanged -- it's a generic conditional-GAN + frozen-VGG16-head
setup that isn't tied to chest X-rays. Only the data pipeline (ddsm_acgan/data.py)
is domain-specific.

Generator:  G(c, z) -> 112x112x3 image
Discriminator: D(x) -> (validity logit, class logits)   [AC-GAN, two heads]
Classifier: frozen VGG16 backbone + small trainable head.

Source ROI patches can be any native size/bit-depth (e.g. DDSM's 384x384
16-bit grayscale) -- ROIDataset resizes to 112x112 and replicates grayscale
to 3 channels on load, so the model itself always operates at a fixed
112x112x3 regardless of the source. Changing IMAGE_SIZE here does NOT
change what resolution the model trains at by itself: the generator's
upsample stack (7->14->28->56->112, four doublings) and the discriminator's
hardcoded 7x7x512 flatten dim are both built specifically for 112x112, and
112 isn't reachable by doubling from 7 to reach some other target like 384
anyway -- that would need a redesigned upsample stack, not just a constant
change.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

Z_DIM = 100
EMBED_DIM = 50
IMAGE_SIZE = 112
NUM_CLASSES = 2  # benign, malignant

PAPER_NOISE_STD = 0.02
"""The paper's latent noise scale. At 0.02 the noise vector barely varies, so the
generator's output is driven almost entirely by the class label rather than by z --
the condition behind mode collapse and the class-conditioned fingerprint. Kept as the
default for fidelity to the paper; pass --noise-std 1.0 to train_gan.py to disable it."""


class Generator(nn.Module):
    """AC-GAN generator: label + noise -> 112x112x3 image in [-1, 1]."""

    def __init__(self, num_classes: int = NUM_CLASSES, z_dim: int = Z_DIM, embed_dim: int = EMBED_DIM,
                 noise_std: float = PAPER_NOISE_STD):
        super().__init__()
        self.z_dim = z_dim
        self.noise_std = noise_std

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_dense = nn.Linear(embed_dim, 7 * 7 * 1, bias=False)

        self.noise_dense = nn.Sequential(
            nn.Linear(z_dim, 1024 * 7 * 7, bias=False),
            nn.ReLU(inplace=True),
        )

        def up_block(in_ch, out_ch, final=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=2,
                                    padding=2, output_padding=1, bias=False)
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
        combined = torch.cat([noise_map, label_map], dim=1)
        return self.upsample(combined)

    def sample_z(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device) * self.noise_std


class Discriminator(nn.Module):
    """AC-GAN discriminator: image -> (validity logit, class logits).
    Outputs raw logits; use BCEWithLogitsLoss / CrossEntropyLoss for training."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()

        def down_block(in_c, out_c, stride):
            return [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5),
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
        self.validity_head = nn.Linear(flat_dim, 1, bias=False)
        self.class_head = nn.Linear(flat_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        return self.validity_head(feats), self.class_head(feats)


# VGG16 `.features` is a flat Sequential of 31 layers grouped into 5 conv
# blocks by MaxPool boundaries. These are the indices at which each conv
# block STARTS, so "unfreeze the top k blocks" = make every layer from
# _BLOCK_START_IDX[-k] onward trainable.
#   block 1: 0..3     block 2: 5..8     block 3: 10..15
#   block 4: 17..22   block 5: 24..29
_BLOCK_START_IDX = [0, 5, 10, 17, 24]


class Classifier(nn.Module):
    """VGG16 with the top `unfreeze_blocks` conv blocks fine-tuned + an
    optionally BatchNorm-regularized head. Returns raw logits.

    Rationale (matches the CovidGAN Stage 2 classifier, same architecture
    family): with the whole backbone frozen (unfreeze_blocks=0, the paper's
    original design), only the ~33K head params train, so synthetic images
    can only nudge a small linear boundary on top of fixed ImageNet features
    -- they can't reshape what the network actually looks at. Unfreezing the
    top block(s) lets the highest-level (most task-specific) conv features
    adapt to mammography texture, while lower blocks (generic edge/texture
    detectors, which transfer reasonably across domains) stay frozen and
    cheap to train on a modest dataset. head_bn=True adds BatchNorm1d after
    the head's Dense(64) to stabilize the larger trainable parameter set
    once blocks are unfrozen.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, unfreeze_blocks: int = 0,
                 head_bn: bool = False, pretrained: bool = True):
        super().__init__()
        if not 0 <= unfreeze_blocks <= len(_BLOCK_START_IDX):
            raise ValueError(f"unfreeze_blocks must be 0..{len(_BLOCK_START_IDX)}")
        self.unfreeze_blocks = unfreeze_blocks

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vgg16(weights=weights).features
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            start = _BLOCK_START_IDX[-unfreeze_blocks]
            for idx in range(start, len(self.backbone)):
                for p in self.backbone[idx].parameters():
                    p.requires_grad = True

        head_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 64)]
        if head_bn:
            head_layers.append(nn.BatchNorm1d(64))
        head_layers += [nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(64, num_classes)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def param_groups(self, head_lr: float, backbone_lr: float):
        """Discriminative-LR optimizer groups: the head trains at head_lr;
        the unfrozen (pretrained) backbone layers train at the smaller
        backbone_lr so their ImageNet filters are only gently nudged rather
        than overwritten by large early gradients."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups


def build_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                      unfreeze_blocks: int = 0, head_bn: bool = False) -> Classifier:
    return Classifier(num_classes=num_classes, unfreeze_blocks=unfreeze_blocks,
                       head_bn=head_bn, pretrained=pretrained)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
