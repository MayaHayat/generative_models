"""
CovidGAN model definitions, reconstructed from Waheed et al. (2020),
"CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved
Covid-19 Detection", Figs. 1-3 and Sec. III-B.

Generator:  G(c, z) -> 112x112x3 image
Discriminator: D(x) -> (validity logit, class logits)   [AC-GAN, two heads]
Classifier: frozen VGG16 backbone + small trainable head, Sec. II-B.

The paper's Fig. 3 shows the noise branch dense layer taking a 100-d input
(labelled "?x100"), so Z_DIM defaults to 100 -- not the 20,000-d vector used
in this repo's original exploratory notebook, which was a bug: it inflated
the first dense layer to >20M params on its own and doesn't match the
figure.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

Z_DIM = 100
EMBED_DIM = 50
IMAGE_SIZE = 112
NUM_CLASSES = 2


class Generator(nn.Module):
    """AC-GAN generator: label + noise -> 112x112x3 image in [-1, 1]."""

    def __init__(self, num_classes: int = NUM_CLASSES, z_dim: int = Z_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.z_dim = z_dim

        # Label branch: Embedding(50) -> Dense(49, linear) -> reshape 7x7x1
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.label_dense = nn.Linear(embed_dim, 7 * 7 * 1, bias=False)

        # Noise branch: Dense(1024*7*7) -> ReLU -> reshape 7x7x1024
        self.noise_dense = nn.Sequential(
            nn.Linear(z_dim, 1024 * 7 * 7, bias=False),
            nn.ReLU(inplace=True),
        )

        # Concatenated 7x7x1025 -> four transpose-conv upsampling blocks.
        # kernel 5, stride 2, padding 2, output_padding 1 doubles H/W exactly:
        # 7 -> 14 -> 28 -> 56 -> 112.
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
        combined = torch.cat([noise_map, label_map], dim=1)  # 7x7x1025
        return self.upsample(combined)

    def sample_z(self, batch_size: int, device=None) -> torch.Tensor:
        # Paper: "random normal distribution with 0.02 standard deviation"
        return torch.randn(batch_size, self.z_dim, device=device) * 0.02


class Discriminator(nn.Module):
    """AC-GAN discriminator: image -> (validity logit, class logits).

    Outputs raw logits rather than sigmoid/softmax probabilities so training
    can use BCEWithLogitsLoss / CrossEntropyLoss, which are numerically
    stable; apply sigmoid/softmax when you need probabilities at inference.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()

        def down_block(in_c, out_c, stride):
            return [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
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
        self.validity_head = nn.Linear(flat_dim, 1, bias=False)
        self.class_head = nn.Linear(flat_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        return self.validity_head(feats), self.class_head(feats)


def build_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """VGG16 conv base (frozen) + custom head, Sec. II-B of the paper:
    GlobalAveragePooling -> Dense(64, relu) -> Dropout(0.5) -> Dense(num_classes, softmax).

    Returns raw logits; use CrossEntropyLoss for training.
    """
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = vgg16(weights=weights).features
    for param in backbone.parameters():
        param.requires_grad = False

    return nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes),
    )


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
