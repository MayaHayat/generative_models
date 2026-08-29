"""
Stage 2 improved classifier.

=====================================================================
WHAT CHANGED vs. the Stage 1 reconstruction (covidgan/models.py:build_classifier)
=====================================================================
The paper's detector -- which we reproduced faithfully in Stage 1 -- freezes
the entire VGG16 convolutional base and trains only a ~33K-parameter head:

    frozen ImageNet VGG16 conv base  ->  GAP -> Dense(64) -> Dropout -> Dense(2)

Our Stage 1 analysis (FINDINGS.md Sec. 8.3) showed *why* GAN augmentation could
not help under that design: with the backbone frozen, synthetic images can only
nudge a tiny linear boundary on top of fixed ImageNet features -- they cannot
reshape what the network actually looks at. FID halved (504 -> 273) yet CNN-SA
stayed flat, isolating the cause as the frozen-head ceiling, NOT image quality.

Stage 2 therefore makes two coupled, well-motivated architectural changes:

  1. UNFREEZE THE TOP VGG16 CONV BLOCK (domain fine-tuning of the encoder).
     The last conv block (block 5: three 3x3 conv layers, ~7.08M params) is set
     trainable so the encoder can adapt ImageNet features to chest-X-ray
     texture/pathology instead of being fixed. This is the capacity that lets
     extra (synthetic) data matter. The lower blocks stay frozen -- generic
     edge/texture filters transfer fine and unfreezing everything would badly
     overfit ~900 images.

  2. ADD BATCHNORM TO THE HEAD (normalization / regularization).
     Now that far more parameters train, the head gets a BatchNorm1d after its
     Dense(64) to stabilize the enlarged trainable set and speed convergence:
         GAP -> Dense(64) -> BatchNorm1d(64) -> ReLU -> Dropout -> Dense(2)

Both are on the assignment's "meaningful modification" list ("Change the
encoder", "Add normalization layers"). Because the unfrozen backbone is
pretrained and the head is fresh, the two groups are trained with
DISCRIMINATIVE LEARNING RATES (small LR for the backbone, normal LR for the
head) -- see `param_groups()` -- a standard fine-tuning recipe that keeps the
pretrained filters from being wiped out by large early gradients.

This file does NOT touch Stage 1 code; it defines a parallel classifier so the
two are trivially comparable.
"""
from typing import List

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# VGG16 `.features` is a flat Sequential of 31 layers grouped into 5 conv
# blocks by MaxPool boundaries. These are the indices at which each conv block
# STARTS, so "unfreeze the top k blocks" = make every layer from
# _BLOCK_START_IDX[-k] onward trainable.
#   block 1: 0..3     block 2: 5..8     block 3: 10..15
#   block 4: 17..22   block 5: 24..29
_BLOCK_START_IDX = [0, 5, 10, 17, 24]


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class Stage2Classifier(nn.Module):
    """VGG16 with the top `unfreeze_blocks` conv blocks fine-tuned + a
    BatchNorm-regularized head. Returns raw logits (use CrossEntropyLoss)."""

    def __init__(self, num_classes: int = 2, unfreeze_blocks: int = 1,
                 head_bn: bool = True, pretrained: bool = True):
        super().__init__()
        if not 0 <= unfreeze_blocks <= len(_BLOCK_START_IDX):
            raise ValueError(f"unfreeze_blocks must be 0..{len(_BLOCK_START_IDX)}")
        self.unfreeze_blocks = unfreeze_blocks
        self.head_bn = head_bn

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vgg16(weights=weights).features

        # Freeze everything, then re-enable the top `unfreeze_blocks` blocks.
        _freeze_all(self.backbone)
        if unfreeze_blocks > 0:
            start = _BLOCK_START_IDX[-unfreeze_blocks]
            for idx in range(start, len(self.backbone)):
                for p in self.backbone[idx].parameters():
                    p.requires_grad = True

        head_layers: List[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
        ]
        if head_bn:
            head_layers.append(nn.BatchNorm1d(64))
        head_layers += [
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        ]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def param_groups(self, head_lr: float, backbone_lr: float):
        """Discriminative-LR optimizer groups: the fresh head trains at the full
        `head_lr`; the unfrozen (pretrained) backbone layers train at the much
        smaller `backbone_lr` so their ImageNet filters are only gently nudged.
        Only parameters with requires_grad=True are included."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups


def build_stage2_classifier(num_classes: int = 2, unfreeze_blocks: int = 1,
                            head_bn: bool = True, pretrained: bool = True) -> Stage2Classifier:
    """Factory mirroring covidgan.models.build_classifier, for the improved model."""
    return Stage2Classifier(num_classes=num_classes, unfreeze_blocks=unfreeze_blocks,
                            head_bn=head_bn, pretrained=pretrained)
