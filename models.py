"""Model definitions for the ARAF-Net Plant Disease Classification project.

The five constructors correspond to the ablation experiments documented in
PlantVillage.ipynb:
    A: build_resnet34_baseline
    B: ResNet34ChannelAttention
    C: ResNet34SpatialAttention
    D: ResNet34ChannelSpatialAttention
    E: ARAFNet

By default, ResNet-34 ImageNet weights are requested, as in the experiment
training cells. Set pretrained=False to initialize without downloading weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _resnet34_backbone(pretrained: bool = True):
    """Create a ResNet-34 backbone using the requested weight setting."""
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    return models.resnet34(weights=weights)


def build_resnet34_baseline(num_classes: int = 7, pretrained: bool = True):
    """Experiment A: ResNet-34 with a replacement classification layer."""
    model = _resnet34_backbone(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ChannelAttention(nn.Module):
    """Channel attention using both global average and max pooling."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention from channel-wise mean and maximum descriptors."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


def _resnet_feature_layers(backbone: nn.Module) -> nn.Sequential:
    """Return ResNet-34 feature layers without its average pool and FC head."""
    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )


class _AttentionResNetBase(nn.Module):
    """Shared ResNet-34 feature extractor and classification head."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        backbone = _resnet34_backbone(pretrained=pretrained)
        self.features = _resnet_feature_layers(backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def classify_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet34ChannelAttention(_AttentionResNetBase):
    """Experiment B: ResNet-34 followed by channel attention."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.channel_attention = ChannelAttention(512, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.channel_attention(x)
        return self.classify_features(x)


class ResNet34SpatialAttention(_AttentionResNetBase):
    """Experiment C: ResNet-34 followed by spatial attention."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.spatial_attention(x)
        return self.classify_features(x)


class ResNet34ChannelSpatialAttention(_AttentionResNetBase):
    """Experiment D: channel attention followed by spatial attention."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.channel_attention = ChannelAttention(512, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.classify_features(x)


class MultiScaleFeatureFusion(nn.Module):
    """Parallel 1x1, 3x3, and 5x5 convolutions followed by 1x1 fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.relu(self.branch1(x))
        f3 = self.relu(self.branch3(x))
        f5 = self.relu(self.branch5(x))
        multi_scale = torch.cat([f1, f3, f5], dim=1)
        return self.relu(self.fusion(multi_scale))


class ARAFNet(nn.Module):
    """Experiment E: ResNet-34 + channel/spatial attention + multi-scale fusion."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        backbone = _resnet34_backbone(pretrained=pretrained)
        self.features = _resnet_feature_layers(backbone)
        self.channel_attention = ChannelAttention(512, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.multi_scale = MultiScaleFeatureFusion(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.multi_scale(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == "__main__":
    # Architecture smoke test; uses random initialization and does not train.
    model = ARAFNet(num_classes=7, pretrained=False)
    sample = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(sample)
    print("ARAFNet output shape:", tuple(output.shape))
