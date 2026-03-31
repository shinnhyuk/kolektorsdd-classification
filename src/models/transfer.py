"""EfficientNet-B0 전이학습 모델 — 2단계 파인튜닝 지원."""

import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Transfer(nn.Module):
    """EfficientNet-B0 기반 이진분류 전이학습 모델.

    출력: (B, 1) logit — BCEWithLogitsLoss와 함께 사용.

    Phase 관리:
        freeze_backbone() : phase1 — backbone 고정, classifier만 학습
        unfreeze_backbone(): phase2 — 전체 레이어 학습

    Grad-CAM target layer: model.features[7][-1] (16x16 feature map)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        dropout_rate: float = cfg.model.get("dropout_rate", 0.2)
        pretrained: bool = cfg.model.get("pretrained", True)

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)

        # backbone: torchvision 원본 구조 그대로 재사용
        self.features = base.features   # Conv blocks (B, 1280, H, W)
        self.avgpool = base.avgpool     # AdaptiveAvgPool2d(1, 1)

        # 헤드: 1000-class → binary logit
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1280, 1),
        )

        if cfg.model.get("freeze_backbone", True):
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Phase 1: backbone 파라미터를 고정한다."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Phase 2: backbone 파라미터 학습을 재개한다."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def build_transfer_model(cfg: DictConfig) -> EfficientNetB0Transfer:
    """cfg에서 EfficientNetB0Transfer 모델을 생성하여 반환한다."""
    return EfficientNetB0Transfer(cfg)
