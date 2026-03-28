""" Baseline CNN — 단순한 3-block 구조.

설계 원칙:
- 사전학습 가중치 없음 (scratch 학습)
- 3개 Conv 블록: 3→16→32→64 채널
- 각 블록: Conv2d + BatchNorm + ReLU + MaxPool
- 분류 헤드: GlobalAvgPool + Dropout(0.5) + Linear(64, 1)
- 출력: logit (sigmoid 미적용) → BCEWithLogitsLoss와 함께 사용

F1 목표 범위: 0.55 ~ 0.75
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU + MaxPool2d 단위 블록."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,  # BatchNorm이 bias 역할을 대신함
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """3-block CNN 베이스라인 모델.

    입력: (B, 3, H, W) — 그레이스케일을 RGB로 변환한 3채널 텐서
    출력: (B, 1) — 결함(1) / 정상(0) 이진분류 logit

    Args:
        cfg: OmegaConf 설정 (model.dropout_rate, model.num_classes 참조)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        dropout_rate: float = cfg.model.dropout_rate
        num_classes: int = cfg.model.num_classes  # 이진분류이므로 1

        # 특징 추출: 3채널 입력 → 64채널 특징맵
        self.features = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=16),   # Block 1: 3  → 16ch
            ConvBlock(in_channels=16, out_channels=32),  # Block 2: 16 → 32ch
            ConvBlock(in_channels=32, out_channels=64),  # Block 3: 32 → 64ch
        )

        # 분류 헤드: GlobalAvgPool → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: (B, 3, H, W) 정규화된 입력 텐서

        Returns:
            (B, 1) logit — BCEWithLogitsLoss에 직접 전달 가능
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(cfg: DictConfig) -> BaselineCNN:
    """cfg에서 BaselineCNN 모델을 생성하여 반환한다.

    Args:
        cfg: merge된 OmegaConf 설정

    Returns:
        초기화된 BaselineCNN 인스턴스
    """
    return BaselineCNN(cfg)
