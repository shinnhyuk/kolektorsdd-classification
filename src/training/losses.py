"""손실 함수 모음 — FocalLoss 등 커스텀 손실함수."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """이진 분류용 Focal Loss.

    클래스 불균형이 심한 경우(결함 52개 vs 정상 347개) BCEWithLogitsLoss 대비
    어려운 샘플(hard example)에 집중할 수 있도록 focal 가중치 (1-pt)^gamma를 적용한다.

    Args:
        alpha: 결함(양성) 클래스에 부여할 가중치. 정상(음성)은 (1-alpha).
               결함이 소수 클래스이므로 alpha > 0.5로 설정 권장 (기본 0.25 — 원논문 기본값).
        gamma: focal 지수. 클수록 easy example 억제 효과가 강해짐 (기본 2.0 — 원논문 기본값).

    Note:
        - 입력은 logits (sigmoid 이전 값). BCEWithLogitsLoss와 동일한 인터페이스.
        - forward(logits, targets): targets는 0.0/1.0 float 텐서.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss를 계산한다.

        Args:
            logits: 모델 출력 (sigmoid 이전). shape: (B, 1) 또는 (B,)
            targets: 정답 레이블 (0.0 또는 1.0). shape: logits와 동일

        Returns:
            스칼라 loss 값
        """
        # 수치 안정적인 BCE 계산: log(sigmoid(x)) = -softplus(-x)
        # reduction='none'으로 sample별 loss 유지 → focal 가중치 적용 후 평균
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # pt: 정답 클래스에 대한 예측 확률
        # pt = sigmoid(logits)  if target == 1
        # pt = 1 - sigmoid(logits)  if target == 0
        pt = torch.exp(-bce_loss)  # exp(-BCE) = exp(log(pt)) = pt

        # alpha_t: 결함(1)이면 alpha, 정상(0)이면 (1 - alpha)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal 가중치: (1 - pt)^gamma — 잘 맞춘 샘플(pt≈1)은 가중치 ≈ 0
        focal_weight = (1.0 - pt) ** self.gamma

        # 최종 loss: alpha_t * focal_weight * BCE
        loss = alpha_t * focal_weight * bce_loss

        return loss.mean()
