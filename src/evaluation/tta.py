"""TTA(Test Time Augmentation) 평가 모듈.

단일 추론 시 이미지 방향 변환에 민감하게 반응하는 편향을 줄이기 위해,
원본과 수평 플립 조합을 여러 번 추론하여 확률 평균을 취한다.
재학습 없이 추론 방식만 변경하므로, 기존 모델 체크포인트를 그대로 활용한다.

기대 효과:
- 방향 편향 감소: 수평 플립 증강 결과를 평균하여 좌우 방향 의존성 완화
- 예측 안정화: 여러 뷰의 앙상블 효과로 분산 감소
- Recall 향상: FN(놓친 결함) 감소 기대
"""

from typing import Any

import albumentations as A
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics


class TTAEvaluator:
    """TTA(Test Time Augmentation) 기반 모델 평가기.

    원본 이미지와 HorizontalFlip 증강 이미지를 조합하여 n_aug회 추론한 뒤
    sigmoid 확률의 평균을 최종 예측값으로 사용한다.

    방향 편향 감소 근거:
        학습 데이터에 HorizontalFlip(p=0.5)이 포함되어 있으나, 개별 이미지에 따라
        모델이 특정 방향에 과적합할 수 있다. TTA로 양방향 예측을 평균하면
        방향에 무관한 robust한 확률 추정이 가능하다.

    기대 효과:
        - Recall↑ (FN 감소): 방향 편향으로 놓쳤던 결함을 잡아낼 가능성 증가
        - 예측 분산 감소: 단일 추론 대비 앙상블 효과

    Attributes:
        model: 평가할 PyTorch 모델 (eval 모드로 설정됨)
        cfg: OmegaConf 설정 객체 (augmentation.val 섹션 필요)
        device: 추론에 사용할 torch.device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        """TTAEvaluator를 초기화한다.

        Args:
            model: 평가할 PyTorch 모델
            cfg: OmegaConf 설정 (augmentation.val.normalize_mean/std, data.image_size 필요)
            device: 추론 디바이스 (cuda 또는 cpu)
        """
        self.model = model
        self.cfg = cfg
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        # val normalize 파라미터 — 항상 동일하게 적용
        mean = list(cfg.augmentation.val.normalize_mean)
        std = list(cfg.augmentation.val.normalize_std)

        # TTA용 증강: HorizontalFlip + 정규화
        # (loader 이미지는 이미 정규화된 tensor이므로, de-normalize 후 재적용)
        self._flip_transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ])

        # 원본 재정규화용 (tensor → numpy → normalize → tensor)
        self._base_transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ])

        # de-normalize용 파라미터 (numpy 변환 시 원본 픽셀값 복원)
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

    def _tensor_to_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        """정규화된 tensor를 uint8 numpy 배열로 변환한다.

        loader에서 꺼낸 이미지는 이미 normalize + ToTensor 처리된 상태이므로,
        albumentations 재적용을 위해 원본 uint8 이미지로 역변환이 필요하다.

        Args:
            tensor: 정규화된 이미지 tensor (C, H, W), float32

        Returns:
            uint8 numpy 배열 (H, W, C), 0~255 범위
        """
        # (C, H, W) → (H, W, C)
        img = tensor.cpu().numpy().transpose(1, 2, 0)

        # de-normalize: pixel = (normalized * std + mean) * 255
        img = (img * self._std + self._mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_tta_augmentation(
        self, img_uint8: np.ndarray, aug_idx: int
    ) -> torch.Tensor:
        """단일 이미지에 TTA 증강을 적용하고 tensor로 반환한다.

        증강 전략:
            - aug_idx == 0: 원본 (flip 없음)
            - aug_idx >= 1: HorizontalFlip 적용

        Args:
            img_uint8: uint8 numpy 이미지 (H, W, C)
            aug_idx: 증강 인덱스 (0=원본, 1~=flip)

        Returns:
            정규화된 float32 tensor (C, H, W)
        """
        if aug_idx == 0:
            # 원본 — 정규화만 재적용
            augmented = self._base_transform(image=img_uint8)
        else:
            # HorizontalFlip + 정규화
            augmented = self._flip_transform(image=img_uint8)

        img_normalized = augmented["image"]  # (H, W, C), float32

        # (H, W, C) → (C, H, W) tensor로 변환
        return torch.from_numpy(img_normalized.transpose(2, 0, 1))

    def predict(
        self,
        loader: DataLoader,
        n_aug: int = 5,
    ) -> np.ndarray:
        """TTA로 예측 확률 배열을 반환한다.

        각 이미지에 대해 n_aug회 추론(원본 1회 + HorizontalFlip (n_aug-1)회)한 뒤
        sigmoid 확률의 산술 평균을 최종 예측값으로 사용한다.

        증강 조합:
            - aug_idx=0: 원본 이미지 (방향 변환 없음)
            - aug_idx=1~(n_aug-1): HorizontalFlip 적용

        Args:
            loader: 검증/테스트용 DataLoader
                    각 배치는 (images: Tensor[B,C,H,W], labels: Tensor[B]) 형태
            n_aug: 총 증강 횟수 (기본값 5 = 원본 1회 + flip 4회)
                   값이 클수록 예측이 안정되나 추론 시간 증가

        Returns:
            np.ndarray: sigmoid 확률 평균값 (shape: [N], dtype: float32)
        """
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                images, _ = batch  # labels는 evaluate()에서 별도 수집
                batch_size = images.size(0)

                # 이 배치의 각 증강 뷰별 확률 합산 (shape: [B])
                batch_proba_sum = torch.zeros(batch_size, dtype=torch.float32)

                for aug_idx in range(n_aug):
                    # 배치 내 각 이미지에 TTA 증강 적용 후 재조립
                    augmented_batch = []
                    for i in range(batch_size):
                        img_uint8 = self._tensor_to_uint8(images[i])
                        aug_tensor = self._apply_tta_augmentation(img_uint8, aug_idx)
                        augmented_batch.append(aug_tensor)

                    # 증강된 배치를 device로 이동하여 추론
                    aug_batch_tensor = torch.stack(augmented_batch).to(self.device)

                    # 모델 출력 (logits) → sigmoid 확률
                    logits = self.model(aug_batch_tensor)
                    proba = torch.sigmoid(logits).squeeze(1).cpu()

                    batch_proba_sum += proba

                # n_aug회 평균
                batch_proba_mean = (batch_proba_sum / n_aug).numpy()
                all_proba.append(batch_proba_mean)

        return np.concatenate(all_proba, axis=0).astype(np.float32)

    def evaluate(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
        n_aug: int = 5,
    ) -> dict[str, Any]:
        """TTA 기반 전체 평가를 수행하고 메트릭 dict를 반환한다.

        Args:
            loader: 검증/테스트용 DataLoader
            threshold: 이진 분류 임계값 (기본값 0.5, Stage 5 최적값 사용 권장)
            n_aug: TTA 증강 횟수 (predict()에 전달)

        Returns:
            compute_metrics() 반환 형식과 동일한 dict:
            {
                "f1": float,
                "auc_roc": float,
                "precision": float,
                "recall": float,
                "average_precision": float,
                "confusion_matrix": [[TN, FP], [FN, TP]],
                "threshold": float,
            }
        """
        # y_true 수집 — loader를 순회하며 레이블 추출
        y_true_list: list[np.ndarray] = []
        for _, labels in loader:
            y_true_list.append(labels.numpy())
        y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)

        # TTA 예측 확률 추출
        y_pred_proba = self.predict(loader, n_aug=n_aug)

        # 메트릭 계산 — src/evaluation/metrics.py 활용
        return compute_metrics(y_true, y_pred_proba, threshold=threshold)
