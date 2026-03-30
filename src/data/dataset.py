"""KolektorSDD 이진분류 Dataset 구현."""

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.samplers import build_weighted_sampler
from src.data.transforms import (
    get_data_centric_train_transforms,
    get_data_centric_val_transforms,
)


class KolektorDataset(Dataset):
    """CSV(image_path, label) 기반 KolektorSDD 데이터셋.

    Args:
        csv_path: image_path, label 두 컬럼을 가진 CSV 경로
        transform: albumentations Compose 변환 객체 (None이면 Tensor 변환만)
        root_dir: image_path가 상대 경로일 때 기준 디렉토리 (None이면 절대 경로 사용)
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        root_dir: Optional[str] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.root_dir = root_dir

        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV에 'image_path', 'label' 컬럼이 필요합니다.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        if self.root_dir is not None:
            img_path = str(Path(self.root_dir) / img_path)

        # 그레이스케일 이미지를 RGB 3채널로 변환 — Stage 2~4 동일 dataset.py 재사용
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        label = float(row["label"])

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_labels(self) -> list[int]:
        """WeightedRandomSampler 구성 등에 활용할 레이블 목록 반환."""
        return self.df["label"].tolist()


def get_train_transforms(cfg: DictConfig) -> A.Compose:
    """학습용 albumentations 변환 파이프라인을 반환한다.

    HorizontalFlip + Resize + Normalize만 사용.

    Args:
        cfg: merge된 OmegaConf 설정 (augmentation.train.* 키 사용)

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(aug_cfg.normalize_mean)
    std = list(aug_cfg.normalize_std)

    transforms = [
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=aug_cfg.horizontal_flip_p),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(cfg: DictConfig) -> A.Compose:
    """검증/테스트용 albumentations 변환 파이프라인을 반환한다.

    증강 없이 Resize + Normalize만 적용.

    Args:
        cfg: merge된 OmegaConf 설정 (augmentation.val.* 키 사용)

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.val
    h, w = cfg.data.image_size

    mean = list(aug_cfg.normalize_mean)
    std = list(aug_cfg.normalize_std)

    transforms = [
        A.Resize(height=h, width=w),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """cfg에서 train/val/test DataLoader를 생성하여 반환한다.

    Stage 3 Data-Centric 모드 지원:
        - cfg.augmentation.train.letterbox == True이면 Stage 3 강화 증강 파이프라인 사용
        - cfg.data.use_weighted_sampler == True이면 WeightedRandomSampler 적용 (shuffle=False)

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        (train_loader, val_loader, test_loader) 튜플
    """
    # letterbox 플래그로 Stage 3 증강 파이프라인 선택
    use_letterbox: bool = cfg.augmentation.train.get("letterbox", False)

    if use_letterbox:
        # Stage 3: 레터박스 리사이즈 + CLAHE + 강화 증강
        train_transform = get_data_centric_train_transforms(cfg)
        val_transform = get_data_centric_val_transforms(cfg)
    else:
        # Stage 2 이하: 기존 간단 증강 파이프라인 유지 (Stage 2 호환성)
        train_transform = get_train_transforms(cfg)
        val_transform = get_val_transforms(cfg)

    train_ds = KolektorDataset(cfg.data.train_csv, transform=train_transform)
    val_ds = KolektorDataset(cfg.data.val_csv, transform=val_transform)
    test_ds = KolektorDataset(cfg.data.test_csv, transform=val_transform)

    # WeightedRandomSampler 사용 여부 결정
    use_weighted_sampler: bool = cfg.data.get("use_weighted_sampler", False)

    if use_weighted_sampler:
        # WeightedRandomSampler와 shuffle 동시 사용 불가 — shuffle=False
        sampler = build_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,  # sampler 사용 시 shuffle 비활성화 필수
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
        )
    else:
        # 기본: 무작위 셔플 (Stage 2 이하 동작 유지)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    return train_loader, val_loader, test_loader
