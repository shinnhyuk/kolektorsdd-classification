"""Stage 3 Data-Centric 강화 증강 파이프라인.

두 가지 모드를 제공한다:
- 레터박스 모드: LongestMaxSize + PadIfNeeded로 종횡비 유지 (Stage 4+ 권장)
- 단순 리사이즈 모드: 기존 Resize + CLAHE + RandomBrightnessContrast (BaselineCNN 호환)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_data_centric_train_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3 학습용 강화 증강 파이프라인 — 레터박스 모드.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    clahe_p: float = aug_cfg.get("clahe_p", 0.5)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    rotate_p: float = aug_cfg.get("rotate_p", 0.0)
    rotate_limit: int = aug_cfg.get("rotate_limit", 0)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0.4)
    hflip_p: float = aug_cfg.get("horizontal_flip_p", 0.5)

    transforms = [
        # 레터박스 리사이즈: 평균 픽셀값(≈186)으로 패딩 — 정규화 후 ~0(중립)
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            fill=186,
        ),
        A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=rotate_limit,
            border_mode=0,
            p=rotate_p,
        ),
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.HorizontalFlip(p=hflip_p),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_data_centric_train_transforms_no_letterbox(cfg: DictConfig) -> A.Compose:
    """Stage 3 학습용 강화 증강 파이프라인 — 단순 리사이즈 모드.

    레터박스 없이 단순 Resize + CLAHE + RandomBrightnessContrast를 사용한다.
    BaselineCNN처럼 capacity가 제한된 모델에서 레터박스 패딩이 오히려
    혼란을 주는 경우에 사용한다.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    clahe_p: float = aug_cfg.get("clahe_p", 0.5)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0.4)
    hflip_p: float = aug_cfg.get("horizontal_flip_p", 0.5)

    transforms = [
        # 단순 리사이즈 — BaselineCNN 호환 (레터박스 패딩 없음)
        A.Resize(height=h, width=w),
        A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p),
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.HorizontalFlip(p=hflip_p),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_additional_elastic_train_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 6 ElasticTransform 실험용 학습 증강 파이프라인 — 레터박스 모드.

    근거: Stage 3 Data-Centric에서 BaselineCNN 대상 ElasticTransform은 효과 없었으나,
    EfficientNet-B0은 ImageNet 사전학습으로 풍부한 특징 표현을 보유해
    과소적합 위험이 낮음. 결함 형태 미세 변형이 정규화 효과로 작용할 가능성.
    기대효과: 과적합 감소 → val/test F1 향상.

    alpha=1, sigma=50은 결함이 육안으로 식별 가능한 수준의 약한 변형을 보장한다.
    elastic_p는 cfg.augmentation.train.elastic_p에서 읽으며 기본값 0.15.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    clahe_p: float = aug_cfg.get("clahe_p", 0.5)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    elastic_p: float = aug_cfg.get("elastic_p", 0.15)
    rotate_p: float = aug_cfg.get("rotate_p", 0.0)
    rotate_limit: int = aug_cfg.get("rotate_limit", 0)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0.4)
    hflip_p: float = aug_cfg.get("horizontal_flip_p", 0.5)

    transforms = [
        # 레터박스 리사이즈: 평균 픽셀값(≈186)으로 패딩 — 정규화 후 ~0(중립)
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            fill=186,
        ),
        A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p),
        # ElasticTransform: alpha=1(변위 강도), sigma=50(변형 부드러움) — 약한 변형
        A.ElasticTransform(alpha=1, sigma=50, p=elastic_p),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=rotate_limit,
            border_mode=0,
            p=rotate_p,
        ),
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.HorizontalFlip(p=hflip_p),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_additional_vflip_train_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 6 VerticalFlip 실험용 증강 파이프라인 — 레터박스 모드.

    근거: KolektorSDD 결함은 표면 위치가 수직 방향으로도 무작위 분포.
    수직 비대칭성이 없으므로 VerticalFlip 추가는 물리적으로 타당함.
    기대효과: 결함 위치 불변성 강화 → 과적합 감소, 일반화 성능 향상.

    vertical_flip_p는 cfg.augmentation.train.vertical_flip_p에서 읽으며 기본값 0.5.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    clahe_p: float = aug_cfg.get("clahe_p", 0.5)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    rotate_p: float = aug_cfg.get("rotate_p", 0.0)
    rotate_limit: int = aug_cfg.get("rotate_limit", 0)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0.4)
    hflip_p: float = aug_cfg.get("horizontal_flip_p", 0.5)
    vflip_p: float = aug_cfg.get("vertical_flip_p", 0.5)

    transforms = [
        # 레터박스 리사이즈: 평균 픽셀값(≈186)으로 패딩 — 정규화 후 ~0(중립)
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            fill=186,
        ),
        A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=rotate_limit,
            border_mode=0,
            p=rotate_p,
        ),
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.HorizontalFlip(p=hflip_p),
        # VerticalFlip: KolektorSDD 결함 수직 위치 무작위성 반영 — 수직 비대칭성 없음
        A.VerticalFlip(p=vflip_p),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_data_centric_val_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3 검증/테스트용 변환 — 레터박스 모드.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    transforms = [
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            fill=186,
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)
