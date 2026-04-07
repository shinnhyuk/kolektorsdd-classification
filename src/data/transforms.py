"""학습/검증용 Albumentations 변환 파이프라인.

단일 파라미터형 `get_train_transforms(cfg)` 함수를 중심으로 구성된다.
실험별 파이프라인 분기는 cfg 값으로 제어하며, 별도 함수를 추가하지 않는다.
기존 코드 호환을 위해 Stage 3/4 함수명은 내부 위임으로 유지한다.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_train_transforms(cfg: DictConfig) -> A.Compose:
    """학습용 단일 파라미터형 증강 파이프라인.

    cfg.augmentation.train의 파라미터를 읽어 변환을 조합한다.
    p=0으로 설정된 변환은 파이프라인에서 제외되어 기존 실험 재현성이 유지된다.

    파이프라인 조합 규칙:
    - letterbox: true이면 LongestMaxSize + PadIfNeeded 적용 (false이면 단순 Resize)
    - clahe_p > 0이면 CLAHE 적용
    - elastic_p > 0이면 ElasticTransform 적용 (Stage 6 추가 실험)
    - horizontal_flip_p > 0이면 HorizontalFlip 적용
    - vertical_flip_p > 0이면 VerticalFlip 적용 (Stage 6 추가 실험)

    대부분의 파이프라인이 공통 구성 요소를 공유하며 파라미터만 다르기 때문에
    실험별로 별도 함수를 만들지 않고 cfg로 조합을 제어한다.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    use_letterbox: bool = aug_cfg.get("letterbox", False)
    clahe_p: float = aug_cfg.get("clahe_p", 0)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    elastic_p: float = aug_cfg.get("elastic_p", 0)
    rotate_limit: int = aug_cfg.get("rotate_limit", 0)
    rotate_p: float = aug_cfg.get("rotate_p", 0)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0)
    horizontal_flip_p: float = aug_cfg.get("horizontal_flip_p", 0)
    vertical_flip_p: float = aug_cfg.get("vertical_flip_p", 0)

    transforms = []

    # 1. 리사이즈 — 레터박스 또는 단순 Resize
    if use_letterbox:
        # 레터박스 리사이즈: 평균 픽셀값(≈186)으로 패딩 — 정규화 후 ~0(중립)
        transforms += [
            A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(
                min_height=h,
                min_width=w,
                border_mode=0,
                fill=186,
            ),
        ]
    else:
        transforms.append(A.Resize(height=h, width=w))

    # 2. CLAHE — 금속 표면 국소 조명 분산 보정
    if clahe_p > 0:
        transforms.append(A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p))

    # 3. ElasticTransform — alpha=1(변위 강도), sigma=50(변형 부드러움): 약한 변형
    if elastic_p > 0:
        transforms.append(A.ElasticTransform(alpha=1, sigma=50, p=elastic_p))

    # 4. ShiftScaleRotate
    transforms.append(
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=rotate_limit,
            border_mode=0,
            p=rotate_p,
        )
    )

    # 5. RandomBrightnessContrast
    transforms.append(A.RandomBrightnessContrast(p=brightness_contrast_p))

    # 6. HorizontalFlip
    if horizontal_flip_p > 0:
        transforms.append(A.HorizontalFlip(p=horizontal_flip_p))

    # 7. VerticalFlip — KolektorSDD 결함 수직 위치 무작위성 반영
    if vertical_flip_p > 0:
        transforms.append(A.VerticalFlip(p=vertical_flip_p))

    # 8. Normalize + ToTensor
    transforms += [
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_data_centric_train_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3/4 학습용 강화 증강 파이프라인 — 레터박스 모드.

    기존 노트북 코드 호환을 위해 유지. 내부적으로 get_train_transforms(cfg)에 위임.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    return get_train_transforms(cfg)


def get_data_centric_train_transforms_no_letterbox(cfg: DictConfig) -> A.Compose:
    """Stage 3 학습용 강화 증강 파이프라인 — 단순 리사이즈 모드.

    기존 노트북 코드 호환을 위해 유지. 내부적으로 get_train_transforms(cfg)에 위임.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정

    Returns:
        A.Compose 변환 객체
    """
    return get_train_transforms(cfg)


def get_data_centric_val_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3+ 검증/테스트용 변환 — 레터박스 모드.

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
