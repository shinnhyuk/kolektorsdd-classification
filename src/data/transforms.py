"""Stage 3 Data-Centric 강화 증강 파이프라인.

이미지 종횡비(500×1250 세로형)를 유지하는 레터박스 리사이즈를 기반으로
CLAHE, 기하 변환, 색상 변환을 조합한다.

레터박스 전략:
    단순 Resize(256,256)는 세로 축을 ~5배 압축하여 결함 패턴이 소실된다.
    LongestMaxSize(256) + PadIfNeeded(256,256) 조합으로 종횡비를 유지하고
    빈 영역을 검정(0)으로 패딩한다.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_data_centric_train_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3 학습용 강화 증강 파이프라인을 반환한다.

    증강 순서:
        1. 레터박스 리사이즈 (종횡비 유지)
        2. CLAHE — 금속 표면 국소 조명 보정
        3. ShiftScaleRotate — 위치/스케일/회전 다양성
        4. ElasticTransform — 결함 형태 변형 다양성
        5. RandomBrightnessContrast — 조명 변동 대응
        6. HorizontalFlip — 좌우 대칭 다양성
        7. Normalize + ToTensorV2

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정.
             cfg.data.image_size, cfg.augmentation.train.* 키를 사용한다.

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.train
    h, w = cfg.data.image_size

    # 정규화 통계: base.yaml에서 읽어 오며, resolve_normalization()이 train 기준값을 주입
    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    # cfg에서 파라미터 읽기 (기본값은 이슈 명세 기준)
    clahe_p: float = aug_cfg.get("clahe_p", 0.5)
    clahe_clip_limit: float = aug_cfg.get("clahe_clip_limit", 2.0)
    rotate_p: float = aug_cfg.get("rotate_p", 0.4)
    rotate_limit: int = aug_cfg.get("rotate_limit", 20)
    elastic_p: float = aug_cfg.get("elastic_transform_p", 0.3)
    brightness_contrast_p: float = aug_cfg.get("brightness_contrast_p", 0.4)
    hflip_p: float = aug_cfg.get("horizontal_flip_p", 0.5)

    transforms = [
        # 1. 레터박스 리사이즈: 긴 쪽을 max_size에 맞추고 나머지를 검정 패딩
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,   # cv2.BORDER_CONSTANT
            value=0,         # 검정 패딩
        ),
        # 2. CLAHE: 금속 표면 국소 조명 분산 보정
        A.CLAHE(clip_limit=clahe_clip_limit, p=clahe_p),
        # 3. ShiftScaleRotate: 위치·스케일·회전 다양성
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=rotate_limit,
            border_mode=0,
            p=rotate_p,
        ),
        # 4. ElasticTransform: 결함 외형 변형 다양성
        A.ElasticTransform(p=elastic_p),
        # 5. RandomBrightnessContrast: 조명 변동 대응
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        # 6. HorizontalFlip: 좌우 대칭
        A.HorizontalFlip(p=hflip_p),
        # 7. 정규화 + 텐서 변환
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_data_centric_val_transforms(cfg: DictConfig) -> A.Compose:
    """Stage 3 검증/테스트용 변환 파이프라인을 반환한다.

    증강은 적용하지 않으며, 학습과 동일한 레터박스 리사이즈만 수행한다.
    일관된 입력 해상도·종횡비를 보장하기 위해 val에도 레터박스를 적용한다.

    Args:
        cfg: resolve_normalization() 처리가 완료된 OmegaConf 설정.
             cfg.data.image_size, cfg.augmentation.val.* 키를 사용한다.

    Returns:
        A.Compose 변환 객체
    """
    aug_cfg = cfg.augmentation.val  # noqa: F841 — val 전용 파라미터 확장 대비
    h, w = cfg.data.image_size

    mean = list(cfg.augmentation.val.normalize_mean)
    std = list(cfg.augmentation.val.normalize_std)

    transforms = [
        # 레터박스 리사이즈 (train과 동일 — 종횡비 유지, 검정 패딩)
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            value=0,
        ),
        # 정규화 + 텐서 변환 (증강 없음)
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)
