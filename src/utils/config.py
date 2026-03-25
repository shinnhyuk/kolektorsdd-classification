"""OmegaConf 기반 config 로더 — base.yaml + experiment yaml merge."""

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    base_path: str = "configs/base.yaml",
    experiment_path: Optional[str] = None,
) -> DictConfig:
    """base.yaml과 experiment yaml을 merge한 config를 반환한다.

    Args:
        base_path: 기본 config 경로 (기본값: configs/base.yaml)
        experiment_path: 실험별 override config 경로 (없으면 base만 반환)

    Returns:
        DictConfig: merge된 OmegaConf 설정 객체
    """
    base_cfg = OmegaConf.load(base_path)

    if experiment_path is not None:
        exp_cfg = OmegaConf.load(experiment_path)
        cfg = OmegaConf.merge(base_cfg, exp_cfg)
    else:
        cfg = base_cfg

    return cfg


def load_state(state_path: str = "state.yaml") -> DictConfig:
    """state.yaml을 로드하여 반환한다.

    Args:
        state_path: state 파일 경로 (기본값: state.yaml)

    Returns:
        DictConfig: state 설정 객체
    """
    return OmegaConf.load(state_path)


def get_normalization_stats(state_path: str = "state.yaml") -> tuple[list[float], list[float]]:
    """state.yaml에서 정규화 통계값(mean, std)을 읽어 반환한다.

    Stage 1 EDA에서 train split 기준으로 계산한 픽셀 통계값을 사용한다.

    Args:
        state_path: state 파일 경로

    Returns:
        (mean, std): 각각 3채널 float 리스트 ([R, G, B] 동일 값 — grayscale)
    """
    state = load_state(state_path)
    mean = list(state.dataset.mean_pixel)
    std = list(state.dataset.std_pixel)
    return mean, std


def resolve_normalization(cfg: DictConfig, state_path: str = "state.yaml") -> DictConfig:
    """cfg 내 normalize_mean/std가 null이면 state.yaml 값으로 채운다.

    Returns:
        인플레이스 수정된 cfg (참조 반환)
    """
    mean, std = get_normalization_stats(state_path)

    if OmegaConf.is_missing(cfg.augmentation.train, "normalize_mean") or cfg.augmentation.train.normalize_mean is None:
        OmegaConf.update(cfg, "augmentation.train.normalize_mean", mean)
    if OmegaConf.is_missing(cfg.augmentation.train, "normalize_std") or cfg.augmentation.train.normalize_std is None:
        OmegaConf.update(cfg, "augmentation.train.normalize_std", std)
    if OmegaConf.is_missing(cfg.augmentation.val, "normalize_mean") or cfg.augmentation.val.normalize_mean is None:
        OmegaConf.update(cfg, "augmentation.val.normalize_mean", mean)
    if OmegaConf.is_missing(cfg.augmentation.val, "normalize_std") or cfg.augmentation.val.normalize_std is None:
        OmegaConf.update(cfg, "augmentation.val.normalize_std", std)

    return cfg
