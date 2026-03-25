"""state.yaml 읽기/쓰기 헬퍼 모듈."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


STATE_PATH = "state.yaml"


def load_state(state_path: str = STATE_PATH) -> DictConfig:
    """state.yaml을 로드하여 DictConfig로 반환한다."""
    return OmegaConf.load(state_path)


def save_state(state: DictConfig, state_path: str = STATE_PATH) -> None:
    """DictConfig를 state.yaml로 저장한다."""
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(state, state_path)


def update_stage_status(
    stage_id: str,
    status: str,
    state_path: str = STATE_PATH,
    agent: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> None:
    """특정 스테이지의 status와 메타데이터를 업데이트한다.

    Args:
        stage_id: 스테이지 번호 문자열 (예: "2")
        status: "pending" | "in_progress" | "complete"
        state_path: state 파일 경로
        agent: 작업 에이전트 이름
        metrics: 기록할 메트릭 dict
        notes: 비고 문자열
    """
    state = load_state(state_path)
    now = datetime.now().isoformat()

    stage_key = f"stages.{stage_id}.status"
    OmegaConf.update(state, stage_key, status)
    OmegaConf.update(state, "project.last_updated", now)

    if status == "complete":
        OmegaConf.update(state, f"stages.{stage_id}.completed_at", now)

    if agent is not None:
        OmegaConf.update(state, f"stages.{stage_id}.agent", agent)

    if metrics is not None:
        for key, value in metrics.items():
            OmegaConf.update(state, f"stages.{stage_id}.metrics.{key}", value)

    if notes is not None:
        OmegaConf.update(state, f"stages.{stage_id}.notes", notes)

    save_state(state, state_path)


def get_stage_metrics(stage_id: str, state_path: str = STATE_PATH) -> dict[str, Any]:
    """특정 스테이지의 metrics dict를 반환한다."""
    state = load_state(state_path)
    metrics_node = state.stages[stage_id].get("metrics", {})
    return OmegaConf.to_container(metrics_node, resolve=True)
