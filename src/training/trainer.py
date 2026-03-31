"""Trainer 클래스 — 학습 루프, early stopping, 체크포인트, TensorBoard 로깅."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.metrics import compute_metrics
from src.training.losses import FocalLoss
from src.utils.logger import get_logger


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """KolektorSDD 이진분류 학습 관리 클래스.

    Args:
        model: 학습할 PyTorch 모델
        cfg: merge된 OmegaConf 설정

    Usage:
        trainer = Trainer(model, cfg)
        results = trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger(__name__)

        # 재현성 보장
        set_seed(cfg.project.seed)

        # 디바이스 설정
        device_str = cfg.project.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() and device_str == "cuda":
            self.logger.warning("CUDA 사용 불가 — CPU로 학습합니다.")

        self.model = model.to(self.device)

        # pos_weight 계산: train split 기준 negative/positive 비율 ≈ 6.78
        # cfg.loss.pos_weight가 null이면 state.yaml에서 자동 계산
        pos_weight_val = cfg.loss.get("pos_weight", None)
        if pos_weight_val is None:
            # state.yaml의 train split 통계에서 자동 계산
            try:
                from src.utils.state import load_state
                state = load_state()
                n_neg = state.dataset.splits.train.n_normal
                n_pos = state.dataset.splits.train.n_defect
                pos_weight_val = n_neg / n_pos
                self.logger.info(f"pos_weight 자동 계산: {n_neg}/{n_pos} = {pos_weight_val:.4f}")
            except Exception:
                pos_weight_val = 6.78  # 기본값 (244/36)
                self.logger.warning(f"state.yaml 로드 실패 — 기본 pos_weight {pos_weight_val} 사용")

        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        # 손실 함수 선택: cfg.loss.type이 "focal"이면 FocalLoss, 그 외 BCEWithLogitsLoss
        loss_type = cfg.loss.get("type", "bce")
        if loss_type == "focal":
            # FocalLoss: 어려운 샘플(hard example)에 집중하는 손실함수
            # alpha로 클래스 불균형 보정, gamma로 easy example 억제 강도 조절
            focal_alpha = cfg.loss.get("focal_alpha", 0.25)
            focal_gamma = cfg.loss.get("focal_gamma", 2.0)
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.logger.info(
                f"손실 함수: FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})"
            )
        else:
            # BCEWithLogitsLoss: sigmoid + BCE를 합친 수치 안정적 손실함수.
            # pos_weight로 클래스 불균형 보정 — 결함(소수 클래스) 누락 페널티를 높임
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.logger.info(f"손실 함수: BCEWithLogitsLoss (pos_weight={pos_weight_val:.4f})")

        # weight_decay: L2 정규화 — 가중치가 너무 커지는 것을 막아 과적합 방지
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        # CosineAnnealingLR: 코사인 커브로 학습률을 서서히 0으로 감소 — 마지막 수렴을 안정적으로
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
        )

        # 디렉토리 생성
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.log_dir = Path(cfg.training.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Early stopping 상태
        self.best_val_f1: float = -1.0
        self.patience_counter: int = 0
        self.patience: int = cfg.training.early_stopping_patience

    # ------------------------------------------------------------------
    # 학습 / 검증 1 에포크
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """1 에포크 학습 수행 후 평균 loss를 반환한다."""
        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.float().to(self.device)
            labels = labels.float().to(self.device).unsqueeze(1)  # (B,) → (B, 1)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        """1 에포크 검증 수행 후 (loss, metrics_dict)를 반환한다."""
        self.model.eval()
        total_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[float] = []

        for images, labels in loader:
            images = images.float().to(self.device)
            labels_device = labels.float().to(self.device).unsqueeze(1)

            logits = self.model(images)
            loss = self.criterion(logits, labels_device)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred_proba=np.array(all_probs),
            threshold=self.cfg.evaluation.threshold,
        )
        return avg_loss, metrics

    # ------------------------------------------------------------------
    # 메인 학습 루프
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """학습 루프를 실행하고 최종 결과 dict를 반환한다.

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader

        Returns:
            {
                "best_val_f1": float,
                "best_epoch": int,
                "history": [{epoch, train_loss, val_loss, val_f1, val_auc_roc}, ...],
            }
        """
        epochs = self.cfg.training.epochs
        frozen_epochs = self.cfg.training.get("frozen_epochs", 0)
        phase2_lr = self.cfg.training.get("phase2_lr", self.cfg.training.lr)
        history: list[dict] = []
        best_epoch = 0

        self.logger.info(f"학습 시작 — device={self.device}, epochs={epochs}, patience={self.patience}")
        if frozen_epochs > 0:
            self.logger.info(f"Phase 1: epoch 1~{frozen_epochs} — classifier only (lr={self.cfg.training.lr})")

        for epoch in range(1, epochs + 1):
            # Phase 전환: frozen_epochs 완료 후 전체 레이어 학습 재개
            if frozen_epochs > 0 and epoch == frozen_epochs + 1:
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                    self.logger.info(f"Epoch {epoch}: Phase 2 — full fine-tuning 시작 (lr={phase2_lr})")
                self.optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=phase2_lr,
                    weight_decay=self.cfg.training.weight_decay,
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs - frozen_epochs,
                )
                self.patience_counter = 0

            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._eval_epoch(val_loader)
            self.scheduler.step()

            val_f1 = val_metrics["f1"]
            val_auc = val_metrics["auc_roc"]

            # TensorBoard 기록
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metrics/val_f1", val_f1, epoch)
            self.writer.add_scalar("Metrics/val_auc_roc", val_auc, epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_f1": round(val_f1, 6),
                "val_auc_roc": round(val_auc, 6),
            })

            self.logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_F1={val_f1:.4f} | "
                f"val_AUC={val_auc:.4f}"
            )

            # Best checkpoint 저장
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best_val_f1.pth", epoch, val_metrics)
                self.logger.info(f"  -> Best model 저장 (val_F1={val_f1:.4f})")
            else:
                self.patience_counter += 1

            # Last checkpoint 매 에포크 저장
            self._save_checkpoint("last.pth", epoch, val_metrics)

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        self.writer.close()
        self.logger.info(f"학습 완료 — best_val_F1={self.best_val_f1:.4f} at epoch {best_epoch}")

        return {
            "best_val_f1": self.best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
        }

    # ------------------------------------------------------------------
    # 테스트 평가
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "test") -> dict:
        """best checkpoint를 로드하여 지정된 split을 평가한다.

        Args:
            loader: 평가할 DataLoader
            split_name: 로그에 표시할 split 이름

        Returns:
            compute_metrics() 반환 dict
        """
        best_ckpt = self.checkpoint_dir / "best_val_f1.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info(f"Best checkpoint 로드 (epoch={ckpt.get('epoch', '?')})")

        _, metrics = self._eval_epoch(loader)
        self.logger.info(
            f"[{split_name}] F1={metrics['f1']:.4f} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # 결과 저장
    # ------------------------------------------------------------------

    def save_results(
        self,
        val_metrics: dict,
        test_metrics: dict,
        output_path: str,
        extra: Optional[dict] = None,
    ) -> None:
        """val/test 메트릭을 results.json으로 저장한다.

        Args:
            val_metrics: 검증 split 메트릭 dict
            test_metrics: 테스트 split 메트릭 dict
            output_path: 저장 경로 (예: experiments/baseline/results.json)
            extra: 추가로 포함할 메타데이터 dict
        """
        # stage 값: cfg에서 읽기 (없으면 "unknown")
        stage_value = self.cfg.get("stage", "unknown")

        results = {
            "stage": str(stage_value),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "metrics": {
                "val": val_metrics,
                "test": test_metrics,
            },
        }
        if extra is not None:
            results.update(extra)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"결과 저장 완료: {output_file}")

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        """모델 상태와 메타데이터를 체크포인트 파일로 저장한다."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
