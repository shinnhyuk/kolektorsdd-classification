"""임계값 최적화 모듈.

비용 행렬 기반 결정 임계값 탐색.
비용 가정: FN=5,000만원/건, FP=50만원/건 → fn_cost=100, fp_cost=1 (상대 비율)
"""

from typing import Literal

import numpy as np
from sklearn.metrics import roc_curve

from src.evaluation.metrics import compute_metrics, extract_tn_fp_fn_tp


class ThresholdOptimizer:
    """비용 행렬 기반 분류 임계값 최적화기.

    Args:
        fn_cost: FN(결함 미검출) 1건당 상대 비용 (기본값: 100)
        fp_cost: FP(오경보) 1건당 상대 비용 (기본값: 1)
    """

    def __init__(self, fn_cost: float = 100, fp_cost: float = 1) -> None:
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _build_threshold_candidates(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> np.ndarray:
        """ROC curve 후보점과 균일 격자를 병합한 임계값 배열을 반환한다.

        ROC curve 상의 후보점(sklearn)과 np.linspace(0.01, 0.99, 99)를
        합산한 뒤 중복 제거 및 오름차순 정렬한다.
        """
        _, _, roc_thresholds = roc_curve(y_true, y_pred_proba)
        # roc_curve는 최대값이 max(y_pred_proba)+eps일 수 있으므로 클리핑
        roc_thresholds = np.clip(roc_thresholds, 0.0, 1.0)

        linspace_thresholds = np.linspace(0.01, 0.99, 99)

        combined = np.unique(np.concatenate([roc_thresholds, linspace_thresholds]))
        return np.sort(combined)

    def _compute_cost(self, fn_count: int, fp_count: int) -> float:
        """총 비용(상대값)을 계산한다."""
        return float(fn_count * self.fn_cost + fp_count * self.fp_cost)

    def _evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float,
    ) -> dict:
        """단일 임계값에 대한 전체 지표와 비용을 계산한다."""
        metrics = compute_metrics(y_true, y_pred_proba, threshold=threshold)
        tn, fp, fn, tp = extract_tn_fp_fn_tp(metrics["confusion_matrix"])
        total_cost = self._compute_cost(fn, fp)

        return {
            "threshold": float(threshold),
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "tp": int(tp),
            "total_cost": total_cost,
        }

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def find_optimal(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: Literal["cost_minimization", "youden", "f1_optimal"] = "cost_minimization",
    ) -> dict:
        """지정한 방법으로 최적 임계값을 탐색한다.

        Args:
            y_true: 정답 레이블 배열 (0 또는 1, shape: [N])
            y_pred_proba: 모델 출력 확률값 배열 (0~1, shape: [N])
            method: 최적화 방법
                - "cost_minimization": total_cost = fn*fn_cost + fp*fp_cost 최소화
                - "youden": Youden's J (sensitivity + specificity - 1) 최대화
                - "f1_optimal": F1-score 최대화

        Returns:
            {
                "threshold": float,
                "f1": float,
                "recall": float,
                "precision": float,
                "fn": int,
                "fp": int,
                "tn": int,
                "tp": int,
                "total_cost": float,
                "method": str,
            }
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        thresholds = self._build_threshold_candidates(y_true, y_pred_proba)

        if method == "cost_minimization":
            best_idx = self._argmin_cost(y_true, y_pred_proba, thresholds)
        elif method == "youden":
            best_idx = self._argmax_youden(y_true, y_pred_proba, thresholds)
        elif method == "f1_optimal":
            best_idx = self._argmax_f1(y_true, y_pred_proba, thresholds)
        else:
            raise ValueError(
                f"지원하지 않는 method: {method!r}. "
                "'cost_minimization' | 'youden' | 'f1_optimal' 중 하나를 선택하세요."
            )

        result = self._evaluate_threshold(y_true, y_pred_proba, thresholds[best_idx])
        result["method"] = method
        return result

    def sensitivity_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        fn_cost_range: list[float],
    ) -> dict:
        """fn_cost를 스윕하며 최적 임계값 변화를 분석한다.

        Args:
            y_true: 정답 레이블 배열 (0 또는 1)
            y_pred_proba: 모델 출력 확률값 배열 (0~1)
            fn_cost_range: 탐색할 fn_cost 값 목록 (fp_cost=1 고정)

        Returns:
            {
                "fn_cost_range": list[float],
                "optimal_thresholds": list[float],
                "min_costs": list[float],
                "f1_at_optimal": list[float],
                "recall_at_optimal": list[float],
            }
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        thresholds = self._build_threshold_candidates(y_true, y_pred_proba)

        optimal_thresholds: list[float] = []
        min_costs: list[float] = []
        f1_list: list[float] = []
        recall_list: list[float] = []

        for fn_cost_val in fn_cost_range:
            # 임시로 fn_cost를 교체하여 탐색
            original_fn_cost = self.fn_cost
            self.fn_cost = float(fn_cost_val)

            best_idx = self._argmin_cost(y_true, y_pred_proba, thresholds)
            result = self._evaluate_threshold(y_true, y_pred_proba, thresholds[best_idx])

            self.fn_cost = original_fn_cost

            optimal_thresholds.append(result["threshold"])
            min_costs.append(result["total_cost"])
            f1_list.append(result["f1"])
            recall_list.append(result["recall"])

        return {
            "fn_cost_range": list(fn_cost_range),
            "optimal_thresholds": optimal_thresholds,
            "min_costs": min_costs,
            "f1_at_optimal": f1_list,
            "recall_at_optimal": recall_list,
        }

    def cost_reduction_pct(self, baseline_cost: float, optimal_cost: float) -> float:
        """baseline 대비 optimal의 비용 절감율(%)을 반환한다.

        Args:
            baseline_cost: 기준 임계값(0.5)에서의 총 비용
            optimal_cost: 최적 임계값에서의 총 비용

        Returns:
            절감율(%). 양수면 비용 절감, 음수면 비용 증가.
        """
        if baseline_cost == 0:
            return 0.0
        return float((baseline_cost - optimal_cost) / baseline_cost * 100)

    # ------------------------------------------------------------------
    # 내부 최적화 로직
    # ------------------------------------------------------------------

    def _argmin_cost(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray,
    ) -> int:
        """cost_minimization 방법으로 최적 threshold 인덱스를 반환한다."""
        costs = np.array([
            self._compute_cost(
                fn=int(np.sum((y_pred_proba < t) & (y_true == 1))),
                fp=int(np.sum((y_pred_proba >= t) & (y_true == 0))),
            )
            for t in thresholds
        ])
        return int(np.argmin(costs))

    def _argmax_youden(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray,
    ) -> int:
        """Youden's J (sensitivity + specificity - 1) 최대화 인덱스를 반환한다."""
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0

        youden_j = np.array([
            (
                np.sum((y_pred_proba >= t) & (y_true == 1)) / n_pos  # sensitivity (TPR)
                + np.sum((y_pred_proba < t) & (y_true == 0)) / n_neg  # specificity (TNR)
                - 1.0
            )
            for t in thresholds
        ])
        return int(np.argmax(youden_j))

    def _argmax_f1(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray,
    ) -> int:
        """F1-score 최대화 인덱스를 반환한다."""
        f1_scores = np.array([
            compute_metrics(y_true, y_pred_proba, threshold=float(t))["f1"]
            for t in thresholds
        ])
        return int(np.argmax(f1_scores))
