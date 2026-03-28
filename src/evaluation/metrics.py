"""분류 메트릭 계산 모듈.

주지표: F1-score, AUC-ROC (정확도는 불균형 데이터에서 무의미하므로 사용하지 않음).
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """이진분류 메트릭 전체를 계산하여 dict로 반환한다.

    Args:
        y_true: 정답 레이블 배열 (0 또는 1, shape: [N])
        y_pred_proba: 모델 출력 확률값 배열 (0~1, shape: [N])
                      sigmoid를 적용한 값이어야 함
        threshold: 이진 분류 임계값 (기본값: 0.5)

    Returns:
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
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # F1-score (결함(=1) 클래스 기준)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # AUC-ROC — 클래스 불균형에서도 의미 있는 지표
    auc_roc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0

    # Precision, Recall (결함 클래스 기준)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)

    # Average Precision (PR-AUC)
    ap = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred_binary).tolist()

    return {
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "precision": float(precision),
        "recall": float(recall),
        "average_precision": float(ap),
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }


def extract_tn_fp_fn_tp(confusion_mat: list[list[int]]) -> tuple[int, int, int, int]:
    """confusion matrix에서 TN, FP, FN, TP를 추출한다.

    Args:
        confusion_mat: [[TN, FP], [FN, TP]] 형태

    Returns:
        (TN, FP, FN, TP) 튜플
    """
    tn = confusion_mat[0][0]
    fp = confusion_mat[0][1]
    fn = confusion_mat[1][0]
    tp = confusion_mat[1][1]
    return tn, fp, fn, tp
