"""Evaluation metrics for classification tasks."""

import numpy as np


def kappa(confusion_matrix: np.ndarray) -> float:
    """Calculate Cohen's Kappa coefficient from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square confusion matrix (N x N).

    Returns
    -------
    float
        Kappa coefficient in range [-1, 1].
    """
    n = confusion_matrix.sum()
    observed = np.diag(confusion_matrix).sum()
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    expected = np.dot(row_sums, col_sums)
    return (n * observed - expected) / (n ** 2 - expected)


def fuzzy_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute a fuzzy confusion matrix for continuous predictions.

    Each sample contributes fractionally to the matrix cells based on
    its continuous membership values (e.g., loss ratios between 0 and 1).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth continuous values in [0, 1].
    y_pred : np.ndarray
        Predicted continuous values in [0, 1].
    thresholds : tuple[float, float] | None
        If provided, (low, high) thresholds to binarize values before
        computing the matrix. Values >= high become 1, values <= low
        become 0, values in between are kept as-is.

    Returns
    -------
    np.ndarray
        2x2 fuzzy confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if thresholds is not None:
        low, high = thresholds
        for arr in (y_true, y_pred):
            arr[arr >= high] = 1.0
            arr[arr <= low] = 0.0

    result = np.zeros((2, 2), dtype=np.float64)
    result[1, 1] = np.minimum(y_true, y_pred).sum()              # TP: actual pos, pred pos
    result[0, 0] = np.minimum(1 - y_true, 1 - y_pred).sum()     # TN: actual neg, pred neg
    result[1, 0] = np.maximum(0.0, y_true - y_pred).sum()        # FN: actual pos, pred neg
    result[0, 1] = np.maximum(0.0, y_pred - y_true).sum()        # FP: actual neg, pred pos
    return result
