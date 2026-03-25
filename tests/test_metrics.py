import numpy as np
import pytest
from geo_helper.metrics import kappa, fuzzy_confusion_matrix


def test_kappa_perfect_agreement():
    """Perfect diagonal confusion matrix should give kappa = 1.0."""
    cnf = np.array([[50, 0], [0, 50]])
    assert kappa(cnf) == pytest.approx(1.0)


def test_kappa_no_agreement():
    """Off-diagonal confusion matrix should give kappa <= 0."""
    cnf = np.array([[0, 50], [50, 0]])
    assert kappa(cnf) <= 0.0


def test_kappa_partial_agreement():
    """Known confusion matrix with expected kappa."""
    cnf = np.array([[40, 10], [5, 45]])
    k = kappa(cnf)
    assert 0.5 < k < 1.0


def test_fuzzy_confusion_matrix_perfect():
    """When predicted == reference, diagonal should dominate."""
    y_true = np.array([1.0, 0.0, 0.8, 0.2])
    y_pred = np.array([1.0, 0.0, 0.8, 0.2])
    result = fuzzy_confusion_matrix(y_true, y_pred)
    assert result.shape == (2, 2)
    assert result[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert result[1, 0] == pytest.approx(0.0, abs=1e-6)


def test_fuzzy_confusion_matrix_opposite():
    """When predicted is opposite of reference."""
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.0, 1.0])
    result = fuzzy_confusion_matrix(y_true, y_pred)
    assert result[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert result[1, 1] == pytest.approx(0.0, abs=1e-6)


def test_fuzzy_confusion_matrix_returns_numpy():
    """Result should be a numpy array, not a DataFrame."""
    y_true = np.array([0.5, 0.5])
    y_pred = np.array([0.5, 0.5])
    result = fuzzy_confusion_matrix(y_true, y_pred)
    assert isinstance(result, np.ndarray)


def test_fuzzy_confusion_matrix_with_thresholds():
    """Thresholds should binarize boundary values."""
    y_true = np.array([0.1, 0.9, 0.5])
    y_pred = np.array([0.1, 0.9, 0.5])
    result = fuzzy_confusion_matrix(y_true, y_pred, thresholds=(0.2, 0.8))
    assert result.shape == (2, 2)
    assert result.sum() == pytest.approx(len(y_true))
