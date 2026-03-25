"""Visualization utilities for rasters and ML evaluation."""

import numpy as np


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    ax=None,
    label: str | None = None,
    **kwargs,
):
    """Plot a ROC curve with AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores / probabilities for the positive class.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Created if None.
    label : str | None
        Legend label prefix.
    **kwargs
        Passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_roc_curve(). "
            "Install with: pip install geo-helper[plot]"
        )
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        raise ImportError(
            "scikit-learn is required for plot_roc_curve(). "
            "Install with: pip install geo-helper[plot]"
        )

    if ax is None:
        _, ax = plt.subplots()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plot_label = f"{label} (AUC = {auc:.4f})" if label else f"AUC = {auc:.4f}"
    ax.plot(fpr, tpr, label=plot_label, **kwargs)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    ax=None,
    label: str | None = None,
    **kwargs,
):
    """Plot a Precision-Recall curve with average precision.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores / probabilities for the positive class.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Created if None.
    label : str | None
        Legend label prefix.
    **kwargs
        Passed to ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_precision_recall_curve(). "
            "Install with: pip install geo-helper[plot]"
        )
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError:
        raise ImportError(
            "scikit-learn is required for plot_precision_recall_curve(). "
            "Install with: pip install geo-helper[plot]"
        )

    if ax is None:
        _, ax = plt.subplots()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plot_label = f"{label} (AP = {ap:.4f})" if label else f"AP = {ap:.4f}"
    ax.plot(recall, precision, label=plot_label, **kwargs)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    return ax


def plot_raster(
    array: np.ndarray,
    *,
    percentile_clip: bool | tuple[float, float] = True,
    ax=None,
    **kwargs,
):
    """Plot a 2D raster array with optional percentile-based scaling.

    Parameters
    ----------
    array : np.ndarray
        2D array to visualize.
    percentile_clip : bool | tuple[float, float]
        If True, clips to (2, 98) percentile. If a tuple, uses those
        percentiles. If False, no clipping.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Created if None.
    **kwargs
        Passed to ax.imshow().

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_raster(). "
            "Install with: pip install geo-helper[plot]"
        )

    if ax is None:
        _, ax = plt.subplots()

    if percentile_clip is not False:
        if percentile_clip is True:
            pmin, pmax = 2, 98
        else:
            pmin, pmax = percentile_clip
        vmin, vmax = np.nanpercentile(array, [pmin, pmax])
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

    ax.imshow(array, **kwargs)
    return ax
