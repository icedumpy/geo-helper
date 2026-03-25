import numpy as np
import pytest


def test_plot_roc_curve_returns_ax():
    """plot_roc_curve should return a matplotlib Axes."""
    plt = pytest.importorskip("matplotlib.pyplot")
    sklearn = pytest.importorskip("sklearn")
    from geo_helper.plot import plot_roc_curve

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    ax = plot_roc_curve(y_true, y_score, label="test")
    assert ax is not None
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    plt.close("all")


def test_plot_roc_curve_with_existing_ax():
    """Should plot on a provided axes."""
    plt = pytest.importorskip("matplotlib.pyplot")
    sklearn = pytest.importorskip("sklearn")
    from geo_helper.plot import plot_roc_curve

    fig, ax = plt.subplots()
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    returned_ax = plot_roc_curve(y_true, y_score, ax=ax, label="test")
    assert returned_ax is ax
    plt.close("all")


def test_plot_precision_recall_curve_returns_ax():
    """plot_precision_recall_curve should return a matplotlib Axes."""
    plt = pytest.importorskip("matplotlib.pyplot")
    sklearn = pytest.importorskip("sklearn")
    from geo_helper.plot import plot_precision_recall_curve

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    ax = plot_precision_recall_curve(y_true, y_score, label="test")
    assert ax is not None
    plt.close("all")


def test_plot_raster_2d():
    """plot_raster should handle a 2D array."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from geo_helper.plot import plot_raster

    arr = np.random.rand(10, 10)
    ax = plot_raster(arr)
    assert ax is not None
    plt.close("all")


def test_plot_raster_percentile_clip():
    """Percentile clipping should handle arrays with outliers."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from geo_helper.plot import plot_raster

    arr = np.zeros((10, 10))
    arr[0, 0] = 1000
    ax = plot_raster(arr, percentile_clip=(2, 98))
    assert ax is not None
    plt.close("all")


def test_plot_raster_no_clip():
    """plot_raster with percentile_clip=False should still work."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from geo_helper.plot import plot_raster

    arr = np.random.rand(10, 10)
    ax = plot_raster(arr, percentile_clip=False)
    assert ax is not None
    plt.close("all")
