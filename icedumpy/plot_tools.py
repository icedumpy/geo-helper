import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_roc_curve(model, x, y, label, color='b-', ax=None):
    """
    Plot roc curve.

    Parameters
    ----------
    model: sklearn's model
        Model for predict x.
    x: numpy array (N, M)
        Input data to the model.
    y: numpy array (N,)
        Input label.
    label: str
        Plot label.
    color: str
        Line color.
    ax: matplotlib suplots ax (optional), default None
        Axis for plot.

    Examples
    --------
    >>> 

    Returns
    -------
    ax, y_predict_prob, fpr, tpr, thresholds, auc
    """
    if ax is None:
        fig, ax = plt.subplots()
    y_predict_prob = model.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predict_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, color, label=f"{label} (AUC = {auc:.4f})")
    ax.plot(fpr, fpr, "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    return ax, y_predict_prob, fpr, tpr, thresholds, auc

def set_roc_plot_template(ax):
    """
    Set template of ROC

    Parameters
    ----------
    ax: matplotlib ax
        from fig, ax = plt.subplot...

    Examples
    --------
    >>> 

    Returns
    -------
    ax
    """
    ax.set_xticks(np.arange(-0.05, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.grid()
    ax.legend(loc=4)
    
    return ax

def plot_vminmax(img, vminmax=(2, 98), ax=None):
    """
    Plot image with nanpercentile cut.

    Parameters
    ----------
    img: 2D-numpy array
        Image array.
    vminmax: tuple of int or float (optional), default (2, 98)
        Tuple of (min percent, max percent).
    ax: matplotlib suplots ax (optional), default None
        Axis for plot.
    
    Examples
    --------
    >>> plot_vminmax(img, vminmax=(2, 98))

    Returns
    -------
    matplotlib fig and ax
    """
    vmin, vmax = np.nanpercentile(img, vminmax)
    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(img, vmin=vmin, vmax=vmax)
        return fig, ax
    else:
        ax.imshow(img, vmin=vmin, vmax=vmax)
        return ax