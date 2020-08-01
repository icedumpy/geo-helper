import numpy as np
from sklearn import metrics

def plot_roc_curve(ax, model, x, y, label, color='b-'):
    """
    Plot roc curve.

    Parameters
    ----------
    ax: matplotlib ax
        from fig, ax = plt.subplot...
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

    Examples
    --------
    >>> 

    Returns
    -------
    ax, y_predict_prob, fpr, tpr, thresholds, auc
    """
    y_predict_prob = model.predict_proba(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predict_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, color, label=f"{label} (AUC = {auc:.4f})")
    ax.plot(fpr, fpr, "--")
    ax.set_xlabel("False Alarm")
    ax.set_ylabel("Detection")

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
    ax.set_yticks(np.arange(0, 1., 0.1))
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.grid()
    ax.legend(loc=4)
    
    return ax