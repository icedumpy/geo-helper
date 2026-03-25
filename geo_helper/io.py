"""Model and data serialization utilities."""

from pathlib import Path


def save_model(model: object, path: str | Path) -> None:
    """Persist a scikit-learn model to disk via joblib.

    Parameters
    ----------
    model : object
        Trained sklearn model (or any picklable object).
    path : str | Path
        Save path. `.joblib` extension is added if missing.
    """
    try:
        from joblib import dump
    except ImportError:
        raise ImportError(
            "joblib is required for save_model(). "
            "Install with: pip install geo-helper[ml]"
        )
    path = Path(path)
    if path.suffix != ".joblib":
        path = path.with_suffix(".joblib")
    dump(model, path)


def load_model(path: str | Path) -> object:
    """Load a scikit-learn model from disk via joblib.

    Parameters
    ----------
    path : str | Path
        Path to the .joblib file.

    Returns
    -------
    object
        The loaded model.
    """
    try:
        from joblib import load
    except ImportError:
        raise ImportError(
            "joblib is required for load_model(). "
            "Install with: pip install geo-helper[ml]"
        )
    return load(path)


def save_h5(data: dict, path: str | Path) -> None:
    """Save a dict of arrays to an HDF5 file.

    Parameters
    ----------
    data : dict
        Dictionary mapping string keys to numpy arrays.
    path : str | Path
        Save path.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for save_h5(). "
            "Install with: pip install geo-helper[ml]"
        )
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def load_h5(path: str | Path) -> dict:
    """Load an HDF5 file into a dict of arrays.

    Parameters
    ----------
    path : str | Path
        Path to the .h5 file.

    Returns
    -------
    dict
        Dictionary mapping dataset names to numpy arrays.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for load_h5(). "
            "Install with: pip install geo-helper[ml]"
        )
    result = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            result[key] = f[key][()]
    return result
