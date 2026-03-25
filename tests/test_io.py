import numpy as np
import pytest


def test_save_load_model_roundtrip(tmp_path):
    """Save and load a simple sklearn model."""
    joblib = pytest.importorskip("joblib")
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from geo_helper.io import save_model, load_model

    model = LinearRegression()
    model.fit([[1], [2], [3]], [1, 2, 3])

    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.allclose(loaded.predict([[4]]), model.predict([[4]]))


def test_save_model_adds_extension(tmp_path):
    """save_model should add .joblib extension if missing."""
    joblib = pytest.importorskip("joblib")
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from geo_helper.io import save_model

    model = LinearRegression()
    model.fit([[1]], [1])

    path = tmp_path / "model"
    save_model(model, path)
    assert (tmp_path / "model.joblib").exists()


def test_save_load_h5_roundtrip(tmp_path):
    """Save and load dict of arrays via HDF5."""
    h5py = pytest.importorskip("h5py")
    from geo_helper.io import save_h5, load_h5

    data = {"a": np.array([1, 2, 3]), "b": np.array([[4, 5], [6, 7]])}
    path = tmp_path / "params.h5"
    save_h5(data, path)

    loaded = load_h5(path)
    assert set(loaded.keys()) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], data["a"])
    np.testing.assert_array_equal(loaded["b"], data["b"])


def test_io_functions_are_importable():
    """All io functions should be importable without triggering lazy deps."""
    from geo_helper.io import save_model, load_model, save_h5, load_h5
    assert callable(save_model)
    assert callable(load_model)
    assert callable(save_h5)
    assert callable(load_h5)
