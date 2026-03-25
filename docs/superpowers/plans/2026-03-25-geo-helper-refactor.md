# geo-helper Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor geo-helper from a PIER-specific utility bag into a clean, general-purpose Python geospatial helper package (v1.0.0).

**Architecture:** Layered modules with a shared `core.py` for geospatial primitives. `raster.py` and `vector.py` depend on `core.py`; `io.py`, `plot.py`, and `metrics.py` are standalone. Lazy imports for all optional dependencies.

**Tech Stack:** Python >=3.12, numpy, rasterio, pyproj, shapely, geopandas, GDAL, matplotlib, scikit-learn, joblib, h5py, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-geo-helper-refactor-design.md`

---

## Parallelization Map

```
Task 1 (scaffolding) ──┬── Task 2 (core.py) ──┬── Task 5 (raster.py)
                        │                       └── Task 6 (vector.py)
                        ├── Task 3 (metrics.py)
                        ├── Task 4 (io.py)
                        ├── Task 7 (plot.py)
                        └── Task 8 (cleanup + __init__.py)
```

- **Task 1** must complete first (creates directory structure and pyproject.toml)
- **Tasks 2, 3, 4, 7** can run in parallel after Task 1
- **Tasks 5, 6** depend on Task 2 (core.py) and can run in parallel with each other
- **Task 8** runs last (deletes old files, wires up __init__.py)

---

### Task 1: Scaffolding — pyproject.toml and directory structure

**Files:**
- Create: `pyproject.toml`
- Create: `geo_helper/__init__.py` (empty placeholder)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create the geo_helper package directory and tests directory**

```bash
mkdir -p geo_helper tests
```

- [ ] **Step 2: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "geo-helper"
version = "1.0.0"
description = "General-purpose Python geospatial helper utilities"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "Pongporamat", email = "pongporamat.c@gmail.com" }]
dependencies = ["numpy", "rasterio"]

[project.optional-dependencies]
geo = ["geopandas", "shapely", "pyproj", "GDAL"]
ml = ["scikit-learn", "joblib", "h5py"]
plot = ["matplotlib", "scikit-learn"]
all = ["geo-helper[geo]", "geo-helper[ml]", "geo-helper[plot]"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create placeholder __init__.py files**

Create `geo_helper/__init__.py`:
```python
"""geo-helper: General-purpose geospatial helper utilities."""
```

Create `tests/__init__.py`:
```python
```

- [ ] **Step 4: Verify the package is installable**

```bash
pip install -e ".[all]" 2>&1 | tail -5
python -c "import geo_helper; print('OK')"
```

Expected: installs successfully, prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml geo_helper/__init__.py tests/__init__.py
git commit -m "feat: add pyproject.toml and package scaffolding for v1.0.0"
```

---

### Task 2: core.py — Shared Geospatial Primitives

**Files:**
- Create: `geo_helper/core.py`
- Create: `tests/test_core.py`
- Reference: `helper/geo_tools.py:70-104` (get_pixel_location_from_coords)
- Reference: `helper/geo_tools.py:27-68` (create_polygon_from_wkt — for transform_coords logic)

- [ ] **Step 1: Write failing tests for core.py**

Create `tests/test_core.py`:

```python
import numpy as np
import pytest
from geo_helper.core import get_pixel_from_coords, get_coords_from_pixel


def test_get_pixel_from_coords_identity():
    """Origin pixel should be (0, 0) when coords match the transform origin."""
    # Affine transform: origin at (100.0, 15.0), pixel size 0.01 x -0.01
    transform = (100.0, 0.01, 0.0, 15.0, 0.0, -0.01)
    col, row = get_pixel_from_coords(100.0, 15.0, transform)
    assert col == 0
    assert row == 0


def test_get_pixel_from_coords_offset():
    """Coords offset by 5 pixels in each direction."""
    transform = (100.0, 0.01, 0.0, 15.0, 0.0, -0.01)
    col, row = get_pixel_from_coords(100.05, 14.95, transform)
    assert col == 5
    assert row == 5


def test_get_coords_from_pixel_roundtrip():
    """Converting pixel->coords->pixel should return the same pixel."""
    transform = (100.0, 0.01, 0.0, 15.0, 0.0, -0.01)
    # Get coords for pixel (3, 7)
    x, y = get_coords_from_pixel(7, 3, transform)
    # Convert back
    col, row = get_pixel_from_coords(x, y, transform)
    assert col == 3
    assert row == 7


def test_get_coords_from_pixel_values():
    """Check actual coordinate values for a known pixel."""
    transform = (100.0, 0.01, 0.0, 15.0, 0.0, -0.01)
    x, y = get_coords_from_pixel(0, 0, transform)
    # Top-left corner of pixel (0,0) = transform origin
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(15.0)


def test_transform_coords():
    """Test coordinate reprojection from EPSG:4326 to EPSG:32647."""
    pyproj = pytest.importorskip("pyproj")
    from geo_helper.core import transform_coords

    # Bangkok is roughly at (100.5, 13.75) in EPSG:4326
    x, y = transform_coords(100.5, 13.75, "EPSG:4326", "EPSG:32647")
    # UTM47N coordinates should be roughly 670,000 E and 1,520,000 N
    assert 600_000 < x < 800_000
    assert 1_400_000 < y < 1_600_000


def test_transform_coords_identity():
    """Same source and dest CRS should return ~same coordinates."""
    pyproj = pytest.importorskip("pyproj")
    from geo_helper.core import transform_coords

    x, y = transform_coords(100.5, 13.75, "EPSG:4326", "EPSG:4326")
    assert x == pytest.approx(100.5, abs=1e-6)
    assert y == pytest.approx(13.75, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_core.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'geo_helper.core'`

- [ ] **Step 3: Implement core.py**

Create `geo_helper/core.py`:

```python
"""Shared geospatial primitives used by raster and vector modules."""

import numpy as np


def transform_coords(
    x: float, y: float, src_crs: str, dst_crs: str
) -> tuple[float, float]:
    """Reproject coordinates from one CRS to another.

    Parameters
    ----------
    x : float
        Longitude or Easting.
    y : float
        Latitude or Northing.
    src_crs : str
        Source CRS string (e.g. "EPSG:4326").
    dst_crs : str
        Destination CRS string (e.g. "EPSG:32647").

    Returns
    -------
    tuple[float, float]
        Transformed (x, y) coordinates.
    """
    try:
        import pyproj
    except ImportError:
        raise ImportError(
            "pyproj is required for transform_coords(). "
            "Install with: pip install geo-helper[geo]"
        )
    transformer = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    )
    return transformer.transform(x, y)


def get_pixel_from_coords(
    x: float, y: float, transform: tuple
) -> tuple[int, int]:
    """Map geographic coordinates to raster pixel position.

    Parameters
    ----------
    x : float
        Longitude or Easting.
    y : float
        Latitude or Northing.
    transform : tuple
        GDAL-style geotransform (origin_x, pixel_width, row_rot,
        origin_y, col_rot, pixel_height).

    Returns
    -------
    tuple[int, int]
        (col, row) pixel indices.
    """
    origin_x, pixel_width, row_rot, origin_y, col_rot, pixel_height = transform
    denom = (pixel_width * pixel_height) - (row_rot * col_rot)
    col = int(np.floor(
        (pixel_height * (x - origin_x) - row_rot * (y - origin_y)) / denom
    ))
    row = int(np.floor(
        (pixel_width * (y - origin_y) - col_rot * (x - origin_x)) / denom
    ))
    return col, row


def get_coords_from_pixel(
    row: int, col: int, transform: tuple
) -> tuple[float, float]:
    """Map raster pixel position to geographic coordinates.

    Returns the coordinate of the top-left corner of the pixel.

    Parameters
    ----------
    row : int
        Row index.
    col : int
        Column index.
    transform : tuple
        GDAL-style geotransform.

    Returns
    -------
    tuple[float, float]
        (x, y) geographic coordinates.
    """
    origin_x, pixel_width, row_rot, origin_y, col_rot, pixel_height = transform
    x = origin_x + col * pixel_width + row * row_rot
    y = origin_y + col * col_rot + row * pixel_height
    return x, y
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_core.py -v
```

Expected: all 6 tests PASS (transform_coords tests skip if pyproj not installed).

- [ ] **Step 5: Commit**

```bash
git add geo_helper/core.py tests/test_core.py
git commit -m "feat: add core.py with coordinate transforms and pixel mapping"
```

---

### Task 3: metrics.py — Evaluation Metrics (Standalone)

**Files:**
- Create: `geo_helper/metrics.py`
- Create: `tests/test_metrics.py`
- Reference: `helper/eval_tools.py` (both functions)

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:

```python
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
    # Diagonal sum should equal total since pred == ref
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
    # 0.1 -> 0.0, 0.9 -> 1.0, 0.5 stays 0.5
    assert result.shape == (2, 2)
    assert result.sum() == pytest.approx(len(y_true))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics.py**

Create `geo_helper/metrics.py`:

```python
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
        2x2 fuzzy confusion matrix where:
        [0,0] = sum(min(pred, true))       — both positive
        [0,1] = sum(min(pred, 1-true))     — pred positive, true negative
        [1,0] = sum(min(1-pred, true))     — pred negative, true positive
        [1,1] = sum(min(1-pred, 1-true))   — both negative
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if thresholds is not None:
        low, high = thresholds
        for arr in (y_true, y_pred):
            arr[arr >= high] = 1.0
            arr[arr <= low] = 0.0

    result = np.zeros((2, 2), dtype=np.float64)
    result[0, 0] = np.minimum(y_pred, y_true).sum()
    result[0, 1] = np.minimum(y_pred, 1 - y_true).sum()
    result[1, 0] = np.minimum(1 - y_pred, y_true).sum()
    result[1, 1] = np.minimum(1 - y_pred, 1 - y_true).sum()
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add geo_helper/metrics.py tests/test_metrics.py
git commit -m "feat: add metrics.py with kappa and fuzzy confusion matrix"
```

---

### Task 4: io.py — Model & Data Serialization (Standalone)

**Files:**
- Create: `geo_helper/io.py`
- Create: `tests/test_io.py`
- Reference: `helper/io_tools.py` (all functions)

- [ ] **Step 1: Write failing tests**

Create `tests/test_io.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_io.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement io.py**

Create `geo_helper/io.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_io.py -v
```

Expected: all 4 tests PASS (tests skip if sklearn/h5py not installed).

- [ ] **Step 5: Commit**

```bash
git add geo_helper/io.py tests/test_io.py
git commit -m "feat: add io.py with model and HDF5 serialization"
```

---

### Task 5: raster.py — Raster File Operations

**Files:**
- Create: `geo_helper/raster.py`
- Create: `tests/test_raster.py`
- Reference: `helper/geo_tools.py:107-187` (create_tiff)
- Reference: `helper/geo_tools.py:189-222` (create_vrt)
- Reference: `helper/geo_tools.py:224-253` (get_raster_date)
- Reference: `helper/geo_tools.py:255-273` (gdal_descriptions)
- Reference: `helper/geo_tools.py:313-342` (gdal_warp)
- Reference: `helper/geo_tools.py:344-417` (gdal_rasterize)
- Depends on: `geo_helper/core.py` (Task 2)

- [ ] **Step 1: Write failing tests**

Create `tests/test_raster.py`:

```python
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture
def sample_tiff(tmp_path):
    """Create a small sample GeoTIFF for testing."""
    path = tmp_path / "sample.tif"
    data = np.random.rand(3, 10, 10).astype(np.float32)
    transform = from_bounds(100.0, 13.0, 101.0, 14.0, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10,
        count=3, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)
        dst.set_band_description(1, "2023-01-15")
        dst.set_band_description(2, "2023-02-15")
        dst.set_band_description(3, "2023-03-15")
    return path, data, transform


def test_create_tiff_2d(tmp_path):
    from geo_helper.raster import create_tiff

    path = tmp_path / "out.tif"
    arr = np.ones((10, 10), dtype=np.float32)
    transform = from_bounds(100.0, 13.0, 101.0, 14.0, 10, 10)

    create_tiff(path, arr, transform, "EPSG:4326")

    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.width == 10
        assert src.height == 10
        np.testing.assert_array_equal(src.read(1), arr)


def test_create_tiff_3d(tmp_path):
    from geo_helper.raster import create_tiff

    path = tmp_path / "out.tif"
    arr = np.ones((3, 10, 10), dtype=np.float32)
    transform = from_bounds(100.0, 13.0, 101.0, 14.0, 10, 10)

    create_tiff(path, arr, transform, "EPSG:4326", band_names=["a", "b", "c"])

    with rasterio.open(path) as src:
        assert src.count == 3
        assert src.descriptions == ("a", "b", "c")


def test_create_tiff_nodata(tmp_path):
    from geo_helper.raster import create_tiff

    path = tmp_path / "out.tif"
    arr = np.ones((10, 10), dtype=np.float32)
    transform = from_bounds(100.0, 13.0, 101.0, 14.0, 10, 10)

    create_tiff(path, arr, transform, "EPSG:4326", nodata=-9999)

    with rasterio.open(path) as src:
        assert src.nodata == -9999


def test_get_band_descriptions(sample_tiff):
    from geo_helper.raster import get_band_descriptions

    path, _, _ = sample_tiff
    descriptions = get_band_descriptions(path)
    assert descriptions == ("2023-01-15", "2023-02-15", "2023-03-15")


def test_get_band_dates_with_format(sample_tiff):
    from geo_helper.raster import get_band_dates

    path, _, _ = sample_tiff
    dates = get_band_dates(path, date_format="%Y-%m-%d")
    assert len(dates) == 3
    assert dates[0].year == 2023
    assert dates[0].month == 1


def test_get_band_dates_without_format(sample_tiff):
    from geo_helper.raster import get_band_dates

    path, _, _ = sample_tiff
    raw = get_band_dates(path)
    assert raw == ["2023-01-15", "2023-02-15", "2023-03-15"]


def test_create_vrt(sample_tiff, tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from geo_helper.raster import create_vrt

    tiff_path, _, _ = sample_tiff
    vrt_path = tmp_path / "out.vrt"
    create_vrt(vrt_path, [str(tiff_path), str(tiff_path)])
    assert vrt_path.exists()


def test_warp_reproject(sample_tiff, tmp_path):
    from geo_helper.raster import warp

    src_path, _, _ = sample_tiff
    dst_path = tmp_path / "warped.tif"

    warp(src_path, dst_path, dst_crs="EPSG:4326")

    with rasterio.open(dst_path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.count == 3


def test_rasterize_simple(tmp_path):
    """Rasterize a simple GeoJSON polygon."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    import json

    # Create a simple GeoJSON file
    geojson_path = tmp_path / "test.geojson"
    geom = box(100.0, 13.0, 101.0, 14.0)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    gdf.to_file(geojson_path, driver="GeoJSON")

    from geo_helper.raster import rasterize

    output_path = tmp_path / "rasterized.tif"
    rasterize(geojson_path, output_path, resolution=0.1, burn_value=1)

    with rasterio.open(output_path) as src:
        data = src.read(1)
        assert src.crs is not None
        assert data.shape[0] == 10  # (14-13)/0.1
        assert data.shape[1] == 10  # (101-100)/0.1
        assert data.max() == 1
        assert src.nodata == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_raster.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement raster.py**

Create `geo_helper/raster.py`:

```python
"""Raster file I/O and operations."""

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


def create_tiff(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: str,
    *,
    band_names: list[str] | None = None,
    nodata: float | None = 0,
    dtype: str | None = None,
) -> None:
    """Write a georeferenced raster from a numpy array.

    Parameters
    ----------
    path : str | Path
        Output file path.
    array : np.ndarray
        2D (height, width) or 3D (bands, height, width) array.
    transform : Affine
        Rasterio affine transform.
    crs : str
        Coordinate reference system (e.g. "EPSG:4326").
    band_names : list[str] | None
        Optional description for each band.
    nodata : float | None
        Nodata value. Set to None to omit.
    dtype : str | None
        Output dtype. Defaults to the array's dtype.
    """
    if array.ndim == 2:
        array = array[np.newaxis, :, :]

    count, height, width = array.shape
    dtype = dtype or str(array.dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array)
        if band_names is not None:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)


def create_vrt(
    output_path: str | Path,
    input_paths: list[str | Path],
    *,
    band_names: list[str] | None = None,
    src_nodata: float | None = None,
    dst_nodata: float | None = None,
) -> None:
    """Create a virtual raster (VRT) from multiple source rasters.

    Requires GDAL.

    Parameters
    ----------
    output_path : str | Path
        Output VRT file path.
    input_paths : list[str | Path]
        List of input raster file paths.
    band_names : list[str] | None
        Optional description for each band.
    src_nodata : float | None
        Source nodata value.
    dst_nodata : float | None
        Destination nodata value.
    """
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError(
            "GDAL is required for create_vrt(). "
            "Install with: pip install geo-helper[geo]"
        )
    options = gdal.BuildVRTOptions(
        separate=True,
        srcNodata=src_nodata,
        VRTNodata=dst_nodata,
    )
    input_strs = [str(p) for p in input_paths]
    ds = gdal.BuildVRT(str(output_path), input_strs, options=options)
    if band_names is not None:
        for i, name in enumerate(band_names, start=1):
            ds.GetRasterBand(i).SetDescription(name)
    ds.FlushCache()
    del ds


def warp(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: str,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    resampling: str = "nearest",
) -> None:
    """Reproject and optionally clip a raster.

    Parameters
    ----------
    src_path : str | Path
        Input raster path.
    dst_path : str | Path
        Output raster path.
    dst_crs : str
        Target CRS (e.g. "EPSG:32647").
    bounds : tuple | None
        Optional output bounds (left, bottom, right, top) in dst_crs.
    resolution : float | None
        Optional output resolution in dst_crs units.
    resampling : str
        Resampling method name (e.g. "nearest", "bilinear").
    """
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    resample_method = Resampling[resampling]

    with rasterio.open(src_path) as src:
        dst_transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )

        if bounds is not None:
            from rasterio.transform import from_bounds
            width = int(abs((bounds[2] - bounds[0]) / dst_transform.a))
            height = int(abs((bounds[3] - bounds[1]) / dst_transform.e))
            dst_transform = from_bounds(*bounds, width, height)

        kwargs = src.meta.copy()
        kwargs.update(
            crs=dst_crs,
            transform=dst_transform,
            width=width,
            height=height,
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resample_method,
                )


def rasterize(
    vector_path: str | Path,
    output_path: str | Path,
    resolution: float | tuple[float, float],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    crs: str | None = None,
    burn_value: int = 1,
    dtype: str = "uint8",
    nodata: int = 0,
) -> None:
    """Rasterize a vector file.

    Parameters
    ----------
    vector_path : str | Path
        Path to vector file (shapefile, GeoJSON, etc.).
    output_path : str | Path
        Output raster path.
    resolution : float | tuple[float, float]
        Pixel size. Single float for square pixels, or (x_res, y_res).
    bounds : tuple | None
        Output bounds (minx, miny, maxx, maxy). Defaults to vector bounds.
    crs : str | None
        Output CRS. Defaults to vector CRS.
    burn_value : int
        Value to burn into rasterized pixels.
    dtype : str
        Output numpy dtype string.
    nodata : int
        Nodata value.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for rasterize(). "
            "Install with: pip install geo-helper[geo]"
        )
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds

    gdf = gpd.read_file(vector_path)
    if crs is not None:
        gdf = gdf.to_crs(crs)
    out_crs = crs or str(gdf.crs)

    if bounds is None:
        bounds = tuple(gdf.total_bounds)
    minx, miny, maxx, maxy = bounds

    if isinstance(resolution, (int, float)):
        x_res = y_res = float(resolution)
    else:
        x_res, y_res = resolution

    width = max(1, int(np.ceil((maxx - minx) / x_res)))
    height = max(1, int(np.ceil((maxy - miny) / y_res)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None]
    arr = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=dtype,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=out_crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def get_band_descriptions(path: str | Path) -> tuple[str, ...]:
    """Read band description strings from a raster file.

    Parameters
    ----------
    path : str | Path
        Raster file path.

    Returns
    -------
    tuple[str, ...]
        Band descriptions.
    """
    with rasterio.open(path) as src:
        return src.descriptions


def get_band_dates(
    path: str | Path, date_format: str | None = None
) -> list[datetime] | list[str]:
    """Extract dates from raster band descriptions.

    Parameters
    ----------
    path : str | Path
        Raster file path.
    date_format : str | None
        strptime format string. If None, returns raw description strings.

    Returns
    -------
    list[datetime] | list[str]
        Parsed dates if date_format given, otherwise raw strings.
    """
    descriptions = get_band_descriptions(path)
    if date_format is None:
        return [d or "" for d in descriptions]
    return [datetime.strptime(d, date_format) for d in descriptions]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_raster.py -v
```

Expected: all tests PASS (create_vrt test skips if GDAL not installed).

- [ ] **Step 5: Commit**

```bash
git add geo_helper/raster.py tests/test_raster.py
git commit -m "feat: add raster.py with tiff/vrt creation, warp, rasterize"
```

---

### Task 6: vector.py — Vector/Geometry Operations

**Files:**
- Create: `geo_helper/vector.py`
- Create: `tests/test_vector.py`
- Reference: `helper/geo_tools.py:27-68` (create_polygon_from_wkt)
- Reference: `helper/geo_tools.py:276-311` (convert_to_geodataframe)
- Depends on: `geo_helper/core.py` (Task 2)

- [ ] **Step 1: Write failing tests**

Create `tests/test_vector.py`:

```python
import pytest


def test_from_wkt_simple():
    """Parse a simple WKT polygon."""
    shapely = pytest.importorskip("shapely")
    from geo_helper.vector import from_wkt

    wkt_str = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
    geom = from_wkt(wkt_str)
    assert geom.is_valid
    assert geom.area == pytest.approx(1.0)


def test_from_wkt_with_reprojection():
    """Parse WKT and reproject from EPSG:4326 to EPSG:32647."""
    shapely = pytest.importorskip("shapely")
    pyproj = pytest.importorskip("pyproj")
    from geo_helper.vector import from_wkt

    wkt_str = "POINT (100.5 13.75)"
    geom = from_wkt(wkt_str, src_crs="EPSG:4326", dst_crs="EPSG:32647")
    # UTM coordinates should be large numbers
    assert geom.x > 100_000
    assert geom.y > 1_000_000


def test_from_wkt_no_reprojection():
    """Without dst_crs, geometry should be unchanged."""
    shapely = pytest.importorskip("shapely")
    from geo_helper.vector import from_wkt

    wkt_str = "POINT (100.5 13.75)"
    geom = from_wkt(wkt_str)
    assert geom.x == pytest.approx(100.5)
    assert geom.y == pytest.approx(13.75)


def test_to_geodataframe():
    """Convert list of geometries to GeoDataFrame."""
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Point
    from geo_helper.vector import to_geodataframe

    geoms = [Point(0, 0), Point(1, 1), Point(2, 2)]
    gdf = to_geodataframe(geoms, crs="EPSG:4326")

    assert len(gdf) == 3
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf.geometry) == geoms


def test_to_geodataframe_no_crs():
    """GeoDataFrame without CRS should have crs=None."""
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Point
    from geo_helper.vector import to_geodataframe

    gdf = to_geodataframe([Point(0, 0)])
    assert gdf.crs is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_vector.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement vector.py**

Create `geo_helper/vector.py`:

```python
"""Vector and geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry


def from_wkt(
    wkt_string: str,
    src_crs: str | None = None,
    dst_crs: str | None = None,
) -> BaseGeometry:
    """Parse a WKT string into a Shapely geometry with optional reprojection.

    Parameters
    ----------
    wkt_string : str
        Well-Known Text geometry string.
    src_crs : str | None
        Source CRS (e.g. "EPSG:4326"). Only needed if dst_crs is set.
    dst_crs : str | None
        Target CRS. If provided, the geometry is reprojected from src_crs.

    Returns
    -------
    BaseGeometry
        Shapely geometry object.
    """
    try:
        from shapely import wkt
    except ImportError:
        raise ImportError(
            "shapely is required for from_wkt(). "
            "Install with: pip install geo-helper[geo]"
        )

    geom = wkt.loads(wkt_string)

    if dst_crs is not None:
        from shapely.ops import transform as shapely_transform
        from geo_helper.core import transform_coords

        src = src_crs or "EPSG:4326"

        def _reproject(x, y, z=None):
            return transform_coords(x, y, src, dst_crs)

        geom = shapely_transform(_reproject, geom)

    return geom


def to_geodataframe(
    geometries: list[BaseGeometry],
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Convert a list of Shapely geometry objects to a GeoDataFrame.

    Parameters
    ----------
    geometries : list[BaseGeometry]
        List of Shapely geometry objects.
    crs : str | None
        Coordinate reference system (e.g. "EPSG:4326").

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with a geometry column.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for to_geodataframe(). "
            "Install with: pip install geo-helper[geo]"
        )

    return gpd.GeoDataFrame(geometry=geometries, crs=crs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_vector.py -v
```

Expected: all 5 tests PASS (tests skip if shapely/geopandas/pyproj not installed).

- [ ] **Step 5: Commit**

```bash
git add geo_helper/vector.py tests/test_vector.py
git commit -m "feat: add vector.py with WKT parsing and GeoDataFrame creation"
```

---

### Task 7: plot.py — Visualization (Standalone)

**Files:**
- Create: `geo_helper/plot.py`
- Create: `tests/test_plot.py`
- Reference: `helper/plot_tools.py` (all functions)

- [ ] **Step 1: Write failing tests**

Create `tests/test_plot.py`:

```python
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
    # Check that it has axis labels (absorbed from set_roc_plot_template)
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
    arr[0, 0] = 1000  # outlier
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_plot.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement plot.py**

Create `geo_helper/plot.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_plot.py -v
```

Expected: all 6 tests PASS (skip if matplotlib/sklearn not installed).

- [ ] **Step 5: Commit**

```bash
git add geo_helper/plot.py tests/test_plot.py
git commit -m "feat: add plot.py with ROC, PR curves, and raster visualization"
```

---

### Task 8: Cleanup — Wire up __init__.py, delete old files, update CLAUDE.md

**Files:**
- Modify: `geo_helper/__init__.py`
- Delete: `setup.py`
- Delete: `test.py`
- Delete: `helper/` (entire directory)
- Modify: `CLAUDE.md`
- Depends on: all other tasks

- [ ] **Step 1: Write test for the lazy import __init__.py**

Add to `tests/test_core.py` (or create `tests/test_init.py`):

Create `tests/test_init.py`:

```python
def test_lazy_import_core():
    """Top-level imports should resolve to core functions."""
    from geo_helper import get_pixel_from_coords, get_coords_from_pixel
    assert callable(get_pixel_from_coords)
    assert callable(get_coords_from_pixel)


def test_lazy_import_raster():
    """Top-level imports should resolve to raster functions."""
    from geo_helper import create_tiff, warp
    assert callable(create_tiff)
    assert callable(warp)


def test_lazy_import_io():
    """Top-level imports should resolve to io functions."""
    from geo_helper import save_model, load_model
    assert callable(save_model)
    assert callable(load_model)


def test_invalid_import_raises():
    """Importing nonexistent name should raise AttributeError."""
    import pytest
    with pytest.raises(AttributeError, match="no attribute"):
        from geo_helper import nonexistent_function
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_init.py -v
```

Expected: FAIL — the placeholder __init__.py has no `__getattr__`.

- [ ] **Step 3: Implement the lazy __init__.py**

Write `geo_helper/__init__.py`:

```python
"""geo-helper: General-purpose geospatial helper utilities."""


def __getattr__(name: str):
    """Lazy import public API functions to avoid loading optional deps."""
    _exports = {
        "transform_coords": "geo_helper.core",
        "get_pixel_from_coords": "geo_helper.core",
        "get_coords_from_pixel": "geo_helper.core",
        "create_tiff": "geo_helper.raster",
        "create_vrt": "geo_helper.raster",
        "warp": "geo_helper.raster",
        "rasterize": "geo_helper.raster",
        "get_band_descriptions": "geo_helper.raster",
        "get_band_dates": "geo_helper.raster",
        "from_wkt": "geo_helper.vector",
        "to_geodataframe": "geo_helper.vector",
        "save_model": "geo_helper.io",
        "load_model": "geo_helper.io",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'geo_helper' has no attribute {name!r}")
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Delete old files**

```bash
rm -rf helper/ setup.py test.py
```

- [ ] **Step 6: Update CLAUDE.md**

Replace the content of `CLAUDE.md` with:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General-purpose Python geospatial helper package for personal reuse. Provides utilities for raster/vector operations, coordinate transforms, model serialization, evaluation metrics, and visualization.

## Commands

```bash
# Install in development mode (core only: numpy + rasterio)
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific groups
pip install -e ".[geo]"    # geopandas, shapely, pyproj, GDAL
pip install -e ".[ml]"     # scikit-learn, joblib, h5py
pip install -e ".[plot]"   # matplotlib, scikit-learn

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_core.py::test_transform_coords -v
```

## Architecture

Package: `geo_helper/` with layered modules and a shared `core.py`.

### Dependency Graph

```
core.py  (numpy; pyproj lazy-imported)
  ├── raster.py  (rasterio; osgeo/GDAL lazy for create_vrt only)
  └── vector.py  (shapely, geopandas)

io.py       (joblib, h5py)        — standalone
plot.py     (matplotlib, sklearn) — standalone
metrics.py  (numpy only)          — standalone
```

### Module Responsibilities

- **core.py** — Coordinate reprojection (`transform_coords`), pixel/coord mapping
- **raster.py** — GeoTIFF/VRT creation, warp, rasterize, band metadata
- **vector.py** — WKT parsing, GeoDataFrame creation (uses core for reprojection)
- **io.py** — Model serialization (joblib), HDF5 parameter storage
- **plot.py** — ROC/PR curve plotting, percentile-based raster visualization
- **metrics.py** — Cohen's Kappa, fuzzy confusion matrix

### Key Design Decisions

- Optional deps are lazy-imported at function call time, not module load time
- `__init__.py` uses `__getattr__` for lazy top-level exports
- All public functions have type hints (Python 3.12+)
- Parameter order is data-first: `save_model(model, path)` not `(path, model)`
```

- [ ] **Step 7: Run full test suite one final time**

```bash
pip install -e ".[all]" && pytest tests/ -v
```

Expected: all tests PASS, package installs cleanly.

- [ ] **Step 8: Commit**

```bash
git rm -r helper/ setup.py test.py
git add geo_helper/__init__.py tests/test_init.py CLAUDE.md
git commit -m "feat: wire up lazy __init__.py, remove old helper/ package, update CLAUDE.md"
```
