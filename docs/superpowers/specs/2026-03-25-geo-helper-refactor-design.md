# geo-helper Refactoring Design Spec

## Goal

Refactor `geo-helper` from a PIER-project-specific utility bag into a clean, general-purpose Python geospatial helper package for personal reuse across future geo projects.

## Decisions

- **Target audience**: Single user (personal toolbox)
- **Python**: >= 3.12
- **Packaging**: `pyproject.toml` with `setuptools` backend (replaces `setup.py`)
- **Approach**: Layered modules with a shared `core.py`
- **What to drop**: All PIER/Landsat8/rice/VEW-specific code (`df_tools.py`, `tf_tools.py`, `ls8_tools.py`). Domain-specific functions are preserved in git history.
- **Breaking changes**: This is a full rewrite. The old `helper` package is removed entirely. Any existing scripts using `from helper.* import ...` will need to update to `from geo_helper.* import ...`. Version will be `1.0.0` to signal the clean break.

## Package Structure

```
geo-helper/
├── pyproject.toml
├── geo_helper/
│   ├── __init__.py    # public API re-exports (lazy, to avoid forcing optional deps)
│   ├── core.py        # shared geospatial primitives
│   ├── raster.py      # raster I/O and operations
│   ├── vector.py      # vector/geometry operations
│   ├── io.py          # model & data serialization
│   ├── plot.py        # visualization
│   └── metrics.py     # evaluation metrics
├── tests/
│   ├── test_core.py
│   ├── test_raster.py
│   ├── test_vector.py
│   ├── test_io.py
│   ├── test_plot.py
│   └── test_metrics.py
```

### Files to delete

- `setup.py`
- `test.py`
- `helper/` (entire old package directory)

## Module Design

### `core.py` — Shared Geospatial Primitives

Functions that both raster and vector modules depend on. `pyproj` is lazy-imported only in `transform_coords`, so `get_pixel_from_coords` and `get_coords_from_pixel` work with just numpy.

| Function | Purpose | Notes |
|----------|---------|-------|
| `transform_coords(x, y, src_crs, dst_crs)` | Coordinate reprojection via pyproj | Lazy-imports pyproj; extracted from `create_polygon_from_wkt` |
| `get_pixel_from_coords(x, y, transform)` | Map geo coordinates to raster pixel (row, col) | Extracted from `get_pixel_location_from_coords` |
| `get_coords_from_pixel(row, col, transform)` | Map pixel position to geo coordinates | **New function** — inverse of above, uses affine transform math |

### `raster.py` — Raster File Operations

Depends on: `core`. Uses `rasterio` as the primary raster library. GDAL (`osgeo`) is lazy-imported only for `create_vrt` (no rasterio equivalent). All other functions use rasterio APIs.

| Function | Purpose | Notes |
|----------|---------|-------|
| `create_tiff(path, array, transform, crs, ...)` | Write georeferenced raster from numpy array | Rewritten to use rasterio (original used GDAL directly) |
| `create_vrt(output_path, input_paths)` | Create virtual raster from multiple sources | Uses GDAL (lazy import); extracted from `create_vrt` |
| `warp(src_path, dst_path, dst_crs, bounds=None, resolution=None, resampling="nearest")` | Reproject/clip rasters | Uses rasterio.warp (replaces GDAL `gdal_warp` shell call). `dst_crs` required; `bounds` and `resolution` optional for clip/resample. |
| `rasterize(vector_path, output_path, resolution: float \| tuple[float, float], bounds=None, crs=None, burn_value=1, dtype="uint8", nodata=0)` | Convert vector to raster | Uses rasterio.features (replaces `gdal_rasterize` `os.system()` call). `resolution` accepts a single float (square pixels) or `(xRes, yRes)` tuple. |
| `get_band_descriptions(path)` | Read band metadata strings | Takes file path (not open dataset); opens internally. Changed from `gdal_descriptions` which took an open GDAL dataset. |
| `get_band_dates(path, date_format=None)` | Parse dates from band descriptions | If `date_format` is provided, uses `strptime`; otherwise returns raw strings. Simplified from original `get_raster_date` which had `string_format` + `start_index`. |

### `vector.py` — Vector/Geometry Operations

Depends on: `core`

| Function | Purpose | Notes |
|----------|---------|-------|
| `from_wkt(wkt_string, src_crs=None, dst_crs=None)` | Parse WKT to Shapely geometry with optional reprojection | Two CRS params: `src_crs` (default EPSG:4326) and `dst_crs` (if provided, reprojects). Extracted from `create_polygon_from_wkt`. |
| `to_geodataframe(geometries, crs=None)` | Convert list of Shapely geometry objects to GeoDataFrame | `geometries` is `list[BaseGeometry]`. Simplified from `convert_to_geodataframe` which also parsed WKT from a DataFrame column. |

### `io.py` — Serialization (Standalone)

API change: parameter order is now `(data, path)` instead of the old `(path, data)`. Data-first is more natural and consistent with numpy/pandas conventions.

| Function | Purpose | Notes |
|----------|---------|-------|
| `save_model(model, path)` | Persist sklearn model via joblib | **Param order changed** from old `(path, model)` |
| `load_model(path)` | Load sklearn model via joblib | Unchanged |
| `save_h5(data, path)` | Save dict of arrays to HDF5 | **Param order changed** from old `(path, dict_save)` |
| `load_h5(path)` | Load HDF5 file to dict | Unchanged |

### `plot.py` — Visualization (Standalone)

Functions are decoupled from sklearn models — they take pre-computed arrays instead of calling `model.predict_proba()` internally.

| Function | Purpose | Notes |
|----------|---------|-------|
| `plot_roc_curve(y_true, y_score, ax=None, label=None, **kwargs)` | ROC curve with AUC display | **Redesigned**: takes y arrays directly instead of `(model, x, y)`. Computes AUC internally via `sklearn.metrics.roc_auc_score`. Includes diagonal reference line and axis labels (absorbs old `set_roc_plot_template`). |
| `plot_precision_recall_curve(y_true, y_score, ax=None, label=None, **kwargs)` | Precision-recall curve | Same redesign. Computes AP via `sklearn.metrics.average_precision_score`. |
| `plot_raster(array, percentile_clip=True, ax=None, **kwargs)` | Raster visualization with percentile-based scaling | Renamed from `plot_vminmax`. |

Note: `set_roc_plot_template` from the original code is **intentionally dropped** — its styling (diagonal line, axis labels) is folded into `plot_roc_curve` directly.

### `metrics.py` — Evaluation (Standalone)

| Function | Purpose | Notes |
|----------|---------|-------|
| `kappa(confusion_matrix)` | Cohen's Kappa from confusion matrix | `confusion_matrix` is a numpy array. Rewritten to use pure numpy (original used pandas internally). |
| `fuzzy_confusion_matrix(y_true, y_pred, thresholds=None)` | Soft-label confusion matrix for continuous predictions | Rewritten to use pure numpy (original returned `pd.DataFrame`; refactored version returns numpy array). |

## Dependency Graph

```
core.py  (numpy; pyproj lazy-imported only in transform_coords)
  ├── raster.py  (rasterio; osgeo/GDAL lazy-imported only for create_vrt)
  └── vector.py  (shapely, geopandas)

io.py       (joblib, h5py)        — standalone
plot.py     (matplotlib, sklearn) — standalone
metrics.py  (numpy only)          — standalone
```

No circular dependencies. `core` is the only shared import.

## Dependencies & Optional Groups

### Required
- numpy
- rasterio

### Optional (via `pip install geo-helper[group]`)

| Group | Packages | Enables |
|-------|----------|---------|
| `geo` | geopandas, shapely, pyproj, GDAL | `core.transform_coords`, `vector.py`, `raster.create_vrt` |
| `ml` | scikit-learn, joblib, h5py | `io.py` |
| `plot` | matplotlib, scikit-learn | `plot.py` |
| `all` | all of the above | everything |

Functions that need an uninstalled optional dep raise `ImportError` with a message like:
`"pyproj is required for transform_coords(). Install with: pip install geo-helper[geo]"`

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "geo-helper"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = ["numpy", "rasterio"]

[project.optional-dependencies]
geo = ["geopandas", "shapely", "pyproj", "GDAL"]
ml = ["scikit-learn", "joblib", "h5py"]
plot = ["matplotlib", "scikit-learn"]
all = ["geo-helper[geo]", "geo-helper[ml]", "geo-helper[plot]"]
```

## API Surface

`__init__.py` uses lazy imports to avoid pulling in optional deps at import time:

```python
# geo_helper/__init__.py
def __getattr__(name):
    """Lazy import public API functions."""
    # Only the most commonly used functions are exported at the top level.
    # Less common functions (metrics, plot, h5) are accessed via their modules:
    #   from geo_helper.metrics import kappa
    #   from geo_helper.plot import plot_roc_curve
    _exports = {
        "transform_coords": "geo_helper.core",
        "get_pixel_from_coords": "geo_helper.core",
        "get_coords_from_pixel": "geo_helper.core",
        "create_tiff": "geo_helper.raster",
        "create_vrt": "geo_helper.raster",
        "warp": "geo_helper.raster",
        "rasterize": "geo_helper.raster",
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

Usage:
```python
from geo_helper import create_tiff, warp, from_wkt
# or
from geo_helper.metrics import fuzzy_confusion_matrix
```

## Key Design Principles

1. **No hardcoded values** — all CRS, paths, thresholds are parameters with sensible defaults
2. **Type hints on all public functions** — Python 3.12+ syntax
3. **Lazy imports for optional deps** — import at function call time, not module load time
4. **Small, focused functions** — each does one thing
5. **Consistent error messages** — clear about what's wrong and how to fix it
6. **Data-first parameter order** — `(data, path)` not `(path, data)`, consistent with numpy/pandas conventions

## Testing

- pytest as test runner
- Each module gets a corresponding test file
- Tests use synthetic data (small numpy arrays, in-memory rasters)
- No real satellite imagery or external file dependencies
- Functions requiring GDAL can be skipped with `pytest.importorskip("osgeo")`
