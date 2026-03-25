# geo-helper

General-purpose Python geospatial helper utilities for personal reuse.

## Installation

```bash
# Core (numpy + rasterio)
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Specific groups
pip install -e ".[geo]"    # geopandas, shapely, pyproj, GDAL
pip install -e ".[ml]"     # scikit-learn, joblib, h5py
pip install -e ".[plot]"   # matplotlib, scikit-learn
```

Requires Python >= 3.12.

## Quick Start

```python
from geo_helper import create_tiff, warp, from_wkt
from geo_helper.raster import rasterize, get_band_dates
from geo_helper.metrics import kappa, fuzzy_confusion_matrix
from geo_helper.plot import plot_roc_curve, plot_raster
from geo_helper.io import save_model, load_model, save_h5, load_h5
```

## Modules

| Module | Purpose | Optional Deps |
|--------|---------|---------------|
| `core` | Coordinate transforms, pixel/coord mapping | pyproj (for CRS reprojection) |
| `raster` | GeoTIFF/VRT creation, warp, rasterize, band metadata | GDAL (for VRT only) |
| `vector` | WKT parsing, GeoDataFrame creation | shapely, geopandas |
| `io` | Model serialization (joblib), HDF5 storage | joblib, h5py |
| `plot` | ROC/PR curves, raster visualization | matplotlib, scikit-learn |
| `metrics` | Cohen's Kappa, fuzzy confusion matrix | none (numpy only) |

## Testing

```bash
pytest tests/ -v
```
