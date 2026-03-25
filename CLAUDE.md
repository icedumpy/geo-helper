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
