"""geo-helper: General-purpose geospatial helper utilities."""

__all__ = [
    "transform_coords",
    "get_pixel_from_coords",
    "get_coords_from_pixel",
    "create_tiff",
    "create_vrt",
    "warp",
    "rasterize",
    "get_band_descriptions",
    "get_band_dates",
    "from_wkt",
    "to_geodataframe",
    "save_model",
    "load_model",
]


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
