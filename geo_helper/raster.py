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
    """Write a georeferenced raster from a numpy array."""
    if array.ndim == 2:
        array = array[np.newaxis, :, :]

    count, height, width = array.shape
    dtype = dtype or str(array.dtype)

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype=dtype, crs=crs, transform=transform, nodata=nodata,
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
    """Create a virtual raster (VRT) from multiple source rasters. Requires GDAL."""
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError(
            "GDAL is required for create_vrt(). Install with: pip install geo-helper[geo]"
        )
    options = gdal.BuildVRTOptions(separate=True, srcNodata=src_nodata, VRTNodata=dst_nodata)
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
    """Reproject and optionally clip a raster."""
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    resample_method = Resampling[resampling]

    with rasterio.open(src_path) as src:
        dst_transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution,
        )

        if bounds is not None:
            from rasterio.transform import from_bounds
            width = int(abs((bounds[2] - bounds[0]) / dst_transform.a))
            height = int(abs((bounds[3] - bounds[1]) / dst_transform.e))
            dst_transform = from_bounds(*bounds, width, height)

        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=dst_transform, width=width, height=height)

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
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
    """Rasterize a vector file."""
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for rasterize(). Install with: pip install geo-helper[geo]"
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
    arr = rio_rasterize(shapes, out_shape=(height, width), transform=transform, fill=nodata, dtype=dtype)

    with rasterio.open(
        output_path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype=dtype, crs=out_crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def get_band_descriptions(path: str | Path) -> tuple[str, ...]:
    """Read band description strings from a raster file."""
    with rasterio.open(path) as src:
        return src.descriptions


def get_band_dates(
    path: str | Path, date_format: str | None = None
) -> list[datetime] | list[str]:
    """Extract dates from raster band descriptions."""
    descriptions = get_band_descriptions(path)
    if date_format is None:
        return [d or "" for d in descriptions]
    return [datetime.strptime(d, date_format) for d in descriptions]
