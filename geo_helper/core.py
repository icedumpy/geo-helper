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
    col = int(np.floor(np.round(
        (pixel_height * (x - origin_x) - row_rot * (y - origin_y)) / denom,
        decimals=10,
    )))
    row = int(np.floor(np.round(
        (pixel_width * (y - origin_y) - col_rot * (x - origin_x)) / denom,
        decimals=10,
    )))
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
