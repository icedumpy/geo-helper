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
