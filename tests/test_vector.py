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
