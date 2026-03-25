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
        assert data.shape[0] == 10
        assert data.shape[1] == 10
        assert data.max() == 1
        assert src.nodata == 0
