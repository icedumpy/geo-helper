import numpy as np
import pytest
from geo_helper.core import get_pixel_from_coords, get_coords_from_pixel


def test_get_pixel_from_coords_identity():
    """Origin pixel should be (0, 0) when coords match the transform origin."""
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
    x, y = get_coords_from_pixel(7, 3, transform)
    col, row = get_pixel_from_coords(x, y, transform)
    assert col == 3
    assert row == 7


def test_get_coords_from_pixel_values():
    """Check actual coordinate values for a known pixel."""
    transform = (100.0, 0.01, 0.0, 15.0, 0.0, -0.01)
    x, y = get_coords_from_pixel(0, 0, transform)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(15.0)


def test_transform_coords():
    """Test coordinate reprojection from EPSG:4326 to EPSG:32647."""
    pyproj = pytest.importorskip("pyproj")
    from geo_helper.core import transform_coords
    x, y = transform_coords(100.5, 13.75, "EPSG:4326", "EPSG:32647")
    assert 600_000 < x < 800_000
    assert 1_400_000 < y < 1_600_000


def test_transform_coords_identity():
    """Same source and dest CRS should return ~same coordinates."""
    pyproj = pytest.importorskip("pyproj")
    from geo_helper.core import transform_coords
    x, y = transform_coords(100.5, 13.75, "EPSG:4326", "EPSG:4326")
    assert x == pytest.approx(100.5, abs=1e-6)
    assert y == pytest.approx(13.75, abs=1e-6)
