def test_lazy_import_core():
    """Top-level imports should resolve to core functions."""
    from geo_helper import get_pixel_from_coords, get_coords_from_pixel

    assert callable(get_pixel_from_coords)
    assert callable(get_coords_from_pixel)


def test_lazy_import_raster():
    """Top-level imports should resolve to raster functions."""
    from geo_helper import create_tiff, warp

    assert callable(create_tiff)
    assert callable(warp)


def test_lazy_import_io():
    """Top-level imports should resolve to io functions."""
    from geo_helper import save_model, load_model

    assert callable(save_model)
    assert callable(load_model)


def test_invalid_import_raises():
    """Importing nonexistent name should raise ImportError."""
    import pytest

    with pytest.raises(ImportError):
        from geo_helper import nonexistent_function  # noqa: F401
