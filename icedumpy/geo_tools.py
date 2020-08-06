import datetime
import numpy as np
import geopandas as gpd
from shapely import wkt
from osgeo import gdal
import rasterio

def create_tiff(path_save, im, projection, geotransform, drivername, list_band_name=None, nodata = -9999, channel_first=True, dtype=gdal.GDT_Float32):
    """
    Write raster from image (can use with gdal and rasterio raster).

    Parameters
    ----------
    path_save: str
        Raster's save path.
    im: 3D numpy array (channel, height, width) or (height, width, channel)
        Image to be saved to raster.
    projection: (gdal)raster.GetProjection() or (rasterio)raster.crs.to_wkt()
        Projection of the raster.
    geotransform: (gdal)raster.GetGeoTransform() or (rasterio)raster.get_transform()
        Geotransform of the raster.
    drivername: str
        Name of gdal driver ex. "GTiff", "ENVI" from https://gdal.org/drivers/raster/index.html
    list_band_name: list of string (optional), default None
        List of the name of each band. Otherwise, blank band's name.
    nodata: int (optional), default -9999
        Nodata value of the raster.
    channel_first: boolean (optional), default True
        Image is channel first or not.
    dtype: str or gdal datatype (optional), default GDT_Float32
        Datatype of the saved raster (gdal.GDT_xxx)

    Examples
    --------
    >>> create_tiff(path_save = path_save,
                    im = flood_im,
                    projection = raster.GetProjection(),
                    geotransform = raster.GetGeoTransform(),
                    drivername = "GTiff",
                    list_band_name = [str(item.date()) for item in raster_year['date']],
                    nodata= -9999,
                    dtype = gdal.GDT_Byte,
                    channel_first=True)

    Returns
    -------
    None
    """
    if type(dtype) == str:
        if dtype == "uint8":
            dtype = gdal.GDT_Byte
        elif dtype == "uint16":
            dtype = gdal.GDT_UInt16
        elif dtype == "uint32":
            dtype = gdal.GDT_UInt32
        elif dtype == "int16":
            dtype = gdal.GDT_Int16
        elif dtype == "int32":
            dtype = gdal.GDT_Int32
        elif dtype == "float32":
            dtype = gdal.GDT_Float32
        elif dtype == "float64":
            dtype = gdal.GDT_Float64
    
    if len(im.shape) == 2:
        im = np.expand_dims(im, 0)
    if not channel_first:
        im = np.moveaxis(im, -1, 0)

    band = im.shape[0]
    row = im.shape[1]
    col = im.shape[2]
    driver = gdal.GetDriverByName(drivername)
    output = driver.Create(path_save, col, row, band, dtype)
    output.SetProjection(projection)
    output.SetGeoTransform(geotransform)
    for i in range(band):
        if list_band_name is not None:
            band_name = list_band_name[i]
            output.GetRasterBand(i+1).SetDescription(band_name)
        output.GetRasterBand(i+1).WriteArray(im[i, :, :])

        if nodata is not None:
            output.GetRasterBand(i+1).SetNoDataValue(nodata)
        output.FlushCache()

    del output
    del driver


def get_raster_date(raster, datetimetype):
    """
    Get band's date from each band of raster.

    Parameters
    ----------
    raster: gdal raster or rasterio raster
        Raster data.
    datetimetype: str
        Want return to be 'date' or 'datetime' format.

    Examples
    --------
    >>> get_raster_date(rasterio.open(path_raster), datetimetype='date', dtype='rasterio')

    Returns
    -------
    Numpy array of raster date/datetime
    """

    if type(raster) == rasterio.io.DatasetReader:
        try:
            raster_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
        except:
            raster_date = [datetime.datetime(int(item.split('-')[0]), int(item.split('-')[1]), int(item.split('-')[2])) for item in raster.descriptions]
    elif type(raster) == gdal.Dataset:
        try:
            raster_date = [datetime.datetime(int(raster.GetRasterBand(i+1).GetDescription()[-8:-4]), int(raster.GetRasterBand(i+1).GetDescription()[-4:-2]), int(raster.GetRasterBand(i+1).GetDescription()[-2:])) for i in range(raster.RasterCount)]
        except:
            raster_date = [datetime.datetime(int(raster.GetRasterBand(i+1).GetDescription().split('-')[0]), int(raster.GetRasterBand(i+1).GetDescription().split('-')[1]), int(raster.GetRasterBand(i+1).GetDescription().split('-')[2])) for i in range(raster.RasterCount)]

    if datetimetype=='date':
        raster_date = [item.date() for item in raster_date]

    return np.array(raster_date)

def convert_to_geodataframe(df, polygon_column='final_polygon', crs={'init':'epsg:4326'}, to_crs=None):
    """
    Convert string of polygon to REAL polygon (geometry).
    Ex. "MULTIPOLYGON(((..., ...)))" as a one datapoint of string of polygon

    Parameters
    ----------
    df: pandas dataframe
        dataframe contains a string of polygon.
    polygon_column: str (optional), default "final_polygon"
        which column of dataframe contains a string of polygon.
    crs: dictionary (optional), default {'init':'epsg:4326'}
        which crs does dataframe contain
        ex. UTM47 is {'init':'epsg:32647'}, UTM48 is {'init':'epsg:32648'},
        lat-lon is {'init':'epsg:4326'}.
    to_crs: dictionary (optional), default None
        If you want to convert crs to another crs ex. {'init':'epsg:32647'} to {'init':'epsg:4326'}

    Examples
    --------
    >>> convert_to_geodataframe(df, polygon_column='final_polygon', crs={'init':'epsg:4326'})

    Returns
    -------
    Geodataframe (can be saved to shapefile using gdf.to_file("xxx.shp"))
    """

    # Avoid changing data of the input dataframe
    df = df.copy()

    gdf = gpd.GeoDataFrame(df)
    gdf['geometry'] = gdf[polygon_column].apply(wkt.loads)
    gdf.crs = crs
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    return gdf

def gdal_warp(path_after_warp, path_before_warp, outputBounds):
    """
    Warp (reproject) raster to the selected outputBounds
    
    Parameters
    ----------
    path_after_warp: str
        Path of the raster after warp (save path).
    path_before_warp: str
        Path of the raster before warp.
    outputBounds: tuple of float
        (x_min, y_min, x_max, y_max)

    Examples
    --------
    >>> 

    Returns
    -------
    none
    """
    # srs = osr.SpatialReference(wkt=raster_original.GetProjection())
    options = gdal.WarpOptions(format="GTiff", # format="VRT",
                               outputBounds=outputBounds,
                               # srcSRS = "EPSG:4326", dstSRS = "EPSG:4326",
                               outputType = gdal.GDT_Float32,
                               resampleAlg = 'near'
                               )    
    ds = gdal.Warp(path_after_warp, path_before_warp, options=options)
    del ds