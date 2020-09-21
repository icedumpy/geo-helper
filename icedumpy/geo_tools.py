import datetime
import numpy as np
import geopandas as gpd
import pyproj
from shapely import wkt
from shapely.ops import transform
from osgeo import gdal
import rasterio
import matplotlib.pyplot as plt


def create_polygon_from_wkt(wkt_polygon, crs="epsg:4326", to_crs=None):
    """
    Create shapely polygon from string (wkt format) "MULTIPOLYGON(((...)))"
    https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects/127432#127432

    Parameters
    ----------
    wkt_polygon: str
        String of polygon (wkt format).
    crs: str
        wkt_polygon's crs (should be "epsg:4326").
    to_crs: str (optional), default None
        Re-project crs to "to_crs".

    Examples
    --------
    >>> create_polygon_from_wkt(wkt_polygon)
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32647")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32648")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32660")
    >>> create_polygon_from_wkt(wkt_polygon, crs="epsg:32647", to_crs="epsg:4326")

    Returns
    -------
    Polygon
    """
    polygon = wkt.loads(wkt_polygon)
    if to_crs is not None:
        if crs == "epsg:4326":
            if to_crs == "epsg:32647":
                project_47 = pyproj.Transformer.from_crs(
                    'epsg:4326',   # source coordinate system
                    'epsg:32647',  # destination coordinate system
                    always_xy=True # Must have
                ).transform                
                polygon = transform(project_47, polygon)
            elif to_crs == "epsg:32648":
                project_48 = pyproj.Transformer.from_crs(
                    'epsg:4326',   # source coordinate system
                    'epsg:32648',  # destination coordinate system
                    always_xy=True # Must have
                ).transform 
                polygon = transform(project_48, polygon)
        else:
            project = pyproj.Transformer.from_crs(
                crs,     # source coordinate system
                to_crs,  # destination coordinate system
                always_xy=True # Must have
            ).transform
            polygon = transform(project, polygon)
    return polygon

def plot_vminmax(img, vminmax, ax=None):
    """
    Plot image with nanpercentile cut.

    Parameters
    ----------
    img: 2D-numpy array
        Image array.
    vmin: tuple
        Tuple of (min percent, max percent).
    ax: matplotlib suplots ax (optional), default None
        axis for plot    
    
    Examples
    --------
    >>> plot_vminmax(img, vminmax=(2, 98))

    Returns
    -------
    matplotlib fig and ax
    """
    vmin, vmax = np.nanpercentile(img, vminmax)
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, vmin=vmin, vmax=vmax)
    return fig, ax


def create_tiff(path_save, im, projection, geotransform, drivername, list_band_name=None, nodata=0, channel_first=True, dtype=gdal.GDT_Float32):
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
    nodata: int (optional), default 0
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
    
def create_vrt(path_save, list_path_raster, list_band_name=None, src_nodata=None, dst_nodata=None):
    """
    Write raster from image (can use with gdal and rasterio raster).

    Parameters
    ----------
    path_save: str
        Virtual's save path.
    list_band_name: list of string (optional), default None
        List of the name of each band. Otherwise, blank band names.
    src_nodata: int (optional), default None
        Nodata value of the source raster.
    dst_nodata: int (optional), default None
        Nodata value of the virtual raster.
        
    Examples
    --------
    >>> create_vrt(path_save="ice.tif", list_path_raster=["raster1.tif", "raster2.tif"], list_band_name=["ice", "lnw"], src_nodata=0, dst_nodata=0)

    Returns
    -------
    None
    """
    options = gdal.BuildVRTOptions(
        separate=True,
        srcNodata = src_nodata,
        VRTNodata = dst_nodata
    )
    outds = gdal.BuildVRT(path_save, list_path_raster, options=options)
    if list_band_name is not None:
        for idx, band_name in enumerate(list_band_name):
            outds.GetRasterBand(idx+1).SetDescription(band_name)
    outds.FlushCache()
    del outds

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

def gdal_descriptions(raster):
    """
    The same as raster.descriptions in rasterio.

    Parameters
    ----------
    raster: gdal raster
        Raster data.

    Examples
    --------
    >>>

    Returns
    -------
    The list of descriptions for each dataset band.
    """
    band_description = [raster.GetRasterBand(i).GetDescription() for i in range(1, raster.RasterCount+1)]
    return band_description


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