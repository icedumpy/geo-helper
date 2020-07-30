import os
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from osgeo import gdal
import rasterio
from . import df_tools
from . import io_tools

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
    dtype: gdal datatype (optional), default GDT_Float32
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

    if len(im.shape)==2:
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


def get_raster_date(raster, datetimetype, dtype):
    """
    Get band's date from each band of raster.

    Parameters
    ----------
    raster: gdal raster or rasterio raster
        Raster data.
    datetimetype: str
        Want return to be 'date' or 'datetime' format.
    dtype: str
        raster data type is 'gdal' or 'rasterio'.

    Examples
    --------
    >>> get_raster_date(rasterio.open(path_raster), datetimetype='date', dtype='rasterio')

    Returns
    -------
    Numpy array of raster date/datetime
    """

    if dtype=='rasterio':
        try:
            raster_date = [datetime.datetime(int(item[-8:-4]), int(item[-4:-2]), int(item[-2:])) for item in raster.descriptions]
        except:
            raster_date = [datetime.datetime(int(item.split('-')[0]), int(item.split('-')[1]), int(item.split('-')[2])) for item in raster.descriptions]
    elif dtype=='gdal':
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

def create_landsat8_color_infrared_img(root_raster, channel=1, rgb_or_bgr='rgb', channel_first=True):
    """
   Create landsat8 color infrared image (Band 5 as Red, Band 4 as Green, Band 3 as Blue).

    Parameters
    ----------
    root raster: str
        Root directory of landsat8 image Ex. os.listdir(root_raster) >> [band1.tif/vrt, band2.tif/vrt, ...].
    channel: int (optional), default 1
        Which channel from each raster to be used (Start from 1).
    rgb_or_bgr: str (optional), default 'rgb'
        Want return image to be "rgb" or "bgr" format.
    channel_first: boolean (optional), default True
        Image is channel first or not.
    Examples
    --------
    >>> create_landsat8_color_infrared_img(root_raster, 10)

    Returns
    -------
    Color infrared image (height, width, 3)
    """
    # Load raster band 5, 4, 3 for R, G, B image
    for file in os.listdir(root_raster):
        if "B5" in file:
            raster_for_band_red = rasterio.open(os.path.join(root_raster, file))
        elif "B4" in file:
            raster_for_band_grn = rasterio.open(os.path.join(root_raster, file))
        elif "B3" in file:
            raster_for_band_blu = rasterio.open(os.path.join(root_raster, file))

    # Combine band 5, 4, 3
    try:
        # Check if every band are loaded
        assert "raster_for_band_red" in locals() and "raster_for_band_grn" in locals() and "raster_for_band_blu" in locals()

        # color_infrared_img.shape = (3, height, width) # (543, height, width)
        color_infrared_img = np.vstack((raster_for_band_red.read([channel]),
                                        raster_for_band_grn.read([channel]),
                                        raster_for_band_blu.read([channel])
                                      ))

        # if bgr, change from rgb to bgr (inverse first channel)
        if rgb_or_bgr=='bgr':
            color_infrared_img = color_infrared_img[::-1]

        # Finally, convert from channel_first to channel_last
        if not channel_first:
            color_infrared_img = np.moveaxis(color_infrared_img, 0, -1)
        return color_infrared_img

    except AssertionError as error:
        return error

def create_rice_mask(path_save, root_vew, root_df_ls8, pathrow, raster, to_crs):
    """
    Create rice mask of the selected pathrow (Every pixel that contain rice).

    Parameters
    ----------
    path_save: str
        Save path (.shp).
    root_vew: str
        Root directory of vew dataframe (for geometry).
    root_df_ls8: str
        Root directory of landsat8's pixel values dataframe (To filter vew by new polygon id).
    pathrow: str
        Selected pathrow.
    raster: gdal's raster
        Any raster of the selected pathrow (Need  geo information of the selected pathrow).
    to_crs: dictionary
        Selected raster's crs

    Examples
    --------
    >>> create_rice_mask(path_save, root_vew, root_df_ls8, pathrow, raster, to_crs)

    Returns
    -------
    None
    """
    # Create folder
    if not os.path.exists(os.path.dirname(path_save)):
        os.makedirs(os.path.dirname(path_save))

    # Load vew of selected pathrow
    list_file, df_ls8 = df_tools.load_ls8_pixel_dataframe_every_year(root_df_ls8, pathrow=pathrow, band="B5")

    # Double clean 1: remove where all column = 0
    #              2: remove where 0 more than 10% of data column
    df_ls8 = df_ls8[~(df_ls8[df_ls8.columns[3:]]==0).all(axis=1)]
    df_ls8 = df_ls8[~((df_ls8[df_ls8.columns[3:]]==0).sum(axis=1)>=1)]

    list_p = [os.path.basename(file)[11:13] for file in list_file]
    list_new_polygon_id = pd.unique(df_ls8['new_polygon_id']).tolist()
    df_vew = df_tools.load_vew(root_vew, list_p)
    df_vew = df_tools.clean_and_process_vew(df_vew, list_new_polygon_id)

    # Convert dataframe to geodataframe
    gdf_vew = convert_to_geodataframe(df_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'loss_ratio', 'final_polygon']],  to_crs=to_crs)
    gdf_vew['START_DATE'] = gdf_vew['START_DATE'].astype(str)
    gdf_vew['final_plant_date'] = gdf_vew['final_plant_date'].astype(str)
    gdf_vew = gdf_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'loss_ratio', 'geometry']]
    gdf_vew = gdf_vew[gdf_vew['geometry'].type == "MultiPolygon"]

    # Save geodataframe to shapefile
    gdf_vew.to_file(path_save)

    # Rasterize shapefile to raster
    path_rasterize = os.path.join(os.path.dirname(path_save), os.path.basename(path_save).replace('.shp', '.tif'))
    command = f"gdal_rasterize -at -l {os.path.splitext(os.path.basename(path_rasterize))[0]} -burn 1.0 -ts {raster.RasterXSize:.1f} {raster.RasterYSize:.1f} -a_nodata 0.0 -te {raster.GetGeoTransform()[0]} {raster.GetGeoTransform()[3] + raster.RasterYSize*raster.GetGeoTransform()[5]} {raster.GetGeoTransform()[0] + raster.RasterXSize*raster.GetGeoTransform()[1]} {raster.GetGeoTransform()[3]} -ot Byte -of GTiff" + " {} {}".format(path_save.replace('\\', '/'), path_rasterize.replace('\\', '/'))
    os.system(command)

def create_flood_map(root_save, root_raster, path_mask, path_model, threshold, pathrow, bands, val_nodata=2, val_noflood=0):
    """
    Create Flood map.

    Parameters
    ----------
    root_save: str
        Root directory for save.
    root_raster: str
        Root directory of landsat8 raster.
    path_mask: str
        Path of rice mask.
    path_model: str
        Path of model.
    threshold: float
        Flood map threshold.
    pathrow: str
        Pathrow of flood map.
    bands: list of string
        List of bands to be used for model
    val_nodata: float or int (optional), default -9999
        No value data of raster.
    val_noflood: float or int (optional), default 0
        No flood value data of raster

    Examples
    --------
    >>>

    Returns
    -------
    None
    """
    model = io_tools.load_model(path_model)

    # Get rice mask locations
    mask_rice = gdal.Open(path_mask).ReadAsArray()
    row_rice, col_rice = np.nonzero(mask_rice)
    del mask_rice

    # Load every raster into dict
    dict_raster = dict()
    for file in os.listdir(root_raster):
        band = file.split(".")[0].split("_")[2]
        if band in bands:
            dict_raster[band] = gdal.Open(os.path.join(root_raster, file))
    assert len(np.unique([raster.RasterCount for raster in dict_raster.values()]))==1

    # Initial parameters for predicted flood map
    raster = dict_raster[bands[0]]
    raster_date = pd.DataFrame(get_raster_date(raster, datetimetype='datetime', dtype='gdal'), columns=['date'])
    raster_date['year'] = raster_date['date'].dt.year
    raster_date.index = raster_date.index+1

    depth = raster.RasterCount
    width = raster.RasterXSize
    height = raster.RasterYSize

    # Create flood map of each year
    for year, raster_year in raster_date.groupby(['year']):
        path_save = os.path.join(root_save, str(pathrow), f"{pathrow}_y{year}_from_{os.path.basename(os.path.dirname(path_model))}.tif")
        flood_im = val_nodata*np.ones((len(raster_year), height, width), dtype='uint8')

        for idx, channel in enumerate(raster_year.index):
            if (channel-2 < 1) or (channel+1) > depth:
                continue
            print(f"Channel: {channel}", f"[{channel-2}, {channel+1}] {year}")

            x = np.zeros((len(row_rice), 4*len(bands)), dtype='float32')
            for num_band, band in enumerate(bands):

                for i, subchannel in enumerate(range(channel-2, channel+2)):
                    x[:, 4*num_band+i] = dict_raster[band].GetRasterBand(subchannel).ReadAsArray()[row_rice, col_rice]

                if band == 'BQA':
                    x[:, 4*num_band:4*(num_band+1)] = (x[:, 4*num_band:4*(num_band+1)]-x[:, 4*num_band:4*(num_band+1)].min())/(x[:, 4*num_band:4*(num_band+1)].max()-x[:, 4*num_band:4*(num_band+1)].min())

            pred = model.predict_proba(x)
            pred_thresh = (pred[:, 1]>=threshold).astype('uint8')
            flood_im[idx, row_rice, col_rice] = pred_thresh # [0 non-flood, 1 flood, 2 no data]
            flood_im[flood_im==0] = val_noflood

        create_tiff(path_save = path_save,
                    im = flood_im,
                    projection = raster.GetProjection(),
                    geotransform = raster.GetGeoTransform(),
                    drivername = "GTiff",
                    list_band_name = [str(item.date()) for item in raster_year['date']],
                    nodata= 0,
                    dtype = gdal.GDT_Byte,
                    channel_first=True)
    
