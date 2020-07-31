import os
import rasterio
from osgeo import gdal
import numpy as np
import pandas as pd
from . import df_tools
from . import geo_tools
from . import io_tools

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
    gdf_vew = geo_tools.convert_to_geodataframe(df_vew[['new_polygon_id', 'final_plant_date', 'START_DATE', 'loss_ratio', 'final_polygon']],  to_crs=to_crs)
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
    raster_date = pd.DataFrame(geo_tools.get_raster_date(raster, datetimetype='datetime', dtype='gdal'), columns=['date'])
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

        geo_tools.create_tiff(
            path_save = path_save,
            im = flood_im,
            projection = raster.GetProjection(),
            geotransform = raster.GetGeoTransform(),
            drivername = "GTiff",
            list_band_name = [str(item.date()) for item in raster_year['date']],
            nodata= 0,
            dtype = gdal.GDT_Byte,
            channel_first=True
        )
    
def extract_bits(img, position):
    """
    Extract specific bit(s)

    Parameters
    ----------
    img: numpy array (M, N)
        QA image.
    position: tuple(int, int) or int
        Bit(s)'s position read from Left to Right (if tuple)
    
    Examples
    --------
    >>> extract_bits(qa_img, position=(6, 5)) # For cloud confidence
    
    Returns
    -------
    Selected bit(s)
    """    
    if type(position) is tuple:
        bit_length = position[0]-position[1]+1
        bit_mask = int(bit_length*"1", 2)
        return ((img>>position[1]) & bit_mask).astype(np.uint8)
    
    elif type(position) is int:
        return ((img>>position) & 1).astype(np.uint8)
    
class LS8_QA:
    """
    Class for Landsat8 Quality Assetment

    Parameters
    ----------
    raster: raster data (gdal or rasterio)
        QA raster.
    channel: int
        Which channel of QA (1, 2, 3, 4, ..., N).
    
    Examples
    --------
    >>> LS8_QA(raster, channel=8)
    """
    # Channel: Raster band (1, 2, 3, 4, ..., N)
    def __init__(self, raster, channel=1):
        self.raster = raster
        
        if type(raster) == rasterio.io.DatasetReader:
            self.img = raster.read(channel)
            
        elif type(raster) == gdal.Dataset:
            self.img = raster.GetRasterBand(channel).ReadAsArray()
            
        # Shift bits for cloud, cloud confidence, cloud shadow confidence, cirrus confidence
        # https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band?qt-science_support_page_related_con=0#qt-science_support_page_related_con
        
        '''
        Cloud
        0:No 
        1:Yes
        '''
        self.cloud = extract_bits(self.img, 4)
        
        '''
        Cloud confidence, Cloud shadow confidence, Snow confidence, Cirrus confidence
        00(0):Not Determined
        01(1):Low
        10(2):Medium
        11(3):High
        '''
        self.cloud_confidence = extract_bits(self.img, (6, 5)) 
        self.cloud_shadow_confidence = extract_bits(self.img, (8, 7)) 
        self.snow_confidence = extract_bits(self.img, (10, 9))
        self.cirrus_confidence = extract_bits(self.img, (12, 11))