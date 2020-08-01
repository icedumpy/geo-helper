import os
import pandas as pd
import numpy as np

def load_ls8_pixel_dataframe_every_year(root, p="", pathrow="", band=""):   
    """
    Will replace load_ls8_pixel_dataframe in the future due to the new dataframe format.
    Load landsat8 pixels dataframe of the selected p, pathrow, band
    
    Parameters
    ----------
    root: str
        Root directory of landsat8 pixel dataframe.
    p: str, (optional), default ""
        Selected province Ex. "p10".
    pathrow: str, (not optional) need to define
        Selected pathrow.
    band:  str, (not optional) need to define
        Selected band.
        
    Examples
    --------
    >>> load_ls8_pixel_dataframe_every_year(root, p="p10", pathrow="129050", band="B5")
    >>> load_ls8_pixel_dataframe_every_year(root, p="", pathrow="129050", band="B5")

    Returns
    -------
    list of landsat8 pixel dataframe files, Concatenated landsat8 pixel dataframe of every year
    """
    # Get all files
    dict_file = dict()
    for file in os.listdir(root):
        check_name = ""
        for item in [p, pathrow, band]:
            if item=="":
                continue
            else:
                check_name += f"_{item}"
        
        if check_name in file:
            if not file.split("_")[2] in dict_file.keys():
                dict_file[file.split("_")[2]] = [file]
            else:
                dict_file[file.split("_")[2]].append(file)
    
    list_file = []
    for item in list(dict_file.values()):
        for item2 in item:
            list_file.append(item2)
    
    list_df = []       
    for key in dict_file.keys():
        for file in dict_file[key]:
            if not "df" in locals():
                df = pd.read_parquet(os.path.join(root, file))
            else:
                df = pd.merge(df, pd.read_parquet(os.path.join(root, file)), how="inner", on=['new_polygon_id', 'row', 'col'])
        if len(df)!=0:
            list_df.append(df)
        del df
        
    df = pd.concat(list_df, ignore_index=True)
    return list_file, df

def load_ls8_pixel_dataframe(root, pathrow, band):
    '''
    Load landsat8 pixels dataframe with selected band and pathrow

    Parameters
    # =============================================================================
    # root: root directory of landsat8 pixel dataframe
    # pathrow: selected band pathrow
    # band: selected band
    # =============================================================================
    
    Return
    # =============================================================================
    # list of landsat8 pixel dataframe file name, Concatenated landsat8 pixel dataframe of every year
    # =============================================================================
    '''    
    columns_old = None
    list_df = []
    list_file = []
    for file in os.listdir(root):
        # province_file = file[11:13]
        pathrow_file = file.split(".")[0].split("_")[3]
        band_file = file.split(".")[0].split("_")[4]
        if band_file==band and pathrow_file==pathrow:
            path_file = os.path.join(root, file)
            list_file.append(path_file)
            
            df = pd.read_parquet(path_file)
            if columns_old is None:
                columns_old = df.columns
            else:
                assert (columns_old == df.columns).all()   
                
            list_df.append(df)
    df = pd.concat(list_df, ignore_index=True)
    
    return list_file, df

def load_ls8_cloudmask_dataframe(root_cloudmask, pathrow, filter_row_col=None):
    '''
    Load landsat8 cloudmask dataframe with selected pathrow and (filter_row_col)

    Parameters
    # =============================================================================
    # root_cloudmask: root directory of landsat8 cloudmask dataframe
    # pathrow: selected band pathrow
    # filter_row_col: default: None, filter cloudmask dataframe by input row_col << 100000*row + col
    # =============================================================================
    
    Return
    # =============================================================================
    # Cloudmask dataframe of every year
    # =============================================================================
    ''' 
    list_file = [file for file in os.listdir(root_cloudmask) if pathrow in file]
    list_file = sorted(list_file, key=lambda x: int(x.split(".")[0][-4:]), reverse=False)
    
    for i, file in enumerate(list_file):
        path_file = os.path.join(root_cloudmask, file)
        if i==0:
            df_cloudmask = pd.read_parquet(path_file)
            if filter_row_col is not None:
                df_cloudmask = df_cloudmask[(100000*df_cloudmask['row']+df_cloudmask['col']).isin(filter_row_col)]
        else:
            df_cloudmask_temp = pd.read_parquet(path_file)
            df_cloudmask = pd.merge(df_cloudmask, df_cloudmask_temp, how='inner', on=['row', 'col', 'scene_id'])
    return df_cloudmask

def load_mapping(root_mapping, list_p=None, list_pathrow=None):
    """
    Load mapping dataframe of the selected componants (p/pathrow)

    Parameters
    ----------
    root_mapping: str
        root directory of mapping dataftame.
    list_p: list of str (optional), default None
        List of p that you want.
    list_pathrow: list of str (optional), default None
        List of pathrow that you want.

    Examples
    --------
    >>> load_mapping(root_mapping, list_pathrow=['129049', '129048'])

    Returns
    -------
    Concatenated mapping dataframe, list_p (if input list_pathrow) vise versa
    """
    if list_p is not None:
        # list_other == list_pathrow
        list_other = [os.path.splitext(file)[0].split("_")[-1] for file in os.listdir(root_mapping) if os.path.splitext(file)[0].split("_")[-2][1:] in list_p]
        df = pd.concat([pd.read_parquet(os.path.join(root_mapping, file)) for file in os.listdir(root_mapping) if os.path.splitext(file)[0].split("_")[-2][1:] in list_p], ignore_index=True)
    
    elif list_pathrow is not None:
        # List_other == list_p
        list_other = [os.path.splitext(file)[0].split("_")[-2][1:] for file in os.listdir(root_mapping) if os.path.splitext(file)[0].split("_")[-1] in list_pathrow]
        df = pd.concat([pd.read_parquet(os.path.join(root_mapping, file)) for file in os.listdir(root_mapping) if os.path.splitext(file)[0].split("_")[-1] in list_pathrow], ignore_index=True)
    return df, list_other

def load_vew(root_vew, list_p):
    '''
    Load vew dataftame of selected P 

    Parameters
    # =============================================================================
    # root_vew: root directory of vew dataftame
    # list_p: list of selected province code in string ex. ['10', '12', '45']
    # =============================================================================
    
    Return
    # =============================================================================
    # vew dataframe
    # =============================================================================
    ''' 
    list_file = [file for file in os.listdir(root_vew) if os.path.splitext(file)[0][-2:] in list_p]
    df_vew = pd.concat([pd.read_parquet(os.path.join(root_vew, file)) for file in list_file], ignore_index=True)
    return df_vew

def clean_and_process_vew(df_vew, list_new_polygon_id=None):
    '''
    Clean and procress data of vew dataframe 
    1. Filter vew by new_polygon_id
    2. Remove row with zero plant area
    3. Calculate loss ratio and insert to "loss_ratio" column
    4. Drop row with Nan "final_plant_date"
    5. Filter only Normal and Flood 
    6. Convert string datetime into datetime (in "final_plant_date" and "START_DATE" columns)
    7. Drop deflect case (Flood but loss_ratio is 0)
    8. Drop deflect case (loss ratio more than 1)
    9. Keep only in-season rice (south = [Nakhon Si Thammarat, Songkhla, Phatthalung, Pattani, Yala, Narathewat])
        9.1) For every province except south, keep only if plant date is between (1st May, 31th October)
        9.2) For south, keep only if plant date is between (16th June, 28th Feb)
    
    Parameters
    # =============================================================================
    # df_vew: vew dataftame
    # list_new_polygon_id: default: None, filter vew dataframe by list of new_polygon_id
    # =============================================================================
    
    Return
    # =============================================================================
    # Cleaned and processed vew dataframe
    # =============================================================================
    '''
    # Filter by new_polygon_id
    if not list_new_polygon_id is None:
        df_vew = df_vew[df_vew['new_polygon_id'].isin(list_new_polygon_id)]
    
    # Remove zero plant area
    df_vew = df_vew[df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']!=0]

    # Get Loss ratio
#    df_vew['loss_ratio'] = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA'])
    df_vew = df_vew.assign(loss_ratio = np.where(pd.isna(df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']).values, 0, df_vew['TOTAL_DANGER_AREA_IN_WA']/df_vew['TOTAL_ACTUAL_PLANT_AREA_IN_WA']))
    
    
    # Drop NaN final_plant_date
    df_vew = df_vew.dropna(subset=['final_plant_date'])

    # Select only Normal and Flood
    df_vew = df_vew[(df_vew['DANGER_TYPE_NAME']=='อุทกภัย') | (pd.isnull(df_vew['DANGER_TYPE_NAME']))]
    
    # Convert date string into datetime
    df_vew['final_plant_date'] = pd.to_datetime(df_vew['final_plant_date'])

    # Replace no START_DATAE data with NaT
    df_vew.loc[df_vew[pd.isnull(df_vew['START_DATE'])].index, 'START_DATE'] = pd.NaT
    df_vew['START_DATE'] = pd.to_datetime(df_vew['START_DATE'],  errors='coerce')
    
    # Drop case [flood but loss_ratio==0] :== select only [no flood or loss_ratio!=0]
    df_vew = df_vew[pd.isna(df_vew['START_DATE']) | (df_vew['loss_ratio']!=0)]

    # Drop out of range loss ratio
    df_vew = df_vew[(df_vew['loss_ratio']>=0) & (df_vew['loss_ratio']<=1)]

    # In-season rice
    p_south = [80, 90, 93, 94, 95, 96]
    
    # True if (not south) & (plant_date between month (5, 10))
    non_south_flag = (~df_vew["PLANT_PROVINCE_CODE"].isin(p_south))&(df_vew['final_plant_date'].dt.month.between(5, 10))
    
    # True if (south) & (plant_date from (16 June to 28 Feb))
    south_flag = (df_vew["PLANT_PROVINCE_CODE"].isin(p_south)) & (((df_vew['final_plant_date'].dt.day >= 16) & (df_vew['final_plant_date'].dt.month > 6)) | (df_vew['final_plant_date'].dt.month < 3))
    
    # Filter only in-season rice
    season_rice_flag = non_south_flag | south_flag 
    df_vew = df_vew[season_rice_flag]
    
    # Reset index 
    df_vew = df_vew.reset_index(drop=True) 
    
    return df_vew

def set_index_for_loc(df, column=None, index=None):
    '''
    Set value in selected column as index and sort index 

    Parameters
    # =============================================================================
    # df: dataframe
    # column: which column to be set to index
    # =============================================================================
    
    Return
    # =============================================================================
    # Dataframe with changed and sorted index
    # =============================================================================
    ''' 
    if column is not None:
        df.index = df[column]
    elif index is not None:
        df.index = index
    # Sort if not sorted
    if not df.index.is_monotonic:
        df = df.sort_index()
    return df



