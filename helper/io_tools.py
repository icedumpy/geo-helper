import os
import h5py
from joblib import dump, load


def save_model(path, model):
    """
    Save sklearn's model.

    Parameters
    ----------
    path: str
        save path.
    model: Sklearn's model datatype
        Sklearn's model.

    Examples
    --------
    >>>
    
    Returns
    -------
    None
    """    

    if not path.endswith(".joblib"):
        path = os.path.splitext(path)[0] + ".joblib"
    dump(model, path)

    
def load_model(path):
    """
    Load sklearn's model.

    Parameters
    ----------
    path: str
        Model path.

    Examples
    --------
    >>>
    
    Returns
    -------
    Loaded model
    """   
    return load(path) 

def save_h5(path, dict_save):
    """
    Save parameters in .h5 fotmat (h5py).

    Parameters
    ----------
    path: str
        save path.
    dict_save: dictionary
        Dictionary contains parameters to be saved.

    Examples
    --------
    >>> save_h5(path, dict_save={"ice":np.array([10, 0]),
                                 "lnw":np.array([10, 20],
                                                [30, 40])})

    Returns
    -------
    
    """   
    with h5py.File(path, 'w') as hf:
        for key in dict_save.keys():
            hf.create_dataset(key, data=dict_save[key])
            
def load_h5(path):
    """
    Load parameters from .h5 file (h5py).

    Parameters
    ----------
    path: str
        File path.

    Examples
    --------
    >>> 

    Returns
    -------
    
    """ 
    dict_output = dict()
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            dict_output[key] = f[key][()]
    return dict_output