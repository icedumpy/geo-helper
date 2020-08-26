import numpy as np
import pandas as pd

def kappa_from_cnf_matrix(cnf_matrix):
    """
    Calculate kappa coefficient from confusion matrix.

    Parameters
    ----------
    cnf_matrix: numpy array
        Confusion matrix.

    Examples
    --------
    >>>

    Returns
    -------
    kappa coefficient
    """
    N = cnf_matrix.sum()
    M = np.diag(cnf_matrix).sum()
    G = cnf_matrix.sum(axis=1)
    C = cnf_matrix.sum(axis=0)
    GC = np.dot(G, C)
    kappa_from_cnf_matrix = (N*M - GC)/(np.power(N, 2) - GC)

    return kappa_from_cnf_matrix

def get_fuzzy_confusion_matrix(predicted, ref):
    '''
    This function return fuzzy confusion matrix
    where
        predicted is list of predicted loss ratio
        ref is list of groundtruth loss ratio
    '''
    '''
                        Ref
                   |Flooded|No Flood|
    Pred    Flooded|_______|________|
         No Flooded|_______|________|
    '''
    f_cnf_matrix = np.zeros((2, 2), dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    df_results = pd.DataFrame({'Pred_flooded' : predicted, 'Ref_flooded' : ref, 'Pred_no_flood' : 1-predicted, 'Ref_no_flood' : 1-ref})

    # Add data in (Pred_flooded, Ref_flooded)
    f_cnf_matrix[0, 0] = df_results[['Pred_flooded', 'Ref_flooded']].min(axis=1).sum()
    # Add data in (Pred_flooded, Ref_no_flood)
    f_cnf_matrix[0, 1] = df_results[['Pred_flooded', 'Ref_no_flood']].min(axis=1).sum()
    # Add data in (Pred_no_flood, Ref_flooded)
    f_cnf_matrix[1, 0] = df_results[['Pred_no_flood', 'Ref_flooded']].min(axis=1).sum()
    # Add data in (Pred_no_flood, Ref_no_flood)
    f_cnf_matrix[1, 1] = df_results[['Pred_no_flood', 'Ref_no_flood']].min(axis=1).sum()

    return f_cnf_matrix