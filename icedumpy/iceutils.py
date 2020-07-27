import numpy as np

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