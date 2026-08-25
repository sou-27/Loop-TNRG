import numpy as np
from scipy.sparse.linalg import cg
from .environment import *


def loop_optimize(S,T,niter):
    """Loop optimize the network
    
    Parameters
    ----------
    S : List of ndarrays, rank-3 tensors obtained after decomposition of original network
    T : List of ndarrays, rank-4 tensors making up the original network
    niter : int, number of passes in optimization procedure
    
    Returns
    -------
    S : List of ndarray, rank-3 tensors obtained after optimization
    """
    cost_start = cost_function(S,T)
    print("Starting cost = ",cost_start)
    for i in range(niter):
  
        for j in range(8):
            N = create_N(S,j)
            W = create_W(T, S, j)
            
            S[j] = optimize_bond(np.copy(N),np.copy(W), S[j])
    
    cost_final = cost_function(S,T)
    print("Final cost = ",cost_final)
        
    return S