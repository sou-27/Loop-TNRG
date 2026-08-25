import numpy as np
from ncon import ncon
from scipy.sparse.linalg import cg

def create_N(S,i):
    """ Create the tensor N_i used for Loop optimization.

    Parameters
    ---------
    S : list, length 8
    List of tensors in the tensor network after decomposing into rank-3 tensors
    i : int, site of tensor to be optimized

    Returns
    -------
    N : ndarray
    Tensor used in optimization procedure

    """
    local_tensors = []
    for k in range(i+1, i+8):
        q = ncon([np.conj(S[np.mod(k,8)]), S[np.mod(k,8)]],[[-1,1,-2],[-3,1,-4]])
        local_tensors.append(q)
    
    N = local_tensors[0]
    for k in range(1,len(local_tensors)):
        N = ncon([N, local_tensors[k]],[[-1,1,-3,2],[1,-2,2,-4]])
    
    d = S[i].shape[1]
    I = np.eye(d)
    N = ncon([N,I],[[-3,-1,-6,-4],[-2,-5]])
    return N




def create_W(T,S,i):
    """Create the tensor W_i used for Loop optimization.

    Parameters
    ----------
    T : list, length 4
    List of rank-4 tensors in the network

    S : list, length 4
    List of tensors in the network after decompositions into rank 3 tensors

    i : int
    Index of tensor site to be optimized

    Returns
    -------
    W : ndarray
    Tensor W used in optimization procedure

    """
    if i%2 == 0:
        wj = i//2
        local_tensors = []
        for k in range(wj+1, wj+4):
            q = ncon([np.conj(S[2*np.mod(k,4)]), np.conj(S[(2*np.mod(k,4))+1]), T[np.mod(k,4)]],[[-2,1,2],[2,3,-3],[-1,1,3,-4]])
            local_tensors.append(q)
        
        W = ncon([np.conj(S[i+1]), T[wj]],[[-3,1,-4],[-1,-2,1,-5]])
        for tensor in local_tensors:
            W = ncon([W,tensor],[[-1,-2,-3,2,1],[1,2,-4,-5]])
        
        W = ncon([W],[[1,-2,-3,-1,1]])
    else:
        wj = i//2
        local_tensors = []
        for k in range(wj+1, wj+4):
            q = ncon([np.conj(S[2*np.mod(k,4)]), np.conj(S[(2*np.mod(k,4))+1]), T[np.mod(k,4)]],[[-2,1,2],[2,3,-3],[-1,1,3,-4]])
            local_tensors.append(q)
        
        W = ncon([np.conj(S[i-1]), T[wj]],[[-2,1,-3],[-1,1,-4,-5]])
        for tensor in reversed(local_tensors):
            W = ncon([tensor, W],[[-1,-2,2,1],[1,2,-3,-4,-5]])
        
        W = ncon([W],[[1,-3,-1,-2,1]])

    
    return W

def cost_function(S, T):
    """ Calculates the cost functions 
    
    Parameters
    ----------
    S : List, length 8
    List of rank-3 tensors obtained after decmomposition of tensor network

    T : List, length 4
    List of rank-4 tensors making up the tensor network 
    """
    #Calulating <S|S>
    local_tensors = []
    for k in range(8):
        q = ncon([np.conj(S[k]), S[k]],[[-1,1,-2],[-3,1,-4]])
        local_tensors.append(q)
    ss = local_tensors[0]
    for k in range(1,len(local_tensors)):
        ss = ncon([ss, local_tensors[k]],[[-1,1,-3,2],[1,-2,2,-4]])
    
    ss = ncon([ss], [1,1,2,2])

    #Calculating <S|T>
    local_tensors = []
    for k in range(4):
        q = ncon([np.conj(S[2*k]), np.conj(S[(2*k)+1]), T[k]],[[-2,1,2],[2,3,-3],[-1,1,3,-4]])
        local_tensors.append(q)
    
    st = local_tensors[0]
    for k in range(1,len(local_tensors)):
        st = ncon([st,local_tensors[k]],[[-1,-2,2,1],[1,2,-3,-4]])
    
    st = ncon([st],[[1,2,2,1]])

    #Calculating <T|T>

    local_tensors = []
    for k in range(4):
        q = ncon([T[k],np.conj(T[k])],[[-1,1,2,-4],[-2,1,2,-3]])
        local_tensors.append(q)
    
    tt = local_tensors[0]

    for k in range(1,len(local_tensors)):
        tt = ncon([tt, local_tensors[k]],[[-1,-2,2,1],[1,2,-3,-4]])
    
    tt = ncon([tt],[1,2,2,1])

    fidelity = (st)**2/(ss*tt)
    cost = 1-fidelity
    return cost    

def optimize_bond(N,W, ss):
    """ Bond at site is optimized via a eigenvalue problem

    Parameters
    ----------
    N,W : ndarrays

    Parameters to be used in the optimization procedure

    ss: ndarray
    Initial guess for the conjugate gradient optimization procedure.

    Returns
    -------
    Optimized tensor S 

    """
    N_matrix = N.reshape(N.shape[0] * N.shape[1] * N.shape[2], N.shape[3] * N.shape[4] * N.shape[5])

    W_vector = W.reshape(W.shape[0] * W.shape[1] * W.shape[2])

    s, info = cg(N_matrix, W_vector, ss.reshape(ss.shape[0] * ss.shape[1] * ss.shape[2]))
    S = s.reshape(W.shape[0], W.shape[1], W.shape[2])

    return S