import numpy as np
from ncon import ncon
from .linalg_utils import trim_indices
import scipy.linalg as SA

def check_isotropic(T):
    """Check if tensor network is isotropic by checking singular values in all four directions
    
    Parameters
    ----------
    T : ndarray, rank-4 tensor making up the tensor network.


    Returns
    -------
    NIL
    """
    A = np.copy(T)
    u, s, v = np.linalg.svd(A.reshape(A.shape[0],-1))
    print(s/ np.max(s))

    A = A.transpose(3,0,1,2)

    u, s, v = np.linalg.svd(A.reshape(A.shape[0],-1))
    print(s/np.max(s))

    A = A.transpose(3,0,1,2)

    u, s, v = np.linalg.svd(A.reshape(A.shape[0],-1))
    print(s/np.max(s))

    A = A.transpose(3,0,1,2)

    u, s, v = np.linalg.svd(A.reshape(A.shape[0],-1))
    print(s/np.max(s))



def compress_layer(T, n_layers, chi):
    """Compresses n_layers vertical layers (time direction) of the tensor network, Needed to precondition the tensor network for RG.

    Parameters
    ----------

    T : ndarray, rank-4 tensor making up the tensor network.
    n_layers : int, number of layers to be compressed
    chi : int, maxmimum bond dimension to tensors to be kept after truncation
    
    Returns
    -------
    [T1,T2,T3,T4] : List of 4 rank-4 tensors that make up the network after preconditioning.
    """
    for i in range(n_layers):
        T = ncon([T,T], [[-1,-2,1,-6],[1,-3,-4,-5]])
        T = T.transpose(0,1,2,3,5,4)
        T = T.reshape(T.shape[0], T.shape[1] * T.shape[2], T.shape[3], T.shape[4] * T.shape[5])
        
        T = trim_indices(T,chi)
        print('Shape : ',T.shape)
        #check_isotropic(T)
        print("============")

    T1 = T
    T2 = T.transpose(1,2,3,0)

    T3 = T1.transpose(2,3,0,1)
    T4 = T2.transpose(2,3,0,1)
    
    

    return [T1,T2,T3,T4]


def decompose_network(Ts,chi):
    """Decompose tensor network by taking Ti --> Si . Si+1, where T is rank-4 and S are rank 3 tensors.

    Parameters
    ----------

    Ts: List of ndarrays, rank-4 tensors making up the initial network.
    chi: int, maximum bond dimension of tensors to be kept after truncation.


    Returns
    -------
    decomposed_network: list of ndarrays, list of rank-3 tensors obtained from decomposition of T.
    
    """
    decomposed_network = []
    for i in range(4):
        u,s,v = SA.svd(Ts[i].reshape(Ts[i].shape[0]*Ts[i].shape[1], Ts[i].shape[2] * Ts[i].shape[3]), full_matrices = False)
        chitemp = min(len(s),chi)
        u = u[:,:chitemp]
        v = v[:chitemp, :]
        s = s[:chitemp]

        S1 = (u @ np.sqrt(np.diag(s))).reshape(Ts[i].shape[0], Ts[i].shape[1], chitemp)
        S2 = (np.sqrt(np.diag(s)) @ v).reshape(chitemp, Ts[i].shape[2], Ts[i].shape[3])
        decomposed_network.append(S1)
        decomposed_network.append(S2)

    return decomposed_network



def recombine_network(S, iteration):
    """Recombines the rankl-3 tensors S into a new network of rank-4 tensors T.

    Parameters
    ----------
    S : List of ndarrays, rank-3 tensors to be combined into the new Ts.
    iteration : Number of the iteration. We need to recombine the tensors in a specific order for iteration%2, in order to 
    preserve the correct spacetime geometry.
 
    Returns
    -------
    [T1,T2,T3,T4] : List of ndarrays, rank-4 tensors making up the tensor network after recombination.
    """

    T0 = ncon([S[1], S[2], S[6], S[5]],[[-1,1,2],[2,3,-4],[4,1,-2],[-3,3,4]],[2,4,1,3])
    T1 = ncon([S[3],S[4], S[0], S[7]],[[-1,2,1],[1,4,-4],[3,2,-2],[-3,4,3]],[1,3,2,4])
    T2 = T0.transpose(2,3,0,1)
    T3 = T1.transpose(2,3,0,1)

    if iteration %2 == 0:
        return [T0,T1,T2,T3]
    else:
        return [T1,T2,T3,T0]



def normalize_network(network):
    """Normalizes the tensor network
    
    Parameters
    ----------
    network: List of ndarrays, rank-4 tensors making up the network.
    
    Returns
    -------
    g : float, normalization factor
    network: List of ndarrays, tensor network after normalization
    """
    g = ncon([network[0], network[1], network[2], network[3]], [[1,2,3,4], [4,5,2,6], [6,7,5,8], [8,3,7,1]])
    for i in range(len(network)):
        network[i]  = network[i]/(g**(1/4))
        
    return g, network
