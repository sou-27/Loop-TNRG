import numpy as np
from ncon import ncon
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs


def cft_data(T):
    """Calculates central charge and scaling dimensions from tensor network T. Uses naive implementation using tensors from current network.
    
    Parameters
    ---------
    T : list of ndarrays, rank-4 tensors making up the tensor network

    Returns
    -------
    eps : float, normalization used in computing cft data
    c : float, central charge
    v :  float, speed of light
    scaling_dims : array, array containing scaling dimensions.
    """
    def T_22(v):
        v = v.reshape(T[0].shape[2], T[1].shape[1])
        result = ncon([v,T[0],T[1],T[2],T[3]],[[1, 2], [4,8,-1,6], [6,-2,8,5], [5,7,2,3],[3,1,7,4]], [1,2,3,7,4,5,6,8])
        result = result.reshape(T[0].shape[2] * T[1].shape[1])
        return result
    
    def T_22_tilde(v):
        v = v.reshape(T[3].shape[2], T[0].shape[1])
        result = ncon([v,T[0],T[1],T[2],T[3]],[[8,7],[1,-2,2,3], [3,5,7,4], [4,8,5,6], [6,2,-1,1]], [7,8,4,5,6,3,1,2])
        result = result.reshape(T[3].shape[2] * T[0].shape[1])
        return result
    
    def T_42(v):
        v = v.reshape(T[0].shape[2], T[1].shape[1], T[0].shape[2], T[1].shape[1])
        result = ncon([v,T[0],T[1],T[2],T[3],T[0],T[1],T[2],T[3]], [[1,2,3,4], [9,16,-1,13], [13,-2,14,10], [10,6,2,5], [5,1,8,9], [11,14,-3,15], [15,-4,16,12], [12,8,4,7],[7,3,6,11]], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        result = result.reshape((T[0].shape[2] * T[1].shape[1])**2)
        return result
    
    L_22 = LinearOperator((T[0].shape[2] * T[1].shape[1], T[0].shape[2] * T[1].shape[1]), matvec = T_22)
    L_22_tilde = LinearOperator((T[0].shape[1] * T[3].shape[2], T[0].shape[1] * T[3].shape[2]), matvec = T_22_tilde)
    L_42 = LinearOperator(((T[0].shape[2] * T[1].shape[1])**2, (T[0].shape[2] * T[1].shape[1])**2), matvec = T_42)

    eig_22 = -np.sort(-eigs(L_22, k = 10,which='LM', return_eigenvectors=False))
    eig_22_tilde = -np.sort(-eigs(L_22_tilde, k = 1, which = 'LM', return_eigenvectors=False))
    eig_42 = -np.sort(-eigs(L_42, k = 1,which = 'LM', return_eigenvectors=False))
    eps = (-1/6) * np.log(eig_42[0]/np.sqrt(eig_22[0]))

    eig_42 = eig_42 * np.exp(8 * eps)
    eig_22 = eig_22 * np.exp(4 * eps)
    eig_22_tilde = eig_22_tilde * np.exp(4 * eps)

    
    v = np.sqrt(np.log(eig_22[0])/np.log(eig_22_tilde[0]))

    c = (6/np.pi) * np.log(eig_22[0])/v

    scaling_dims = []
    for i in range(len(eig_22)-1):
        scaling_dims.append((1/(2 * np.pi * v))*np.log(eig_22[0]/eig_22[i]))
    

    return eps,c, v, scaling_dims



    

def spec_v(T):
    """Diagonalizes specific tensor required for computing conformal data. 
    
    Parameters
    ----------
    T : list of ndarrays, rank-4 tensors makig up the tensor network.
    
    Returns
    -------
    eig_22: list of floats, desired eigenvalues.
    """
    def T_22(v):
        v = v.reshape(T[0].shape[2], T[1].shape[1])
        result = ncon([v,T[0],T[1],T[2],T[3]],[[1, 2], [4,8,-1,6], [6,-2,8,5], [5,7,2,3],[3,1,7,4]], [1,2,3,7,4,5,6,8])
        result = result.reshape(T[0].shape[2] * T[1].shape[1])
        return result
    
    L_22 = LinearOperator((T[0].shape[2] * T[1].shape[1], T[0].shape[2] * T[1].shape[1]), matvec = T_22)
    eig_22 = -np.sort(-eigs(L_22, k = 10,which='LM', return_eigenvectors=False))
    
    return eig_22

    

def spec_h(T):
    """Diagonalizes specific tensor required for computing conformal data. 
        
        Parameters
        ----------
        T : list of ndarrays, rank-4 tensors makig up the tensor network.
        
        Returns
        -------
        eig_22_tilde: list of floats, desired eigenvalues.
        """
    def T_22_tilde(v):
        v = v.reshape(T[3].shape[2], T[0].shape[1])
        result = ncon([v,T[0],T[1],T[2],T[3]],[[8,7],[1,-2,2,3], [3,5,7,4], [4,8,5,6], [6,2,-1,1]], [7,8,4,5,6,3,1,2])
        result = result.reshape(T[3].shape[2] * T[0].shape[1])
        return result

    L_22_tilde = LinearOperator((T[0].shape[1] * T[3].shape[2], T[0].shape[1] * T[3].shape[2]), matvec = T_22_tilde)

    eig_22_tilde = -np.sort(-eigs(L_22_tilde, k = 1, which = 'LM', return_eigenvectors=False))

    return eig_22_tilde

    
def reduced_cft_data(T, eig_22, eig_22_tilde):
    """Calculates central charge and scaling dimensions from tensor network T. Uses optimized implementation using data from previous RG step.
        
        Parameters
        ---------
        T : list of ndarrays, rank-4 tensors making up the tensor network
    
        Returns
        -------
        eps : float, normalization used in computing cft data
        c : float, central charge
        v :  float, speed of light
        scaling_dims : array, array containing scaling dimensions.
        """

    def A(v):
        v = v.reshape(T[3].shape[1], T[3].shape[0], T[1].shape[3], T[1].shape[2])
        result = ncon([v, T[0], T[1], T[2], T[3]], [[1,2,3,4], [6,-2,-3,7], [7,8,4,3], [-1,5,8,-4],[2,1,5,6]], [1,2,3,4,5,6,7,8])
        result = result.reshape(-1)
        return result
    

    L = LinearOperator((T[3].shape[1] * T[3].shape[0] * T[1].shape[3] * T[1].shape[2], T[0].shape[1] * T[0].shape[2] * T[2].shape[3] * T[2].shape[0]), matvec = A)


    eig_42 = -np.sort(-eigs(L, k = 20,which = 'LM', return_eigenvectors=False))
    eps = (-1/6) * np.log(eig_42[0]/np.sqrt(eig_22[0]))

    eig_42 = eig_42 * np.exp(8 * eps)
    eig_22 = eig_22 * np.exp(4 * eps)
    eig_22_tilde = eig_22_tilde * np.exp(4 * eps)

    
    v = np.sqrt(np.log(eig_22[0])/np.log(eig_22_tilde[0]))

    c = (12/np.pi) * np.log(eig_42[0])/v

    scaling_dims = []
    for i in range(len(eig_22)-1):
        scaling_dims.append((1/(np.pi * v))*np.log(eig_42[0]/eig_42[i]))
    

    return eps,c, v, scaling_dims