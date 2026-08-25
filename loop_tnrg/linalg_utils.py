import numpy as np
from ncon import ncon
import scipy.linalg as SA
import copy

def unique_qr(A):
    """ Performs a QR decomposition of the matrix A. The inbuilt qr module of numpy does not return a unique decomposition. This module fixes the gauge.

    Parameters
    ----------
    A : matrix
    Matrix to be decomposed

    Returns
    -------
    Q, R : matrices such that A = Q @ R, where Q is orthogonal and R is upper triangular.
    """
    Q, R = np.linalg.qr(A, mode = 'reduced')
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]
    return Q, R


def lq(A):
    """Performs LQ decompositon of matrix A.
    
    Parameters
    ----------
    A : matrix
    Matrix to be decomposed


    Returns
    -------
    L,Q : matrices such that A = L @ Q, where Q is orthogonal and L is lower triangular.
    
    """
    q_qr,R_qr = unique_qr(A.T)
    q = q_qr.T
    L = R_qr.T

    return L,q

def trim_svd(A, chi, stol, print_err = False):
    """ Performs a truncated SVD for the matrix A.

    Parameters
    ----------
    A : matrix, matrix to be decomposed.
    chi : int, maximum bond dimension to be truncated to.
    stol : float, tolerance for truncations. Singular values greater than stol are kept.
    print_err : boolean, if true, print the truncation error.

    Returns
    -------
    U,s,V : matrices such that A = U @ s @ V. The matrix dimensions are appropriately truncated according to chi and stol.
    """
    U, s, V = SA.svd(A, full_matrices=False)


    chitol = ([i for i in range(len(s)) if s[i] > stol][-1])

    chitemp = min(chitol+1, chi, len(s))

    U = U[:,:chitemp]
    s = s[:chitemp]
    V = V[:chitemp,:]

    s_err = s[chitemp:]
    if print_err:
        print("Truncation error = ",str(np.linalg.norm(s_err)))

    return U, s, V


def trim_indices(T, chi, stol = 1E-9, niter = 50, tol = 1E-5):
    """ Trim the internal indices of a layer of tensor network by treating it as an iMPS.

    Parameters
    ----------
    T : ndarray, tensor making up the network
    chi : int, maximum bond dimension to be truncated to
    stol : float, tolerance for svd truncation
    niter : int, maximum number of passes in trying to find appropriate L/R.
    tol : float, If L/R is converged up to tol, we stop.

    Returns
    -------
    T : ndarray, tensor with trimmed indices.    
    
    
    """
    q,L_prev = unique_qr(T.reshape(-1, T.shape[3]))
    L_prev = L_prev / np.max(np.diag(L_prev))

    flag_L = False

    for i in range(niter):
        A = ncon([L_prev, T],[[-2,1],[-1,1,-3,-4]])
        q, L =unique_qr(A.reshape(-1, A.shape[3]))
        
        L = L / np.max(np.diag(L))

        if np.linalg.norm(L - L_prev) < tol:
            flag_L = True
            break

        L_prev = L

    A = T.transpose(1,0,3,2)
    R_prev,q = lq(A.reshape((A.shape[0], -1)))
    R_prev = R_prev / np.max(np.diag(R_prev))
    flag_R = False
    
    for i in range(niter):
        XX = ncon([A, R_prev],[[-1,-2,1,-4],[1,-3]])
        R,q = lq(XX.reshape((XX.shape[0], -1)))
        R = R / np.max(np.diag(R))

        if np.linalg.norm(R - R_prev) < tol:
            flag_R = True
            break

        R_prev = R
    

    if flag_R == False or flag_L == False:
        print("Not converged!")


    u, s, v = SA.svd(L @ R, full_matrices=False)
    chitol = ([i for i in range(len(s)) if s[i] > stol][-1] // 4) * 4
    chitemp = min(len(s), chi, chitol)

    u = u[:,:chitemp]
    s = s[:chitemp]
    v = v[:chitemp, :]

    #s_trim = s * (s > stol) + stol * (s < stol)
    s_trim = s
    Pr = R @ (v.T @ np.diag(np.sqrt(1/s_trim)))
    Pl = np.diag(np.sqrt(1/s_trim)) @ (u.T @ L)

    T = ncon([Pl, T, Pr], [[-2,1],[-1,1,-3,2],[2,-4]])

    
    return T