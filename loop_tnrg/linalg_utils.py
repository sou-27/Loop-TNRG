import numpy as np
from ncon import ncon
import scipy.linalg as SA
import copy

def unique_qr(A):
    Q, R = np.linalg.qr(A, mode = 'reduced')
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]
    return Q, R


def lq(A):
    q_qr,R_qr = unique_qr(A.T)
    q = q_qr.T
    L = R_qr.T

    return L,q

def trim_svd(A, chi, stol, print_err = False):
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

def condition_network(S,chi, stol=1e-7):
    S_cond = copy.deepcopy(S)
    S_cond.append(S[0])
    
    q,R = unique_qr(S_cond[0].reshape(S_cond[0].shape[0] * S_cond[0].shape[1], S_cond[0].shape[2]))
    Rs = [R]

    for i in range(1,len(S_cond)):
        q,R = unique_qr(ncon([Rs[-1],S_cond[i]],[[-1,1],[1,-2,-3]]).reshape(-1,S_cond[i].shape[2]))
        Rs.append(R)
    
    L, q = lq(S_cond[-1].reshape(S_cond[-1].shape[0], -1))
    Ls = [L]

    for i in range(-2,-(len(S_cond)+1),-1):
        L, q = lq(ncon([S_cond[i], Ls[0]],[[-1,-2,1],[1,-3]]).reshape(S_cond[i].shape[0],-1))
        Ls.insert(0,L)
    
    Rs.pop(-1)
    Ls.pop(0)

    P_as = []
    P_bs = []

    for i in range(len(S_cond)-1):
        R = Rs[i]
        L = Rs[i]

        u,s,vdag = SA.svd(R @ L, full_matrices=False)

        s_trim = s * (s > stol) + stol * (s < stol)

        chitemp = min(len(s_trim), chi)
        u = u[:,:chitemp]
        vdag = vdag[:chitemp, :]
        s_trim = s_trim[:chitemp]

        Pa = L @ np.conj(vdag).T @ np.diag(1/np.sqrt(s))
        Pb = np.diag(1/np.sqrt(s)) @ np.conj(u).T @ R

        P_as.append(Pa)
        P_bs.append(Pb)
    
    newS = []

    S_tilde = ncon([P_bs[-1], S[0], P_as[0]], [[-1,1],[1,-2,2],[2,-3]])
    newS.append(S_tilde)

    for i in range(1,len(S)):
        S_tilde = ncon([P_bs[i-1], S[i], P_as[i]], [[-1,1],[1,-2,2],[2,-3]])
        newS.append(S_tilde)
    
    return newS

def trim_indices(T, chi, stol = 1E-9, niter = 50, tol = 1E-5):

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