import numpy as np
from ncon import ncon
from linalg_utils import trim_indices
import scipy.linalg as SA

def check_isotropic(T):
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

    T0 = ncon([S[1], S[2], S[6], S[5]],[[-1,1,2],[2,3,-4],[4,1,-2],[-3,3,4]],[2,4,1,3])
    T1 = ncon([S[3],S[4], S[0], S[7]],[[-1,2,1],[1,4,-4],[3,2,-2],[-3,4,3]],[1,3,2,4])
    T2 = T0.transpose(2,3,0,1)
    T3 = T1.transpose(2,3,0,1)

    if iteration %2 == 0:
        return [T0,T1,T2,T3]
    else:
        return [T1,T2,T3,T0]



def normalize_network(network):
    g = ncon([network[0], network[1], network[2], network[3]], [[1,2,3,4], [4,5,2,6], [6,7,5,8], [8,3,7,1]])
    for i in range(len(network)):
        network[i]  = network[i]/(g**(1/4))
        
    return g, network
