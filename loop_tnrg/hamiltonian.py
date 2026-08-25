import numpy as np
from ncon import ncon
from scipy.linalg import expm
from .linalg_utils import trim_svd, trim_indices


def Ising_fusion_hamiltonian(r,theta, repel, delta_t, chi_init, stol = 1E-15):
    d = 3
    # |1> ---> 0
    # |\psi> ---> 1
    # |\sigma> ---> 2

    H = np.zeros((d**3, d**3))
    hi = {
        "000" : [[np.cos(theta),"020"]],
        "111" : [[np.cos(theta),"121"]],
        "002" : [[r,"002"],[np.sin(theta),"022"]],
        "112" : [[r,"112"],[np.sin(theta),"122"]],
        "020" : [[np.cos(theta), "000"]],
        "121" : [[np.cos(theta), "111"]],
        "200" : [[r,"200"],[np.sin(theta),"220"]],
        "211" : [[r,"211"],[np.sin(theta),"221"]],
        "022" : [[r,"022"],[np.sin(theta),"002"]],
        "122" : [[r,"122"],[np.sin(theta),"112"]],
        "202" : [[-np.cos(theta)/np.sqrt(2), "222"]],
        "212" : [[-np.cos(theta)/np.sqrt(2), "222"]],
        "220" : [[r,"220"], [np.sin(theta), "200"]],
        "221" : [[r,"221"], [np.sin(theta), "211"]],
        "222" : [[-np.cos(theta)/np.sqrt(2), "202"],[-np.cos(theta)/np.sqrt(2),"212"]]
    }

    states = [np.base_repr(k,3).zfill(3) for k in range(27)]
    for idx, state in enumerate(states):
        if state in hi:
            H_action = hi[state]
            for overlap, new_state in H_action:
                H[int(new_state,3), idx] = -overlap
        
    
    repel_states = ['101', '201', '010', '012', '100', '102', '011', '001', '110', '210']

    repel_term = np.zeros((d**3, d**3))

    for state in repel_states:
        idx = int(state,3)
        repel_term[idx,idx] += repel
    

    HH = H + repel_term

    if r > 0:
        shift = r * 1.8
        eng_shift = shift * np.eye(d**3)
        H = H + eng_shift
    U = expm(- delta_t * HH).reshape(d,d,d,d,d,d)

    #h = 0.5 * (U + U.transpose(3,4,5,0,1,2))
    #D,vv = SA.eigh(h.reshape(d**3, d**3))
    #S5 = (vv@np.diag(np.sqrt(D))).reshape(d,d,d, len(D))
    #S6 = np.conj(S5).transpose(3,0,1,2)

    u, s, v = trim_svd(U.reshape(U.shape[0] * U.shape[1] * U.shape[2], -1), chi_init, stol)
    S5 = (u @ np.diag(np.sqrt(s))).reshape(d,d,d,len(s))
    S6 = (np.diag(np.sqrt(s)) @ v).reshape(len(s),d,d,d)
    uu = U.transpose(0,3,1,4,2,5)

    #u, s, v = SA.svd(uu.reshape(uu.shape[0] * uu.shape[1], -1), full_matrices=False)
    u, s, v = trim_svd(uu.reshape(uu.shape[0] * uu.shape[1], -1), chi_init, stol)

    S1 = ( u @ np.diag(np.sqrt(s))).reshape(d,d,len(s))
    S2 = (np.diag(np.sqrt(s)) @ v).reshape(len(s),d,d,d,d)

    #u, s, v = SA.svd(uu.reshape(uu.shape[0] * uu.shape[1] * uu.shape[2] * uu.shape[3], -1), full_matrices=False)
    u, s, v = trim_svd(uu.reshape(uu.shape[0] * uu.shape[1] * uu.shape[2] * uu.shape[3], -1), chi_init, stol)
    S3 = ( u @ np.diag(np.sqrt(s))).reshape(d,d,d,d,len(s))
    S4 = (np.diag(np.sqrt(s)) @ v).reshape(len(s),d,d)

    T = ncon([S6,S4,S3,S2,S1,S5],[[-1,1,2,3],[-2,1,4], [2,5,3,6,-5], [-3,4,7,5,8], [6,9,-6], [7,8,9,-4]])

    T = T.reshape(T.shape[0], T.shape[1] * T.shape[2], T.shape[3], T.shape[4] * T.shape[5])
    print('Shape before trimming : ',T.shape)
    T = trim_indices(T, chi_init)
    print('Shape after trimming : ',T.shape)
    return T