import numpy as np
from .hamiltonian import Ising_fusion_hamiltonian
from .network_ops import compress_layer, decompose_network, recombine_network, normalize_network
from .optimize import loop_optimize
from .cft_data import cft_data, spec_v, spec_h, reduced_cft_data


def tnrg(r,theta,repel,delta_t, chi_init, n_layers, chi, niterations, niter = 30, rtol = 1e-5, atol = 1e-10):

    T = Ising_fusion_hamiltonian(r,theta, repel, delta_t, chi_init)
    print("T created")
    T = compress_layer(T, n_layers, chi_init)
    Cs = []
    SD = []
    #norm, C, v, scaling_dims = virtual_transfer_matrix(T)
    norm, C, v, scaling_dims = cft_data(T)
    print("Speed of light = ",v)
    Cs.append(C)
    SD.append(scaling_dims)
    energy_v = []
    energy_h = []



    for i in range(niterations):
        print(i)
        print("-------")
        S = decompose_network(T, chi)
        S = loop_optimize(S,T,chi,niter,atol,rtol)
        T = recombine_network(S,i)
        k,T = normalize_network(T)
        if i%2 == 1:
            norm, C, v, scaling_dims = cft_data(T)
            #norm, C, v, scaling_dims = virtual_transfer_matrix(T)
            Cs.append(C)
            SD.append(scaling_dims)
            print("Speed of light = ",v)
            print("Central Charge = ",C)
           #print("First scaling dimension = ", scaling_dims[1])
        
        print("Normalization = ",k)
    scaling_dimensions = np.array(SD)
    CentralCharges = np.array(Cs)

    return CentralCharges, scaling_dimensions


def reduced_tnrg(r,theta,repel,delta_t, chi_init, n_layers, chi, niterations, niter = 20, rtol = 1e-5, atol = 1e-10):
    T = Ising_fusion_hamiltonian(r,theta, repel, delta_t, chi_init)
    print("T created")
    T = compress_layer(T, n_layers, chi_init)
    Cs = []
    SD = []
    
    eig_22 = spec_v(T)
    eig_22_tilde = spec_h(T)

    for i in range(niterations):
        print(i)
        print("==========")
        S = decompose_network(T, chi)
        S = loop_optimize(S,T,chi,niter,atol,rtol)
        T = recombine_network(S,i)
        if i%2 == 0:
            norm, C, v, scaling_dims = reduced_cft_data(T, eig_22, eig_22_tilde)
            Cs.append(C)
            SD.append(scaling_dims)
            print("Speed of light = ",v)
            print("Central Charge = ",C)
        
        k,T = normalize_network(T)

        if i%2 == 1:
            eig_22 = spec_v(T)
            eig_22_tilde = spec_h(T)
        
        print("Normalization = ",k)
    
    scaling_dimensions = np.array(SD)
    CentralCharges = np.array(Cs)

    return CentralCharges, scaling_dimensions
