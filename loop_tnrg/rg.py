import numpy as np
from .hamiltonian import Ising_fusion_hamiltonian
from .network_ops import compress_layer, decompose_network, recombine_network, normalize_network
from .optimize import loop_optimize
from .cft_data import cft_data, spec_v, spec_h, reduced_cft_data


def tnrg(r,theta,repel,delta_t, chi_init, n_layers, chi, niterations, niter = 30):
    """Performs Loop-TNRG. Naive implementation using tensors are step i to calculate the cft data.
    
    Parameters
    ----------
    r : float, parameter in hamiltonian
    theta : float, parameter in hamiltonian
    repel : float, parameter in hamiltonian
    delta_t : float, parameter in euclidean path integral
    chi_init : int, maximumg bond dimensions to initialize tensor nework in
    n_layers : int, number of layers of network to be compressed for preconditioning
    chi : int, maximum bond dimenions of tensors to be kept during RG iterations
    niterations : int, numeber of iterations of RG to be done
    niter : number of passes of optimization to be done in Loop-optimization step
    
    Returns
    -------
    CentralCharges : ndarray, array conaining central charges calculated during RG steps
    scaling_dimensions : ndarray, array containing scaling dimensions calculated during RG steps
    """

    T = Ising_fusion_hamiltonian(r,theta, repel, delta_t, chi_init)
    print("T created")
    T = compress_layer(T, n_layers, chi_init)
    Cs = []
    SD = []
    norm, C, v, scaling_dims = cft_data(T)
    print("Speed of light = ",v)
    Cs.append(C)
    SD.append(scaling_dims)



    for i in range(niterations):
        print(i)
        print("-------")
        S = decompose_network(T, chi)
        S = loop_optimize(S,T,niter)
        T = recombine_network(S,i)
        k,T = normalize_network(T)
        if i%2 == 1:
            norm, C, v, scaling_dims = cft_data(T)
            Cs.append(C)
            SD.append(scaling_dims)
            print("Speed of light = ",v)
            print("Central Charge = ",C)
        
        print("Normalization = ",k)
    scaling_dimensions = np.array(SD)
    CentralCharges = np.array(Cs)

    return CentralCharges, scaling_dimensions


def reduced_tnrg(r,theta,repel,delta_t, chi_init, n_layers, chi, niterations, niter = 20):
    """Performs Loop-TNRG. Optimized implenentation using tensors from previous time step to calculate conformal data.
        
        Parameters
        ----------
        r : float, parameter in hamiltonian
        theta : float, parameter in hamiltonian
        repel : float, parameter in hamiltonian
        delta_t : float, parameter in euclidean path integral
        chi_init : int, maximumg bond dimensions to initialize tensor nework in
        n_layers : int, number of layers of network to be compressed for preconditioning
        chi : int, maximum bond dimenions of tensors to be kept during RG iterations
        niterations : int, numeber of iterations of RG to be done
        niter : number of passes of optimization to be done in Loop-optimization step
        
        Returns
        -------
        CentralCharges : ndarray, array conaining central charges calculated during RG steps
        scaling_dimensions : ndarray, array containing scaling dimensions calculated during RG steps
        """
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
        S = loop_optimize(S,T,niter)
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
