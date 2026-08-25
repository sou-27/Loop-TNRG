import numpy as np
from scipy.sparse.linalg import cg
from .environment import *


def loop_optimize(S,T, chi,niter, atol, rtol):
    cost_start = cost_function(S,T)
    print("Starting cost = ",cost_start)
    for i in range(niter):
        #if i%5 == 0:
        #    S = condition_network(S,chi)
        #print("**",i,"**")
        for j in range(8):
            N = create_N(S,j)
            W = create_W(T, S, j)
            
            S[j] = optimize_bond(np.copy(N),np.copy(W), S[j])

            #cost = cost_function(N,W,S[j],T)
            #print(cost)
            
            #if cost < atol or np.abs(cost - cost_prev)/cost_prev < rtol:
            #    print("Optimization ended at iteration : ",i)
            #    print("Breaking cost = ",cost)
            #    return S
    
    cost_final = cost_function(S,T)
    print("Final cost = ",cost_final)
        
    return S