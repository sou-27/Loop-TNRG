import numpy as np
from scipy.sparse.linalg import cg
import environment


def loop_optimize(S,T, chi,niter, atol, rtol):
    cost_start = environment.cost_function(S,T)
    print("Starting cost = ",cost_start)
    for i in range(niter):
        #if i%5 == 0:
        #    S = condition_network(S,chi)
        #print("**",i,"**")
        for j in range(8):
            N = environment.create_N(S,j)
            W = environment.create_W(T, S, j)
            
            S[j] = environment.optimize_bond(np.copy(N),np.copy(W), S[j])

            #cost = cost_function(N,W,S[j],T)
            #print(cost)
            
            #if cost < atol or np.abs(cost - cost_prev)/cost_prev < rtol:
            #    print("Optimization ended at iteration : ",i)
            #    print("Breaking cost = ",cost)
            #    return S
    
    cost_final = environment.cost_function(S,T)
    print("Final cost = ",cost_final)
        
    return S