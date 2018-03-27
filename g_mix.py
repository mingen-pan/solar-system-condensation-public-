#G_mix function set
import numpy as np
import math

def ideal_solution_G_mix(X,T, **kwargs):
    # X should have the same length as endmember
    return np.sum(X*np.log(X))*8.314*T

def ideal_solution_G_mix_prime(X,T, **kwargs):
    #the derivative(s) of G_mix
    # X should has the same length as endmember
    return np.log(X)[:-1]*8.314*T - math.log(X[-1])
    
def Gibbs_energy_fit(x, a, b, c):
    return a + b*x + c/(x*x)

def G_melilite_mix(X, T):
    #X[0] is geh, X[1] is ak
    W_geh = -24.288 #5.805 kcal
    W_ak = -0.5021 #0.12 kcal
    H_mix = X[0]*X[1]**2*W_geh + X[1]*X[0]**2*W_ak
    # the unit of S_mis is J/k, 1.38629 = ln(4)
    S_mix = -8.314*(np.sum(X*np.log(X)) + X[0]*math.log((X[0]+2)/2) + (2-X[0])*math.log((2-X[0])/2) + 1.38629*X[0])
    return H_mix*1e3 - T*S_mix

def G_melilite_mix_prime(X, T):
    #X[0] is geh, X[1] is ak
    W_geh = -24.288 #5.805 kcal
    W_ak = -0.5021 #0.12 kcal
    # d means derivition
    H_mix_d = 3*(W_geh - W_ak)*X[0]**2 + (-4*W_geh + 2*W_ak)*X[0] + W_geh
    # the unit of S_mis is J/k, 1.38629 = ln(4)
    S_mix_d = -8.314*( math.log(X[0]*(X[0]+2)*2/((X[1])*2*(2-X[0]))) + X[0]/(X[0]+2) - 1 + 1.38629)  
    return H_mix_d*1e3 - T*S_mix_d