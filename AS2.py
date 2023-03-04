import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sv = pd.read_excel('sv.xlsx', sheet_name= 'Sheet1')

# a
plt.plot(sv)
plt.show()
print(np.mean(sv), np.std(sv))

# b
y = sv['GBPUSD'].values
mu = np.mean(y)
x = np.log((y-mu)**2)

plt.plot(x)
plt.show()

# c
# #KALMAN FILTER FUNCTION
# def KF_LL(data, a_ini, P_ini, sig_e, sig_eta):
#     n = len(data)

#     a =  np.zeros(n)
#     a[0] = a_ini

#     v = np.zeros(n)
#     v[0] = data[0] - a[0]

#     P = np.zeros(n)
#     P[0] = P_ini

#     F = np.zeros(n)
#     F[0] = P[0] + sig_e
#     K = np.zeros(n)
#     K[0] = P[0] / F[0]

#     for t in range(0, n-1):
#         if np.isnan(data[t]):
#             v[t+1] = v[t]
#             F[t] = P[t] + sig_e
#             K[t] = P[t]/F[t]
#             P[t+1] = P[t] + sig_eta 
#             a[t+1] = a[t]
#         else:
#             v[t] = data[t] - a[t]
#             F[t] = P[t] + sig_e
#             K[t] = P[t] / F[t]
#             a[t+1] = a[t] + K[t] * v[t]
#             P[t+1] = K[t] * sig_e + sig_eta
            
#     return (v, P, F, a, K)

# #KALMAN SMOOTHER FUNCTION
# def KS_LL(data, a_ini, P_ini, sig_e, sig_eta):

#     v, P, F, a, K = KF_LL(data, a_ini, P_ini, sig_e, sig_eta)

#     r =  np.zeros(len(df))
#     r[-1] = 0

#     N = np.zeros(len(df))
#     N[-1] = 0
    
#     a_hat = np.zeros(len(df))

#     V = np.zeros(len(df))

#     for t in range(len(df)-2, -1, -1):
#         if np.isnan(data[t]):
#             r[t] = r[t+1]
#             N[t] = N[t+1]
#         else:
#             r[t] = F[t]**-1*v[t] + (1-K[t]) * r[t+1]
#             N[t] = F[t]**-1 + (1-K[t])**2*N[t+1]

#     for t in range(0, len(df)):
#         a_hat[t] = a[t] + P[t] *r[t]
#         V[t] = P[t] - P[t]**2*N[t]

#     u_star, u, D, r_star = sm_obs_dist(F,v,K,r,N) 

#     return (r, N, a_hat, V, u_star, u, D, r_star)

a_ini = 0
P_ini = 10**7
sig_e = 15099
sig_eta = 1469.1

#MAXIMUM LIKELIHOOD FUNCTIONS
def KF_LL_ML(data, params):
    sig_eta, phi, omega = params
    a_ini = 0
    P_ini = 10**7
    H = (np.pi**2)/2

    a =  np.zeros(len(data)+1)
    a[0] = a_ini

    v = np.zeros(len(data))
    v[0] = data[0] - a[0]

    P = np.zeros(len(data)+1)
    P[0] = P_ini

    F = np.zeros(len(data))
    F[0] = P[0] + sig_e
    K = np.zeros(len(data))
    K[0] = P[0] / F[0]


    for t in range(1, len(data)):
        # Prediction error
        v[t] = data[t] - a[t]

        # Pred. err. variance
        F[t] = P[t] + H

        # Kalman gain
        K[t] = phi * P[t] / F[t]

        # Filtering step
        a[t] = a[t] + (v[t]/F[t]) * P[t] 
        P[t] = P[t] - P[t]**2 / F[t]

        # Prediction step
        a[t+1] = phi * a[t] + K[t] * v[t] + omega
        P[t+1] = phi * P[t] * phi + sig_eta - K[t]**2 * F[t]

    a = a[1:]
    P = P[1:]    
    v = v[1:]
    F = F[1:]
    n = len(v)

    LogL = - (n/2) * np.log(2 * np.pi) - 1/2 * np.sum(np.log(F) + F**(-1) * v**2)
    
    return LogL

# Initial values
phi_ini = 0.9950 #np.cov(y[1:], y[:-1])[0][1]/(np.var(y[1:])- np.pi**2/2)
omega_ini = (1 - phi_ini) * (np.mean(y) + 1.27)
sig_eta = (1 - phi_ini**2) * (np.var(y) - (np.pi**2)/2)

params = [phi_ini, omega_ini, sig_eta]
LogL = KF_LL_ML(y, params)

def wrapper(phi):
    omega = (1 - phi) * (np.mean(y) + 1.27)
    sig_eta = (1 - phi**2) * (np.var(y) - (np.pi**2)/2)
    params = [phi, omega, sig_eta]
    return  KF_LL_ML(y, params)

opt = minimize(wrapper, method='nelder-mead', x0 = phi_ini, bounds = ((0,1)))


