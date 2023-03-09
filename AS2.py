import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sv = pd.read_excel('sv.xlsx', sheet_name= 'Sheet1')

# a
# plt.plot(sv)
# plt.show()

# b
y = sv['GBPUSD'].values
mu = np.mean(y)
x = np.log((y-mu)**2)

# plt.plot(x)
# plt.show()

#KALMAN FILTER FUNCTION
def KF_LL(data, params, ml_flag):
    phi, sig_eta, omega = params

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
    F[0] = P[0] + sig_eta
    K = np.zeros(len(data))
    K[0] = P[0] / F[0]


    for t in range(0, len(data)):
        # Prediction error
        v[t] = data[t] - a[t] +1.27

        # Pred. err. variance
        F[t] = P[t] + H

        # Kalman gain
        K[t] = phi * P[t] / F[t]

        # Prediction step
        a[t+1] = phi * a[t] + K[t] * v[t] + omega
        P[t+1] = phi * P[t] * phi + sig_eta - K[t]**2 * F[t]
    
    a = a[1:]
    P = P[1:]    
    v = v[1:]
    F = F[1:]
    K = K[1:]
    n = len(v)

    LogL = - (n/2) * np.log(2 * np.pi) - 1/2 * np.sum(np.log(F) + F**(-1) * v**2)

    if ml_flag==1:
        return LogL
    else:
        return a,P,v,F,K,n

# Initial value
phi_ini = 0.9731 

def wrapper(phi):
    omega = (1 - phi) * (np.mean(x) + 1.27)
    sig_eta = (1 - phi**2) * (np.var(x) - (np.pi**2)/2)
    params = [phi, sig_eta, omega]
    return  -KF_LL(x, params, 1)

bounds = [(0, 1)]
opt = minimize(wrapper, method='Nelder-Mead', x0 = phi_ini, bounds = bounds)

phi = opt.x[0]
omega = (1 - phi) * (np.mean(x) + 1.27)
sig_eta = (1 - phi**2) * (np.var(x) - (np.pi**2)/2)
ml_params = [phi, omega, sig_eta]
print(f'phi: {phi}\nomega: {omega}\nsig_eta: {sig_eta}')


a,P,v,F,K,n = KF_LL(x, ml_params, 0)
plt.plot(x, color = 'grey')
plt.plot(a, color='red')
plt.show()

## d
#KALMAN SMOOTHER FUNCTION
def KS_LL(data, a_ini, P_ini, sig_e, sig_eta):

    v, P, F, a, K = KF_LL(data, a_ini, P_ini, sig_e, sig_eta)

    r =  np.zeros(len(df))
    r[-1] = 0

    N = np.zeros(len(df))
    N[-1] = 0
    
    a_hat = np.zeros(len(df))

    V = np.zeros(len(df))

    for t in range(len(df)-2, -1, -1):
        if np.isnan(data[t]):
            r[t] = r[t+1]
            N[t] = N[t+1]
        else:
            r[t] = F[t]**-1*v[t] + (1-K[t]) * r[t+1]
            N[t] = F[t]**-1 + (1-K[t])**2*N[t+1]

    for t in range(0, len(df)):
        a_hat[t] = a[t] + P[t] *r[t]
        V[t] = P[t] - P[t]**2*N[t]

    u_star, u, D, r_star = sm_obs_dist(F,v,K,r,N) 

    return (r, N, a_hat, V, u_star, u, D, r_star)


#This function is needed for plot 2.8
def sm_obs_dist(F,v,K,r,N):
    u = F**-1 * v - K*r
    D = F**-1 + K**2 * N
    u_star = D**(-1/2) * u
    r_star = N**(-1/2) * r
    
    return u_star, u, D, r_star
