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
# Z_t = 1
# d_t = 0
# e_t = u_t
# a_t = h_t
# T_t = phi
# c_t = omega (constant)
# R_t = 1

# H_t = Var[log e^2]
# Q_t = sig_eta^2

#KALMAN FILTER FUNCTION
def KF_LL(y, params, ml_flag):
    phi, sig_eta, omega = params

    Z = 1
    d = -1.27
    T = phi
    Q = sig_eta
    R = 1
    
    a_ini = omega/(1-phi)
    P_ini = sig_eta/(1-phi**2)
    H = (np.pi**2)/2
    c = omega
    a =  np.zeros(len(y)+1)
    a[0] = a_ini
    P = np.zeros(len(y)+1)
    P[0] = P_ini

    v = np.zeros(len(y))
    F = np.zeros(len(y))
    K = np.zeros(len(y))


    for t in range(0, len(y)):
        # Prediction error
        v[t] = y[t] - Z * a[t] - d

        # Pred. err. variance
        F[t] = Z*P[t]*Z + H

        # Kalman gain
        K[t] = T * P[t] * Z / F[t]

        # Prediction step
        a[t+1] = T * a[t] + K[t] * v[t] + c
        P[t+1] = T * P[t] * T + R*Q*R - K[t] * F[t] * K[t]
    
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

# Initial values
phi_ini = 0.9731 #np.cov(y[1:], y[:-1])[0][1]/(np.var(y[1:])- np.pi**2/2)
omega_ini =  (1 - phi_ini) * (np.mean(x) + 1.27) 
sig_eta_ini = (1 - phi_ini**2) * (np.var(x) - (np.pi**2)/2)

def wrapper(phi):
    omega = (1 - phi) * (np.mean(x) + 1.27)
    sig_eta = (1 - phi**2) * (np.var(x) - (np.pi**2)/2)
    params = [phi, sig_eta, omega]
    return  -KF_LL(x, params, 1)

bounds = [(0,1), (0, None), (None, None)]
opt = minimize(lambda params: -KF_LL(x,params, 1), method='Nelder-Mead', bounds = bounds, x0 = [phi_ini, sig_eta_ini, omega_ini], options= {'maxiter': 1e10, 'maxfev': 100000})#KALMAN SMOOTHER FUNCTION

## d

phi = opt.x[0]
omega = opt.x[2]
sig_eta = opt.x[1]
# omega = (1 - phi) * (np.mean(x) + 1.27)
# sig_eta = (1 - phi**2) * (np.var(x) - (np.pi**2)/2)
ml_params = [phi, sig_eta, omega]
print(f'phi: {phi}\nomega: {omega}\nsig_eta: {sig_eta}')


a,P,v,F,K,n = KF_LL(x, ml_params, 0)
plt.plot(x, color = 'grey')
plt.plot(a, color='red')
plt.show()

def KS_LL(data, v, P, F, a, K, phi):
    r =  np.zeros(len(data))
    r[-1] = 0

    N = np.zeros(len(data))
    N[-1] = 0
    
    a_hat = np.zeros(len(data))

    V = np.zeros(len(data))
    L = phi - K 
    for t in range(len(data)-2, -1, -1):
        r[t] = F[t]**-1*v[t] + L[t] * r[t+1] 
        N[t] = F[t]**-1 + L[t]**2*N[t+1]

    for t in range(0, len(data)):
        a_hat[t] = a[t] + P[t] *r[t]
        V[t] = P[t] - P[t]**2*N[t]


    return (r, N, a_hat, V)

r, N, a_hat, V = KS_LL(x, v, P, F, a, K, phi)
plt.plot(x, color = 'grey')
plt.plot(a, color='red')
plt.plot(a_hat, color = 'blue')
plt.show()

## e

print('\n')

realized_volatility = pd.read_csv('realized_volatility.csv')
realized_volatility = realized_volatility[realized_volatility['Symbol'] == '.SPX']
realized_volatility = realized_volatility[-1512:]

y = np.array(np.log(realized_volatility['close_price']/realized_volatility['close_price'].shift(1)).dropna()) * 100
mu = np.mean(y)
x = np.log((y-mu)**2)

# Initial values
phi_ini = 0.99 #np.cov(y[1:], y[:-1])[0][1]/(np.var(y[1:])- np.pi**2/2)
omega_ini =  (1 - phi_ini) * (np.mean(x) + 1.27) 
sig_eta_ini = (1 - phi_ini**2) * (np.var(x) - (np.pi**2)/2)

bounds = [(0,1), (0, None), (None, None)]
opt = minimize(lambda params: -KF_LL(x,params, 1), method='Nelder-Mead', bounds = bounds, x0 = [phi_ini, sig_eta_ini, omega_ini], options= {'maxiter': 1e10, 'maxfev': 100000})#KALMAN SMOOTHER FUNCTION

phi = opt.x[0]
omega = opt.x[2]
sig_eta = opt.x[1]

ml_params = [phi, sig_eta, omega]
print(f'phi: {phi}\nomega: {omega}\nsig_eta: {sig_eta}')

a,P,v,F,K,n = KF_LL(x, ml_params, 0)
plt.plot(x, 'o', color = 'grey')
plt.plot(a, color='red')
plt.show()

r, N, a_hat, V = KS_LL(x, v, P, F, a, K, phi)
plt.plot(x, 'o', color = 'grey')
plt.plot(a, color='red')
plt.plot(a_hat, color = 'blue')
plt.show()

x = np.array(np.log(realized_volatility['rv5'])) - 1.27

bounds = [(0,1), (0, None), (None, None)]
phi_ini = 0.95 #np.cov(y[1:], y[:-1])[0][1]/(np.var(y[1:])- np.pi**2/2)
omega_ini =  (1 - phi_ini) * (np.mean(x) + 1.27) 
sig_eta_ini = (1 - phi_ini**2) * (np.var(x) - (np.pi**2)/2)

opt = minimize(lambda params: -KF_LL(x,params, 1), method='Nelder-Mead', bounds = bounds, x0 = [phi_ini, sig_eta_ini, omega_ini], options= {'maxiter': 1e10, 'maxfev': 100000})#KALMAN SMOOTHER FUNCTION

phi = opt.x[0]
omega = opt.x[2]
sig_eta = opt.x[1]

ml_params = [phi, sig_eta, omega]
print(f'phi: {phi}\nomega: {omega}\nsig_eta: {sig_eta}')

a,P,v,F,K,n = KF_LL(x, ml_params, 0)
plt.plot(x, 'o', color = 'grey')
plt.plot(a, color='red')
plt.show()