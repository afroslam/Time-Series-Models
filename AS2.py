import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

sv = pd.read_excel('sv.xlsx', sheet_name= 'Sheet1')

# A #################################################################################################################
y_a = sv['GBPUSD'].values/100
plt.plot(y_a, color = 'grey')
plt.ylabel('log returns')
plt.savefig('tsm_ass2_a')
plt.show()

sample_stats_a = pd.DataFrame({'Moment': ['Mean', 'Variance', 'Skewness', 'Kurtosis'], 
                  'Value': [np.mean(y_a), np.var(y_a), skew(y_a), kurtosis(y_a)]})

print(sample_stats_a)
# B #################################################################################################################
mu_b = np.mean(y_a)
x_b = np.log((y_a-mu_b)**2)

plt.plot(x_b, color = 'grey')
plt.ylabel(r'$x_t$')
plt.savefig('tsm_ass2_b')
plt.show()


# C #################################################################################################################
#KALMAN FILTER FUNCTION
def KF_LL(y, params, ml_flag):
    '''
    variables
        Z_t = 1
        d_t = 0
        e_t = u_t
        a_t = h_t
        T_t = phi
        c_t = omega (constant)
        R_t = 1

        H_t = Var[log e^2]
        Q_t = sig2_eta
    '''

    phi, sig2_eta, omega = params

    Z = 1
    T = phi
    Q = sig2_eta
    R = 1
    d=-1.27

    a_ini = omega/(1-phi)
    P_ini = sig2_eta/(1-phi**2)
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

#QML Function
def getQMLparams(x, phi_ini, bounds = [(0,1), (0, None), (None, None)]):
    #Initialize omega and sig2_eta using phi_ini
    omega_ini =  (1 - phi_ini) * (np.mean(x) + 1.27) 
    sig2_eta_ini = (1 - phi_ini**2) * (np.var(x) - (np.pi**2)/2)
    #QML Optimization
    opt = minimize(lambda params: -KF_LL(x, params, 1), 
                    method='Nelder-Mead', 
                    bounds = bounds, 
                    x0 = [phi_ini, sig2_eta_ini, omega_ini], 
                    options= {'maxiter': 1e10, 'maxfev': 100000})

    #Obtain parameters
    phi = opt.x[0]
    sig2_eta = opt.x[1]
    omega = opt.x[2]
    psi_hat = omega/(1-phi)
    ml_params = [phi, sig2_eta, omega]
    print(f'phi: {phi}\nomega: {omega}\nsig2_eta: {sig2_eta}')

    return ml_params, psi_hat

phi_ini_c = 0.9731
ml_params_c, psi_hat_c = getQMLparams(x_b, phi_ini_c)

# D #################################################################################################################
#Running KF with the QML obtained parameters and plotting h_t
a_d, P_d, v_d, F_d, K_d, n_d = KF_LL(x_b, ml_params_c, 0)

#KALMAN SMOOTHER FUNCTION
def KS_LL(y, v, P, F, a, K, phi, saveFig=''):
    d = -1.27
    r =  np.zeros(len(y))
    r[-1] = 0

    N = np.zeros(len(y))
    N[-1] = 0
    
    a_hat = np.zeros(len(y))

    V = np.zeros(len(y))
    L = phi - K 
    for t in range(len(y)-2, -1, -1):
        r[t] = F[t]**-1*v[t] + L[t] * r[t+1] 
        N[t] = F[t]**-1 + L[t]**2*N[t+1]

    for t in range(0, len(y)):
        a_hat[t] = a[t] + P[t] *r[t]
        V[t] = P[t] - P[t]**2*N[t]


    if len(saveFig)>0:
        plt.plot(y-d, 'o', markersize = 3, color = 'grey', label = r'Transformed $x_t$')
        plt.plot(a_hat, color = 'red', label = 'Kalman smoother')
        plt.legend()
        plt.savefig(saveFig)
        plt.show()

    return (r, N, a_hat, V)

#Running KS with QML obtained parameters
r_d, N_d, a_hat_d, V_d = KS_LL(x_b, v_d, P_d, F_d, a_d, K_d, ml_params_c[0], 
                               saveFig= 'tsm_ass2_d1')


#Plotting filtered and smoothed h_tilde
kf_h_tilde_d = a_d-psi_hat_c
ks_h_tilde_d = a_hat_d-psi_hat_c

plt.plot(kf_h_tilde_d, color='blue', label = r'$E[\tilde{h}_t|x_1,...,x_t]$')
plt.plot(ks_h_tilde_d, color = 'red', label = r'$E[\tilde{h}_t|x_1,...,x_n]$')
plt.legend()
plt.savefig('tsm_ass2_d2')
plt.show()


# E #################################################################################################################
df = pd.read_csv('realized_volatility.csv')
df = df[df['Symbol'] == '.SPX']
rv_df = df[-2000:]

y_e = np.array(np.log(rv_df['close_price']/rv_df['close_price'].shift(1)).dropna())
mu_e = np.mean(y_e)
x_e = np.log((y_e-mu_e)**2)

plt.plot(y_e, color = 'grey')
plt.ylabel('log returns')
plt.savefig('tsm_ass2_e1')
plt.show()

sample_stats_e = pd.DataFrame({'Moment': ['Mean', 'Variance', 'Skewness', 'Kurtosis'], 
                  'Value': [np.mean(y_e), np.var(y_e), skew(y_e), kurtosis(y_e)]})

print(sample_stats_e)

plt.plot(x_e, color = 'grey')
plt.ylabel(r'$x_t$')
plt.savefig('tsm_ass2_e2')
plt.show()

#QML estimation
phi_ini_e = 0.99
ml_params_e, psi_hat_e = getQMLparams(x_e, phi_ini_e)

#Running KF using QML params
a_e, P_e, v_e, F_e, K_e, n_e = KF_LL(x_e, ml_params_e, 0)

#Running KS using QML params
r_e, N_e, a_hat_e, V_e = KS_LL(x_e, v_e, P_e, F_e, a_e, K_e, ml_params_e[0], 
                               saveFig = 'tsm_ass2_e3')


#Plotting filtered and smoothed h_tilde
kf_h_tilde_e = a_e-psi_hat_e
ks_h_tilde_e = a_hat_e-psi_hat_e

plt.plot(kf_h_tilde_e, color='blue', label = r'$E[\tilde{h}_t|x_1,...,x_t]$')
plt.plot(ks_h_tilde_e, color = 'red', label = r'$E[\tilde{h}_t|x_1,...,x_n]$')
plt.legend()
plt.savefig('tsm_ass2_e4')
plt.show()


#Obtaining log realized volatility values and subtracting 1.27
x_rv = np.array(np.log(rv_df['rv5'][1:])) - 1.27

#Running KF for realized vol returns
a_star, P_star, x_star, F_star, K_star, n_star = KF_LL(x_rv, ml_params_e, 0)

#Obtaining beta using GLS
var_beta_hat = np.sum(x_star**2 * F_star**(-1))**(-1)
beta_hat = var_beta_hat * np.sum(x_star * F_star**(-1) * v_e)
print('Beta: ', beta_hat)


#Filtering realized volatility; obtain returns and transform
y = np.array(np.log(rv_df['close_price']/rv_df['close_price'].shift(1)).dropna()) 
mu = np.mean(y)
x = np.log((y-mu)**2)
x_demeaned = x - beta_hat * (x_rv + 1.27)

#QML estimation
phi_ini_rv = 0.995
ml_params_rv, psi_hat_rv = getQMLparams(x_demeaned, phi_ini_rv)

#Obtaining and plotting KF and KS
a_rv, P_rv, v_rv, F_rv, K_rv, n_rv = KF_LL(x_demeaned, ml_params_rv, 0)
r_rv, N_rv, a_hat_rv, V_rv = KS_LL(x_demeaned, v_rv, P_rv, F_rv, a_rv, 
                                   K_rv, ml_params_rv[0], saveFig = 'tsm_ass2_e5')

#Plotting filtered and smoothed h_tilde
kf_h_tilde_rv = a_rv-psi_hat_rv
ks_h_tilde_rv = a_hat_rv-psi_hat_rv

plt.plot(kf_h_tilde_rv, color='blue', label = r'$E[\tilde{h}_t|x_1,...,x_t]$')
plt.plot(ks_h_tilde_rv, color = 'red', label = r'$E[\tilde{h}_t|x_1,...,x_n]$')
plt.legend()
plt.savefig('tsm_ass2_e6')
plt.show()

# F #################################################################################################################
def bootstrap_filter_method(y, M, sig2_eta, phi, psi):#, a_ini, P_ini): 
    """
    Function to do bootstrap filtering
    Parameters:
    y - Observations
    M - Number of Particles
    sig2_eta - Variance
    phi - Coefficient in state updating equation
    psi - Parameter in exponential
    """
    T = len(y)
    h_vector = np.zeros(T)
    h_mean = 0 # Initial values #np.dot(phi, np.random.normal(a_ini, np.sqrt(P_ini), size = M))
    for i in range(0, T):
        a_tilde = np.random.normal(loc = h_mean, scale = np.sqrt(sig2_eta), size = M) # Density of state updating equation
        sig_t =np.exp((psi + a_tilde)/2) ## Standard deviation of y
        weights = st.norm.pdf(y[i], loc = np.zeros(M), scale = sig_t) #p(y_t|a_t)
        weights /= sum(weights) ## Normalize weights
        h_vector[i]= np.sum(weights*a_tilde) ## Compute state
        a = np.random.choice(a_tilde, size = M, p = weights, replace = True) ## Sample using weights
        h_mean = phi*a
    return h_vector


phi = ml_params_c[0]
sig2_eta = ml_params_c[1]
omega = ml_params_c[2]
psi = omega/(1-phi)
M = 10000
h_c = bootstrap_filter_method(y_a-np.mean(y_a), M, sig2_eta, phi, psi)#, a_ini, P_ini)

plt.plot(h_c, color = 'red', label = r"QML $E[\tilde{h}_t|x_1,...,x_t]$")
plt.plot(a_d-psi, color = 'blue', label = r'Bootstrapped $E[\tilde{h}_t|y_1,...,y_t]$')
plt.legend()
plt.savefig('tsm_ass2_f1')
plt.show()

phi = ml_params_e[0]
sig2_eta = ml_params_e[1]
omega = ml_params_e[2]
psi = omega/(1-phi)
M = 10000
h_e = bootstrap_filter_method(y_e-np.mean(y_e), M, sig2_eta, phi, psi)#, a_ini, P_ini)

plt.plot(h_e, color = 'red', label = r"QML $E[\tilde{h}_t|x_1,...,x_t]$")
plt.plot(a_e-psi, color = 'blue', label = r'Bootstrapped $E[\tilde{h}_t|y_1,...,y_t]$')
plt.legend()
plt.savefig('tsm_ass2_f2')
plt.show()