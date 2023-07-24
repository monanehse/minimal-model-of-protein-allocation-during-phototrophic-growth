# -*- coding: utf-8 -*-
"""
self-optimising model

@author: nehse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import Model, fit_report

# define fixed parameter
n_R = 7358                  # [unit of a/molecule]
n_P = 95451                 # [unit of a/molecule]
k_catR = 10*60*60           # [h^(-1)]
k_catP = k_catR             # [h^(-1)]
K_I = 10**8                 # [molecules/cell]
K_R = 0.5                   # [gCDW/unit of a]
sigma = 0.1*10**(6)         # [not defined]
epsilon = 1.0 * 10**(-8.0)  # [not defined]
k_D = sigma*epsilon         # [not defined]
delta = 1                   # [gCDW/unit of a]


def selfopti_photomodel(x, t, I):

    # define variables[mol/gCDW]
    a = x[0]
    P = x[1]

    # define active photosynthtic unit concentration [mol/gCDW]
    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    # define alpha_R
    f1 = 0.202
    f2 = 0.306
    f3 = 1.23e-04

    alpha_R = (f1*(P1/(P-P1)))/(f2+(P1/(P-P1))+(f3/f2) *
                                (P1/(P-P1))**2)

    # define ribosomal concentration [mol/gCDW]
    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    # define growth rate [h^(-1)]
    mu = delta * k_catP * P1 * (1/1+(a/K_I))

    # define rate equations
    v_2 = k_catP * P1 * (1/1+(a/K_I))
    v_D = k_D * P1 * I
    gamma_P = (a/(K_R+a)) * (k_catR/n_P) * (1 - alpha_R) * R
    gamma_total = (a/(K_R+a))*k_catR*R  # y_total = n_P*y_P + n_R*y_R

    # define ODEs
    dadt = v_2 + (n_P*v_D) - gamma_total - (mu * a)
    dPdt = gamma_P - mu*P - v_D
    return[dadt, dPdt]


"""simulation"""

# create a lists for store values
mu_list = []
P_list = []
R_list = []
a_list = []

# light intensity range
I_values = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 50, 60, 100, 150, 200, 250, 300,
            400, 500]
# I_values = [0.1, 0.5, 1, 5, 10, 60, 100, 500]

for I in I_values:

    # time interval
    t = np.arange(0.1, 1000, 1.0)

    # initial conditions
    x0 = [0.1, 0.000001]

    x = odeint(selfopti_photomodel, x0, t, args=(I, ))

    a = x[:, 0]
    P = x[:, 1]

    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    mu = delta * k_catP * P1 * (1/1+(a/K_I))

    P_list.append(P[-1])
    P_opti = np.array(P_list)
    R_list.append(R[-1])
    R_opti = np.array(R_list)
    a_list.append(a[-1])
    a_opti = np.array(a_list)
    mu_list.append(mu[-1])
    mu_opti = np.array(mu_list)

    if I == I_values[0]:
        a_1 = a
        P_1 = P
        P1_1 = P1
        R_1 = R
        mu_1 = mu
    if I == I_values[2]:
        a_2 = a
        P_2 = P
        P1_2 = P1
        R_2 = R
        mu_2 = mu
    if I == I_values[3]:
        a_3 = a
        P_3 = P
        P1_3 = P1
        R_3 = R
        mu_3 = mu
    if I == I_values[5]:
        a_4 = a
        P_4 = P
        P1_4 = P1
        R_4 = R
        mu_4 = mu
    if I == I_values[6]:
        a_5 = a
        P_5 = P
        P1_5 = P1
        R_5 = R
        mu_5 = mu
    if I == I_values[9]:
        a_6 = a
        P_6 = P
        P1_6 = P1
        R_6 = R
        mu_6 = mu
    if I == I_values[10]:
        a_7 = a
        P_7 = P
        P1_7 = P1
        R_7 = R
        mu_7 = mu
    if I == I_values[16]:
        a_8 = a
        P_8 = P
        P1_8 = P1
        R_8 = R
        mu_8 = mu

# define mass fractions
mP = delta * n_P * P_opti
mR = delta * n_R * R_opti
ma = delta * a_opti

# calculte division time [h]
Td = np.log(2)/mu_opti


# define linear fit function


def linearFit(l, slope, b):
    return (slope*l + b)


"""plotting"""

# Growth rate over time. Figure S3d of the thesis
plt.plot(t, mu_1)
plt.plot(t, mu_2)
plt.plot(t, mu_3)
plt.plot(t, mu_4)
plt.plot(t, mu_5)
plt.plot(t, mu_6)
plt.plot(t, mu_7)
plt.plot(t, mu_8)
plt.ylabel(r' $\mu [h^{-1}]$')
plt.xlabel('t [h]')
plt.legend(['I = 0.1', 'I = 0.5', 'I = 1', 'I = 5', 'I = 10', 'I = 60',
            'I = 100', 'I = 500'])
plt.show()

# Precursor molecule concentration over time. Figure S3c of the thesis
plt.plot(t, a_1)
plt.plot(t, a_2)
plt.plot(t, a_3)
plt.plot(t, a_4)
plt.plot(t, a_5)
plt.plot(t, a_6)
plt.plot(t, a_7)
plt.plot(t, a_8)
plt.xlabel('t [h]')
plt.ylabel('a [gCDW]')
plt.legend(['I = 0.1', 'I = 0.5', 'I = 1', 'I = 5', 'I = 10', 'I = 60',
            'I = 100', 'I = 500'])
plt.show()

# Photisynthtic unit concentration over time. Figure S3b of the thesis
plt.plot(t, P_1)
plt.plot(t, P_2)
plt.plot(t, P_3)
plt.plot(t, P_4)
plt.plot(t, P_5)
plt.plot(t, P_6)
plt.plot(t, P_7)
plt.plot(t, P_8)
plt.ylabel('P [gCDW]')
plt.xlabel('t [h]')
plt.legend(['I = 0.1', 'I = 0.5', 'I = 1', 'I = 5', 'I = 10', 'I = 60',
            'I = 100', 'I = 500'])
plt.show()

# Ribosomal concentration over time. Figure S3a of the thesis
plt.plot(t, R_1)
plt.plot(t, R_2)
plt.plot(t, R_3)
plt.plot(t, R_4)
plt.plot(t, R_5)
plt.plot(t, R_6)
plt.plot(t, R_7)
plt.plot(t, R_8)
plt.xlabel('t [h]')
plt.ylabel('R [gCDW]')
plt.legend(['I = 0.1', 'I = 0.5', 'I = 1', 'I = 5', 'I = 10', 'I = 60',
            'I = 100', 'I = 500'])
plt.show()


# Growth laws. Figure 11 of the thesis
mP_fit = mP[0:7]
mR_fit = mR[0:7]
mu_fit = mu_opti[0:7]

mod = Model(linearFit)
result1 = mod.fit(mP_fit, l=mu_fit, slope=(-0.5), b=0.9)
result2 = mod.fit(mR_fit, l=mu_fit, slope=0.5, b=0.05)

result1_mod = -0.936 * mu_fit + 0.888
result2_mod = 0.465 * mu_fit + 0.062

plt.plot(mu_opti, mP, 'o')
plt.plot(mu_opti, mR, 'v')
# plt.plot(mu_opti, ma, 'x')

plt.plot(mu_fit, result1_mod, '--')
plt.plot(mu_fit, result2_mod, '--')


plt.legend([r'$\omega_{P} \, P$', r'$\omega_{R} \,R$',
            r'fit (linear part) $\omega_{P} \, P$',
            r'fit (linear part) $\omega_{R} \, R$'])
plt.xlabel(r' $\mu [h^{-1}]$')
plt.ylabel('protein mass fraction')
plt.xlim([0, 0.26])
plt.ylim([0, 1.0])
plt.show()

# Comparison growth rate models. Figure 09 of the thesis

# optimal growth rate from optimized model
growth_optimized = [0.068, 0.119, 0.16, 0.195, 0.224, 0.236, 0.242, 0.243,
                    0.241, 0.24, 0.234, 0.228, 0.222, 0.216, 0.211, 0.201,
                    0.192]

plt.plot(I_values, mu_opti, 'v', color='tab:orange')
plt.plot(I_values, growth_optimized, 'xb')
y_list = []
for I in I_values:
    growth_opti = 0.249
    K_A = 0.273
    gamma = 1.6252e-04
    y = (growth_opti*I)/(K_A+I+(gamma/K_A)*I**2)
    y_list.append(y)
plt.plot(I_values, y_list, '--', color='silver')
plt.legend([r' $\mu^*$ self-optimising model', r'optimal $\mu$ optimised model',
            'Aiba/Haldane'], loc='center right')
plt.xlabel(r' I [$\frac{\mu mol} {m^{2}} \, \frac{1}{s}]$')
plt.ylabel(r' $\mu [h^{-1}]$')
plt.xlim([-10, 310])
plt.ylim([0, 0.25])
plt.vlines(x=10, ymin=0, ymax=0.25, color='lightblue', linestyle='dotted')
plt.vlines(x=60, ymin=0, ymax=0.25, color='lightblue', linestyle='dotted')
plt.show()


# Growth rate for different alpha_R. Figure 10 of the thesis

# mu_fixed values from optimised_model
mu_fixed_01 = [0.068, 0.117, 0.153, 0.178, 0.197, 0.203, 0.207, 0.208, 0.208,
               0.207, 0.206, 0.203, 0.201, 0.198, 0.196, 0.191, 0.186]
mu_fixed_02 = [0.066, 0.117, 0.16, 0.193, 0.22, 0.23, 0.235, 0.237, 0.236,
               0.235, 0.231, 0.226, 0.221, 0.216, 0.211, 0.2, 0.189]
mu_fixed_03 = [0.063, 0.115, 0.158, 0.194, 0.224, 0.236, 0.242, 0.243, 0.241,
               0.239, 0.234, 0.226, 0.218, 0.21, 0.202, 0.186, 0.17]

plt.plot(I_values, mu_opti, 'o')
plt.plot(I_values, mu_fixed_01, '--')
plt.plot(I_values, mu_fixed_02, '--')
plt.plot(I_values, mu_fixed_03, '--')
plt.xlabel(r' log (I [$\frac{\mu mol} {m^{2}} \, \frac{1}{s}])$')
plt.ylabel(r' $\mu [h^{-1}]$')
plt.legend([r'$\alpha_{R} = f\,(\frac{P^*}{P^0})$', r'$\alpha_{R} = 0.1$',
            r'$\alpha_{R} = 0.15$', r'$\alpha_{R} = 0.2$'], loc='lower right')
plt.xscale('log')
plt.ylim([0, 0.26])
plt.show()


"""
self-optimising model - periodic changes in I

"""


def selfopti_photomodel_changes_I(x, t):

    # define variables[mol/gCDW]
    a = x[0]
    P = x[1]

    # define changes in light intensity
    T = 200  # periode
    sin = np.sin(2*np.pi*t/T)

    if sin > 0:
        I = 0.1
    else:
        I = 300

    # define active photosynthtic unit concentration [mol/gCDW]
    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    # define alpha_R
    f1 = 0.202
    f2 = 0.306
    f3 = 1.23e-04

    alpha_R = (f1*(P1/(P-P1)))/(f2+(P1/(P-P1))+(f3/f2) *
                                (P1/(P-P1))**2)

    # define ribosomal concentration [mol/gCDW]
    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    # define growth rate [h^(-1)]
    mu = delta * k_catP * P1 * (1/1+(a/K_I))

    # define rate equations
    v_2 = k_catP * P1 * (1/1+(a/K_I))
    v_D = k_D * P1 * I
    gamma_P = (a/(K_R+a)) * (k_catR/n_P) * (1 - alpha_R) * R
    gamma_total = (a/(K_R+a))*k_catR*R  # y_total = n_P*y_P + n_R*y_R

    # ODEs
    dadt = v_2 + (n_P*v_D) - gamma_total - (mu * a)
    dPdt = gamma_P - mu*P - v_D
    return[dadt, dPdt]


"""simulation"""

# define I
I1 = np.repeat(0.1, 100.0)  # corresponds to phase when sin > 0
I2 = np.repeat(300.0, 100.0)  # corresponds to phase when sin < 0
I12 = np.concatenate((I1, I2))
I = np.concatenate((I12, I12, I12, I12, I12))

# time interval
t = np.arange(0.1, 1000, 1.0)

# initial conditions
x0 = [0.1, 0.000001]

x = odeint(selfopti_photomodel_changes_I, x0, t)

a = x[:, 0]
P = x[:, 1]

R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)
P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)
mu = delta * k_catP * P1 * (1/1+(a/K_I))

# define massfractions
massfracP = n_P*P
massfracR = n_R*R

mu_1 = mu[0:100]  # I=0.1
mu_2 = mu[100:200]  # I=100


"""plotting"""

# Periodic changes in I. Figure 12 of the thesis
plt.plot(t, mu, 'g')
plt.plot(t, a, 'r')
plt.plot(t, massfracR, 'b')
plt.plot(t, massfracP, 'y')
plt.legend([r'$\mu$', r'$\omega_{a} \, a$',
            r'$\omega_{R} \, R$', r'$\omega_{P} \, P$'], loc='best')
plt.xlim([150, 400])
plt.ylim([0, 1.0])
plt.axhline(y=0.068, color='mediumseagreen', linestyle='dotted')
plt.axhline(y=0.234, color='mediumseagreen', linestyle='dotted')
plt.xlabel('time [h]')
plt.ylabel(r'protein mass fraction/ $\mu [h^{-1}]$')
plt.text(152, 1.05, ' I = 300 ', fontsize=12, bbox=dict
         (facecolor='cornflowerblue', alpha=0.7))
plt.text(200, 1.05, '          I = 0.1           ', fontsize=12, bbox=dict
         (facecolor='lightblue', alpha=0.7))
plt.text(300, 1.05, '           I = 300           ', fontsize=12, bbox=dict
         (facecolor='cornflowerblue', alpha=0.7))
plt.show()
