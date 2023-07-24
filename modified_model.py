# -*- coding: utf-8 -*-
"""
Modified model: ’Energy loss during photodamage''

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


def photomodel_mod(x, t, alpha_R, I, gamma):

    # define variables[mol/gCDW]
    a = x[0]
    P = x[1]

    # define active photosynthtic unit concentration [mol/gCDW]
    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    # define ribosomal concentration [mol/gCDW]
    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    # define growth rate [h^(-1)]
    mu = delta * k_catP * P1 * (1/1+(a/K_I)) + delta * n_P*k_D * P1 * I * (gamma-1)

    # rate equations
    v_2 = k_catP * P1 * (1/1+(a/K_I))
    v_D = k_D * P1 * I
    gamma_P = (a/(K_R+a)) * (k_catR/n_P) * (1 - alpha_R) * R
    gamma_total = (a/(K_R+a))*k_catR*R

    # define ODEs
    dadt = v_2 + (gamma*n_P*v_D) - gamma_total - (mu * a)
    dPdt = gamma_P - mu*P - v_D
    return[dadt, dPdt]


"""simulation"""
# alpha_R_range
alpha_R_range = np.arange(0.01, 1.0, 0.01)

# light intensity range
I_values = [0.1, 0.5, 1, 5, 10, 60, 100, 300]

# set value for gamma
# gamma = 0 --> no reuse of damaged P1
# gamma = 1 --> every a from P1 reused
gamma = 0.5

# Create lists of the growth rate for different light intensity
mu_1 = []
mu_2 = []
mu_3 = []
mu_4 = []
mu_5 = []
mu_6 = []
mu_7 = []
mu_8 = []


# Create lists of optimal values from all light intensities
mu_final = []
a_final = []
P_final = []
P1_final = []
R_final = []
alpha_R_final = []

for I in I_values:
    mu_list = []
    a_list = []
    P_list = []
    P1_list = []
    R_list = []

    for alpha_R in alpha_R_range:

        # time interval
        t = np.arange(0.1, 1000, 1.0)

        # intial conditions
        x0 = [0.1, 0.000001]

        x = odeint(photomodel_mod, x0, t, args=(alpha_R, I, gamma, ))

        a = x[-1, 0]

        P = x[-1, 1]

        P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1 + epsilon)*sigma * I)

        R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

        mu = delta * k_catP * P1 * (1/1+(a/K_I)) + delta * n_P*k_D * P1 * I * (gamma - 1)


        if I == I_values[0]:
            mu_1.append(mu)
        if I == I_values[1]:
            mu_2.append(mu)
        if I == I_values[2]:
            mu_3.append(mu)
        if I == I_values[3]:
            mu_4.append(mu)
        if I == I_values[4]:
            mu_5.append(mu)
        if I == I_values[5]:
            mu_6.append(mu)
        if I == I_values[6]:
            mu_7.append(mu)
        if I == I_values[7]:
            mu_8.append(mu)

        mu_list.append(mu)
        a_list.append(a)
        R_list.append(R)
        P_list.append(P)
        P1_list.append(P1)

    mu_final.append(max(mu_list))
    mu_opti = np.array(mu_final)
    a_final.append(a_list[mu_list.index(max(mu_list))])
    a_opti = np.array(a_final)
    R_final.append(R_list[mu_list.index(max(mu_list))])
    R_opti = np.array(R_final)
    P_final.append(P_list[mu_list.index(max(mu_list))])
    P_opti = np.array(P_final)
    P1_final.append(P1_list[mu_list.index(max(mu_list))])
    P1_opti = np.array(P1_final)
    alpha_R_final.append(alpha_R_range[mu_list.index(max(mu_list))])
    alpha_R_opti = np.array(alpha_R_final)

# define Ratio active to inactive photosynthtic unit
P_ratio = P1_opti/(P_opti-P1_opti)


# calculate values for alpha_R function
def AH(x, f1, f2, f3):
    return (f1*x)/(f2+x+(f3/f2)*x**2)


P1P0 = P_ratio
alpha = alpha_R_opti

mod = Model(AH)
result = mod.fit(alpha, x=P1P0, f2=49, f3=0.02, f1=0.2223)
print(fit_report(result))

# define division time [h]
Td = np.log(2)/mu_opti


""" self-optimising modified """


def selfopti_photomodel_mod(x, t, gamma, I):

    # variables
    a = x[0]
    P = x[1]

    # P_total = P* (= P1) + P_0
    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    # alpha_R
    # gamma = 0 --> no reuse of damaged P*
    # gamma = 1 --> every a from P* reused

    # values for gamma = 0.5
    f1 = 0.21
    f2 = 0.36
    f3 = 5.14e-04

    alpha_R = (f1*(P1/(P-P1)))/(f2+(P1/(P-P1))+(f3/f2)
                                * (P1/(P-P1))**2)

    # R
    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    # growth rate
    mu = delta * k_catP * P1 * (1/1+(a/K_I)) + delta * n_P*k_D * P1 * I * (gamma-1)

    # rate equations
    v_2 = k_catP * P1 * (1/1+(a/K_I))
    v_D = k_D * P1 * I
    gamma_P = (a/(K_R+a)) * (k_catR/n_P) * (1 - alpha_R) * R
    gamma_total = (a/(K_R+a))*k_catR*R  # y_total = n_P*y_P + n_R*y_R

    # ODEs
    dadt = v_2 + (gamma*n_P*v_D) - gamma_total - (mu * a)
    dPdt = gamma_P - mu*P - v_D
    return[dadt, dPdt]


"""simulation"""
# alpha_R_range
alpha_R_range = np.arange(0.01, 1.0, 0.01)

# light intensity range
I_values = [0.1, 0.5, 1, 5, 10, 60, 100, 300]

# set value for gamma
gamma = 0.5

# create a lists for store values
mu_list = []
P_list = []
R_list = []
a_list = []

for I in I_values:

    # time interval
    t = np.arange(0.1, 1000, 1.0)

    # initial conditions
    x0 = [0.1, 0.000001]

    x = odeint(selfopti_photomodel_mod, x0, t, args=(gamma, I))

    a = x[:, 0]
    P = x[:, 1]

    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    mu = delta * k_catP * P1 * (1/1+(a/K_I)) + delta * n_P * k_D * P1 * I * (gamma - 1)

    P_list.append(P[-1])
    P_opti = np.array(P_list)
    R_list.append(R[-1])
    R_opti = np.array(R_list)
    a_list.append(a[-1])
    a_opti = np.array(a_list)
    mu_list.append(mu[-1])
    mu_opti = np.array(mu_list)

# define mass fractions
mP = delta * n_P * P_opti
mR = delta * n_R * R_opti
ma = delta * a_opti

# calculte division time [h]
Td = np.log(2)/mu_opti

""" self-optimising modified - changes in I """
# adopt the autonomous model but define I within funcion definition as follows:

# define changes in light intensity
# T = 200 
# sin = np.sin(2*np.pi*t/T)

# if sin > 0:
#     I = 0.1
# else:
#     I = 300

# and in the simulation like this:
# I1 = np.repeat(0.1, 100)
# I2 = np.repeat(300, 100)
# I12 = np.concatenate((I1, I2))
# I = np.concatenate((I12, I12, I12, I12, I12))
