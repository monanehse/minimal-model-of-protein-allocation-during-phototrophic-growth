# -*- coding: utf-8 -*-
"""
optimised model of cyanobacterial phototrophic growth

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


def photomodel(x, t, alpha_R, I):

    # define variables[mol/gCDW]
    a = x[0]
    P = x[1]

    # define active photosynthtic unit concentration [mol/gCDW]
    P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1+epsilon)*sigma*I)

    # define ribosomal concentration [mol/gCDW]
    R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

    # define growth rate [h^(-1)]
    mu = delta * k_catP * P1 * (1/1+(a/K_I))

    # define rate equations
    v_2 = k_catP * P1 * (1/1+(a/K_I))
    v_D = k_D * P1 * I
    gamma_P = (a/(K_R+a)) * (k_catR/n_P) * (1 - alpha_R) * R
    gamma_total = (a/(K_R+a))*k_catR*R

    # define ODEs
    dadt = v_2 + (n_P*v_D) - gamma_total - (mu * a)
    dPdt = gamma_P - mu*P - v_D
    return[dadt, dPdt]


"""simulation"""

# Create lists of the growth rate for different light intensity
mu_1 = []
mu_2 = []
mu_3 = []
mu_4 = []
mu_5 = []
mu_6 = []
mu_7 = []
mu_8 = []

# Create lists of the growth rate for fixed alpha_R values
mu_alpha1 = []
mu_alpha2 = []
mu_alpha3 = []
mu_alpha4 = []
mu_alpha5 = []

# Create lists of optimal values from all light intensities
mu_final = []
a_final = []
P_final = []
P1_final = []
R_final = []
alpha_R_final = []

# alpha_R range
alpha_R_range = np.arange(0.01, 1.0, 0.01)

# light intensity range
I_values = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 50, 60, 100, 150, 200, 250,
            300, 400, 500]

for I in I_values:
    mu_list = []
    a_list = []
    P_list = []
    P1_list = []
    R_list = []

    for alpha_R in alpha_R_range:

        # time interval
        t = np.arange(0.1, 1000, 1.0)

        # initial conditions
        x0 = [0.1, 0.000001]

        x = odeint(photomodel, x0, t, args=(alpha_R, I, ))

        a = x[-1, 0]
        P = x[-1, 1]

        P1 = (sigma*I*P)/(k_catP * (1/1+(a/K_I)) + (1 + epsilon) * sigma * I)

        R = 1/(delta*n_R) - (a / n_R) - ((n_P*P)/n_R)

        mu = delta * k_catP * P1 * (1/1+(a/K_I))

        if I == I_values[0]:
            mu_1.append(mu)
        if I == I_values[2]:
            mu_2.append(mu)
        if I == I_values[3]:
            mu_3.append(mu)
        if I == I_values[5]:
            mu_4.append(mu)
        if I == I_values[6]:
            mu_5.append(mu)
        if I == I_values[9]:
            mu_6.append(mu)
        if I == I_values[10]:
            mu_7.append(mu)
        if I == I_values[16]:
            mu_8.append(mu)

        if alpha_R == alpha_R_range[9]:
            mu_alpha1.append(mu)
        if alpha_R == alpha_R_range[14]:
            mu_alpha2.append(mu)
        if alpha_R == alpha_R_range[19]:
            mu_alpha3.append(mu)
        if alpha_R == alpha_R_range[24]:
            mu_alpha4.append(mu)
        if alpha_R == alpha_R_range[29]:
            mu_alpha5.append(mu)

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

# define ratio active to inactive photosynthetic unit
P_ratio = P1_opti/(P_opti-P1_opti)

# define mass fractions
mR_opti = delta*n_R*R_opti
mP_opti = delta*n_P*P_opti
ma_opti = delta*a_opti


# calculate values for alpha_R function
def AH(x, f1, f2, f3):
    return(f1*x)/(f2+x+(f3/f2)*x**2)


P1P0 = P_ratio
alpha = alpha_R_opti

mod = Model(AH)
result = mod.fit(alpha, x=P1P0, f2=49, f3=0.02, f1=0.2223)
print(fit_report(result))

# define mass fractions
mR_opti = delta*n_R*R_opti
mP_opti = delta*n_P*P_opti
ma_opti = delta*a_opti

# defien division time [h]
Td = np.log(2)/mu_opti


# define Aida/Haldane equation
def AibaHaldane(I, growth_opti, K_A, gamma):
    HA = (growth_opti*I)/(K_A+I+(gamma/K_A)*I**2)
    return(HA)


# define linear fit function
def linearFit(g, slope, b):
    return (slope*g + b)


"""plotting"""


# Allocation Plot. Figure 03 of the thesis
plt.plot(alpha_R_range, mu_1)
plt.plot(alpha_R_range, mu_2)
plt.plot(alpha_R_range, mu_3)
plt.plot(alpha_R_range, mu_4)
plt.plot(alpha_R_range, mu_5)
plt.plot(alpha_R_range, mu_6)
plt.plot(alpha_R_range, mu_7)
plt.plot(alpha_R_range, mu_8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 0.25])
plt.xlabel(r' $\alpha_R$')
plt.ylabel(r' $\mu [h^{-1}]$')
plt.legend(['I = 0.1', 'I = 0.5', 'I = 1', 'I = 5',
            'I = 10', 'I = 60', 'I = 100', 'I = 500'])
plt.show()


# Massfractions. Figure 06 of the thesis
plt.plot(mu_opti, mP_opti, '--x')
plt.plot(mu_opti, mR_opti, '--v')
plt.plot(mu_opti, ma_opti, '--o')
plt.xlim([0.0, 0.25])
plt.ylim([0.0, 1.0])
plt.xlabel(r' $\mu [h^{-1}]$')
plt.ylabel('protein mass fraction')
plt.legend([r'$\omega_{P} \, P$', r'$\omega_{R} \, R$', r'$\omega_{a} \, a$'])
plt.show()


# Compare different alpha_R. Figure 04 of the thesis
plt.plot(I_values, mu_alpha1, '--x')
plt.plot(I_values, mu_alpha2, '--s')
plt.plot(I_values, mu_alpha3, '--^')
plt.plot(I_values, mu_alpha5, '--v')
plt.legend([r' $\alpha_R$ = 0.1', r' $\alpha_R$ = 0.15', r'$\alpha_R$ = 0.2',
            r' $\alpha_R$ = 0.3'])
plt.xscale('log')
plt.xlabel(r'log (I [$\frac{\mu mol} {m^{2}} \, \frac{1}{s}])$')
plt.ylabel(r' $\mu [h^{-1}]$')
plt.ylim(0, 0.25)
plt.show()


# Growth rate over I fit to Aiba/Haldane. Figure 05 of the thesis
y = mu_opti
light = I_values

mod = Model(AibaHaldane)
result = mod.fit(y, I=light, K_A=49, gamma=0.02, growth_opti=0.2223)
# print(fit_report(result))

plt.plot(light, y, '--o', label='optimal growth rates', color='darkblue')
plt.plot(light, result.best_fit, '--', label='Aiba/Haldane fit',
         color='cornflowerblue')
plt.legend(fontsize=11, loc='lower right')
plt.xlabel(r' I [$\frac{\mu mol} {m^{2}} \, \frac{1}{s}]$')
plt.ylabel(r' $\mu [h^{-1}]$')
plt.xlim([-5, 300])
plt.ylim([0, 0.26])
plt.vlines(x=10, ymin=0, ymax=0.26, color='lightblue', linestyle='dotted')
plt.vlines(x=60, ymin=0, ymax=0.26, color='lightblue', linestyle='dotted')
plt.show()


# Optimal alpha_R over  optimal P*/P0. Figure S2 of the thesis
plt.plot(P_ratio,  alpha_R_opti, 'o', color='darkblue')
plt.xlim([-10, 500])
plt.ylim([0, 0.22])
plt.xlabel(r' $\frac{P^{*}}{P^{0}}$')
plt.ylabel(r' $\alpha_R$')
plt.show()


# P*/P0 over I. Figure 07 of the thesis
plt.plot(I_values, P_ratio, 'o',  color='darkblue')

mod = Model(linearFit)
result = mod.fit(P_ratio, g=I_values, slope=0.5, b=1)
plt.plot(I_values, result.best_fit, '--',  color='cornflowerblue')
plt.xlim([0, 200])
plt.ylim([0, 500])
plt.ylabel(r'$\frac{P^{*}}{P^{0}}$')
plt.xlabel(r' I [$\frac{\mu mol} {m^{2}} \, \frac{1}{s}]$')
plt.show()


# Fit alpha_R as function of P*/P0 to Aiba/Haldane. Figure 08
plt.plot(P1P0, alpha, 'o', label=r'optimal $\alpha_R$', color='darkblue')
plt.plot(P1P0, result.best_fit, '-', color='cornflowerblue',
         label='Aiba/Haldane fit')
plt.legend()
plt.xlabel(r' $\frac{P^{*}}{P^{0}}$')
plt.ylabel(r'$\alpha_R$')
plt.xlim([-10, 500])
plt.ylim([0, 0.21])
plt.show()
