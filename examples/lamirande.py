import numpy as np
import matplotlib.pyplot as plt
import time

from pyodys import EDOs
from pyodys import TableauDeButcher
from pyodys import SolveurRKAvecTableauDeButcher

# Problem definition
PP = 0.12  # choisir entre 0 et 1. Exemples a la page 60 du document de Lamirande
c0 = 0.23; m0 = 0.23; n0 = 0.23; p0 = 0.18; q0 = 0.27; u0 = 3e4 * (1 + 4 * PP); v0 = 0.06; w0 = 300.0 * (1 + 4 * PP)
k_UPic0 = 7e5; k_UPic = k_UPic0 * (1 + PP/2); k_VNorm = 5e5

# Parametres
phi_MC = 0.2; phi_MN = 0.35; phi_CAj = 0.08; phi_MAj = 0.08; phi_NAj = 0.08; phi_WC  = 0.934; phi_WM  = 4.67; phi_WN  = 1.33
mu_C  = 6.33e6; mu_M  = 1.2217e7; mu_N  = 1.2217e7; mu_P  = 5; mu_Q  = 781.1
tau_C = 2; tau_M = 2; tau_N = 2
khi_C = 3.50e-8; khi_M = 3.86e-7; khi_N = 3.86e-7; khi_P = 0.236; khi_Q = 7.78e-4
a_C = 0.05; a_M = 0.05; a_N = 0.05; a_P = 97.674; a_Q = 19.6 
e_UC = 1.18e4; e_UM = 1.18e4; e_VN = 1.18e4
h_CAP = 0.0211; h_CjP = 1.69e-3; h_CQ = 5.33e-5; h_MAP = 0.105; h_MjP = 8.4e-3; h_MQ = 2.66e-4; h_NAP = 0.0301; h_NjP = 2.41e-3; h_NQ = 7.61e-5
h_UC = 8.81e-4; h_UM = 8.81e-4; h_VN = 8.81e-4; h_W = 8.81e-4
k_C = 1.17 * (1 + PP/10); k_M = 0.83 * (1 + PP/10); k_N = 0.2; k_P = 0.1; k_Q = 0.3
k_UFaible = 2e5; k_UPic0 = 7e5; k_UPic = k_UPic0 * (1 + PP/2); k_UStable0 = 5e5; k_UStable = k_UStable0 * (1 + PP/2)
k_VFaible = 1e5; k_VNorm = 5e5; k_WFaible = 2250.0; k_WPic0 = 9700.0; k_WPic = k_WPic0 * (1 + PP/2); k_WStable0 = 4500.0; k_WStable = k_WStable0 * (1 + PP/2)

def calcule_energie(U, V):
    E_NV = (a_N * V) / (1 + a_N * h_VN * V)
    E_MU = (a_M * U) / (1 + a_M * h_UM * U)
    E_CU = (a_C * U) / (1 + a_C * h_UC * U)
    return E_NV, E_MU, E_CU

def calcule_param1(M_j, M_A, N_j, N_A, C_j, C_A):
    alpha_MjP = (0.1 *(M_j + M_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    alpha_MAP = (0.9 *(M_j + M_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    alpha_NjP = (0.1 *(N_j + N_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    alpha_NAP = (0.9 *(N_j + N_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    alpha_CjP = (0.1 *(C_j + C_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    alpha_CAP = (0.9 *(C_j + C_A))/(M_j + M_A + N_j + N_A + C_j + C_A)
    return alpha_MjP, alpha_MAP, alpha_NjP, alpha_NAP,  alpha_CjP, alpha_CAP

def calcule_param2(W, M_j, N_j, C_j):
    alpha_WQ = W/(W + phi_WM * M_j + phi_WN * N_j + phi_WC * C_j)
    alpha_MQ = phi_WM * M_j/(W + phi_WM * M_j + phi_WN * N_j + phi_WC * C_j)
    alpha_NQ = phi_WN * N_j/(W + phi_WM * M_j + phi_WN * N_j + phi_WC * C_j)
    alpha_CQ = phi_WC * C_j/(W + phi_WM * M_j + phi_WN * N_j + phi_WC * C_j)
    return alpha_WQ, alpha_MQ, alpha_NQ, alpha_CQ

def calcule_rep1(M_j, M_A, N_j, N_A, C_j, C_A):
    alpha_MjP, alpha_MAP, alpha_NjP, alpha_NAP,  alpha_CjP, alpha_CAP = calcule_param1(M_j, M_A, N_j, N_A, C_j, C_A)
    eta  = 1 + a_P * (alpha_MjP * h_MjP * M_j + alpha_MAP * h_MAP *  M_A + alpha_NjP * h_NjP * N_j + alpha_NAP * h_NAP * N_A + alpha_CjP * h_CjP * C_j + alpha_CAP * h_CAP * C_A)
    E_PMj = (a_P * alpha_MjP * M_j)/eta
    E_PMA = (a_P * alpha_MAP * M_A)/eta
    E_PNj = (a_P * alpha_NjP * N_j)/eta
    E_PNA = (a_P * alpha_NAP * N_A)/eta
    E_PCj = (a_P * alpha_CjP * C_j)/eta
    E_PCA = (a_P * alpha_CAP * C_A)/eta
    return E_PMj , E_PMA, E_PNj, E_PNA, E_PCj, E_PCA

def calcule_rep2(M_j, W, N_j, C_j):
    [alpha_WQ, alpha_MQ, alpha_NQ, alpha_CQ] = calcule_param2(W, M_j, N_j, C_j)
    E_QW  = (a_Q * alpha_WQ * W)  /(1 + a_Q * (alpha_WQ * h_W * W + alpha_MQ * h_MQ * M_j + alpha_NQ * h_NQ * N_j + alpha_CQ * h_CQ * C_j))
    E_QMj = (a_Q * alpha_MQ * M_j)/(1 + a_Q * (alpha_WQ * h_W * W + alpha_MQ * h_MQ * M_j + alpha_NQ * h_NQ * N_j + alpha_CQ * h_CQ * C_j))
    E_QNj = (a_Q * alpha_NQ * N_j)/(1 + a_Q * (alpha_WQ * h_W * W + alpha_MQ * h_MQ * M_j + alpha_NQ * h_NQ * N_j + alpha_CQ * h_CQ * C_j))
    E_QCj = (a_Q * alpha_CQ * C_j)/(1 + a_Q * (alpha_WQ * h_W * W + alpha_MQ * h_MQ * M_j + alpha_NQ * h_NQ * N_j + alpha_CQ * h_CQ * C_j))
    return E_QW, E_QMj, E_QNj, E_QCj


def calcule_params_pertubes(t):
    """
        Parametres de la modelisation
        Les parametres qui ont deux valeurs distinctes pour l'Est et l'Ouest sont
        sous forme d'une liste de deux valeurs: [valeur_est, valeur_ouest]
    """

    tpertub = 450; #choisir. Exemples a la page 60 du document de Lamirande
    tfaible = tpertub + 5
    tpic = tfaible + (50 - 25 * PP)
    tstable = tpic + 100
    k_U = k_UStable
    k_V = k_VNorm
    k_W = k_WStable
    if t < tfaible and t >= tpertub:
        k_V = k_VFaible
        k_U = k_UFaible
        k_W = k_WFaible
    elif t >= tfaible and t < tpic:
        k_U = k_UPic
        k_W = k_WPic
    elif t >= tpic and t < tstable:
        k_U = k_UPic + (t - tpic) * ((k_UStable - k_UPic)/(tstable - tpic))
        k_W = k_WPic + (t - tpic) * ((k_WStable - k_WPic)/(tstable - tpic))

    return k_U, k_V, k_W

class LamirandeSystem(EDOs):
    def __init__(self, t_init, t_final, initial_state):
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
        
    def evalue(self, t, Y):
        U, V, W, M_j, M_A, N_j, N_A, C_j, C_A, P, Q = Y
        E_PMj, E_PMA, E_PNj, E_PNA, E_PCj, E_PCA = calcule_rep1(M_j, M_A, N_j, N_A, C_j, C_A)
        E_QW, E_QMj, E_QNj, E_QCj = calcule_rep2(M_j, W, N_j, C_j)
        k_U, k_V, k_W = calcule_params_pertubes(t)
        dUdt  = u0 * (1 - U/k_U) - a_M * U * (M_j + M_A)/(1 + a_M * h_UM * U) - a_C * U *(C_j + C_A)/(1 + a_C * h_UC * U)

        dVdt  = v0 * V * (1 - V/k_V)- a_N * V * (N_j + N_A)/(1 + a_N * h_VN * V)
        dWdt  = w0 * (1 - W/k_W) - E_QW * Q
        dMjdt = khi_M * ((a_M * e_UM * U/(1 + a_M * h_UM * U)) - mu_M) * M_A - tau_M * M_j - (m0/k_M) * M_j * (M_j + M_A) - E_PMj * P - E_QMj * Q
        dMAdt = tau_M * M_j - (m0/k_M) * M_A * (M_j + M_A) - E_PMA * P
        dNjdt = khi_N * ((a_N * e_VN * V/(1 + a_N * h_VN * V)) - mu_N) * N_A - tau_N * N_j - (n0/k_N) * N_j * (N_j + N_A) - E_PNj * P - E_QNj * Q
        dNAdt = tau_N * N_j - (n0/k_N) * N_A * (N_j + N_A) - E_PNA * P
        dCjdt = khi_C * ((a_C * e_UC * U/(1 + a_C * h_UC * U)) - mu_C) * C_A - tau_C * C_j - (c0/k_C) * C_j * (C_j + C_A) - E_PCj * P - E_QCj * Q
        dCAdt = tau_C * C_j - (c0/k_C) * C_A *(C_j + C_A) - E_PCA * P
        dPdt  = khi_P * (E_PMA + phi_MAj * E_PMj + phi_MN * (E_PNA + phi_NAj * E_PNj) + phi_MC * (E_PCA + phi_CAj * E_PCj) - mu_P) * P -(p0/k_P) * P**2
        dQdt  = khi_Q * (E_QW + phi_WM * E_QMj + phi_WN * E_QNj + phi_WC * E_QCj - mu_Q) * Q - (q0/k_Q) * Q**2

        return np.array([dUdt, dVdt, dWdt, dMjdt, dMAdt, dNjdt, dNAdt, dCjdt, dCAdt, dPdt, dQdt])
    

if __name__ == '__main__':

    # Initial conditions
    t0 = 0.0
    tf = 900
    Y = [4e5, 5e5, 4.5e3, 0.02, 0.18, 0.02, 0.18, 0.08, 0.46, 0.004, 0.1]
    systeme = LamirandeSystem(t_init=t0, 
                              t_final=tf, 
                              initial_state=Y)

    # solver
    solver = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name('esdirk6'), 
                                           verbose=True, 
                                           progress_interval_in_time=1.0, 
                                           max_jacobian_refresh=1,
                                           export_interval=100000,
                                           export_prefix="resultats/lamirande_model")

    print(TableauDeButcher.from_name('esdirk6'))
    start=time.time()
    times, solutions = solver.solve(
        systeme_EDOs=systeme, 
        initial_step_size=0.0001,
        adaptive_time_stepping=False,
        target_relative_error=1e-8,
        min_step_size=1e-10,
        max_step_size=100,
        )
    Elapsed = time.time() - start
    print(Elapsed)
    print("Saving data to CSV...")
    results = np.column_stack((times, solutions))
    header = "time,U,V,W,Mj,MA,Nj,NA,Cj,CA,P,Q"
    np.savetxt('lamirande.csv', results, delimiter=',', header=header, comments='')
    print("Data saved to lorenz_system.csv")

# Filter data for t >= 400
mask = times >= 0.0
t_sub = times[mask]
sol_sub = solutions[mask]

# ================
# FIGURE 1: U, V, W
# ================
fig1, ax1 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax1[0].plot(t_sub, sol_sub[:, 0], 'b-')
ax1[0].set_ylabel("U(t)")
ax1[0].grid(True)

ax1[1].plot(t_sub, sol_sub[:, 1], 'r-')
ax1[1].set_ylabel("V(t)")
ax1[1].grid(True)

ax1[2].plot(t_sub, sol_sub[:, 2], 'g-')
ax1[2].set_ylabel("W(t)")
ax1[2].set_xlabel("Time")
ax1[2].grid(True)

fig1.suptitle("Evolution of U, V, W (t ≥ 400)")
plt.tight_layout()
plt.show()

# =====================
# FIGURE 2: Aggregates
# =====================
Nj_plus_NA = sol_sub[:, 5] + sol_sub[:, 6]
Mj_plus_MA = sol_sub[:, 3] + sol_sub[:, 4]
Cj_plus_CA = sol_sub[:, 7] + sol_sub[:, 8]
Q_vals = sol_sub[:, 10]
P_vals = sol_sub[:, 9]

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(t_sub, Nj_plus_NA, label="N_j + N_A")
ax2.plot(t_sub, Mj_plus_MA, label="M_j + M_A")
ax2.plot(t_sub, Cj_plus_CA, label="C_j + C_A")
ax2.plot(t_sub, Q_vals, label="Q")

ax2.set_title("Aggregated Populations and Q (t ≥ 400)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True)
plt.show()

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.plot(t_sub, P_vals, label="P")

ax3.set_title("P (t ≥ 400)")
ax3.set_xlabel("Time")
ax3.set_ylabel("Value")
ax3.legend()
ax3.grid(True)
plt.show()