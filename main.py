import numpy as np
import matplotlib.pyplot as plt
from systemes.EDOs import EDOs
from solveurs.runge_kutta.TableauDeButcher import TableauDeButcher
from solveurs.runge_kutta.SolveurRKAvecTableauDeButcher import SolveurRKAvecTableauDeButcher

# Define coupled linear system
class SystemeCouple(EDOs):
    def __init__(self, temps_initial, condition_initiale):
        super().__init__(temps_initial, condition_initiale)
    def evalue(self, t, u):
        x, y = u
        return np.array([-x + y, -y])

# Analytical solution
def solution_analytique(t, t0, u0):
    x0, y0 = u0
    tau = t - t0
    x = np.exp(-tau)*(x0 + y0*tau)
    y = y0 * np.exp(-tau)
    return np.array([x, y])

# Initial conditions
t0 = 0.0
u0 = [1.0, 1.0]
systeme = SystemeCouple(t0, u0)

# RK4 solver
solveur_rk4 = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name('rk4'))
t_rk4, sol_rk4 = solveur_rk4.solve(systeme, pas_de_temps=0.01, nb_pas_de_temps_max=1000)

# Compute analytical solution
sol_ana = np.array([solution_analytique(ti, t0, u0) for ti in t_rk4])

# Compute errors
err_rk4 = np.linalg.norm(sol_rk4 - sol_ana, axis=1)

# Plot solutions
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(t_rk4, sol_rk4[:,0], 'b.-', markersize=2, label='x(t) RK4')
plt.plot(t_rk4, sol_rk4[:,1], 'r-', markersize=2, label='y(t) RK4')
plt.title("Coupled Linear System: Solutions")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(t_rk4, err_rk4, 'b-', label='Error RK4')
plt.yscale('log')
plt.title("Error vs Analytical Solution")
plt.xlabel("Time")
plt.ylabel("L2 Norm Error")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("quick_example.png")

plt.show()
