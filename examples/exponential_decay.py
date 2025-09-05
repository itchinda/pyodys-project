import numpy as np
import matplotlib.pyplot as plt

from pyodys import ODEProblem, ButcherTableau, RKSolverWithButcherTableau

# Define the Robertson stiff system
class Robertson(ODEProblem):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)

    def evalue(self, t, u):
        y1, y2, y3 = u
        dy1 = -0.04*y1 + 1.0e4*y2*y3
        dy2 = 0.04*y1 - 1.0e4*y2*y3 - 3.0e7*y2*y2
        dy3 = 3.0e7*y2*y2
        return np.array([dy1, dy2, dy3])

# Initial conditions and time span
t_init, t_final = 0.0, 1e5
u_init = [1.0, 0.0, 0.0]
robertson = Robertson(t_init, t_final, u_init)

# Adaptive SDIRK solver (implicit, stiff-friendly)
solver = RKSolverWithButcherTableau(
                                        tableau_de_butcher=ButcherTableau.from_name("sdirk_norsett_thomson_23"),
                                        initial_step_size=1e-6,
                                        adaptive_time_stepping=True,
                                        min_step_size=1e-12,
                                        max_step_size=10000.0,
                                        target_relative_error=1e-6,
                                        verbose=False
)

# Solve system
times, sol = solver.solve(robertson)

# Compute adaptive step sizes
dt = np.diff(times)

# --- Plot concentrations ---
plt.figure(figsize=(10,6))
plt.plot(times, sol[:,0], label="y1")
plt.plot(times, sol[:,1], label="y2")
plt.plot(times, sol[:,2], label="y3")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (log scale)")
plt.ylabel("Concentration (log scale)")
plt.title("Robertson Stiff Problem: Adaptive SDIRK Solution")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot adaptive step sizes ---
plt.figure(figsize=(10,6))
plt.plot(times[:-1], dt, 'r.-')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Step size Δt")
plt.title("Adaptive Step Sizes Over Time (Stiff Robertson ODE)")
plt.grid(True, which="both")
plt.show()