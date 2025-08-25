import numpy as np
import matplotlib.pyplot as plt

# Using the top-level imports from the installed package
from EDOsSolverModules import EDOs
from EDOsSolverModules import TableauDeButcher
from EDOsSolverModules import SolveurRKAvecTableauDeButcher

# Define coupled linear system
class SystemeCouple(EDOs):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)
    
    def evalue(self, t, u):
        x, y = u
        return np.array([-x + y, -y])

# Analytical solution
def solution_analytique(t, u0):
    tau = t - 0.0
    x0, y0 = u0
    x = np.exp(-tau) * (x0 + y0 * tau)
    y = y0 * np.exp(-tau)
    return np.array([x, y])

if __name__ == '__main__':
    # Initial conditions
    t_init = 0.0
    t_final = 10.0
    u_init = [1.0, 1.0]
    systeme = SystemeCouple(t_init, t_final, u_init)

    # Use a SDIRK solver for demonstration
    solver_sdirk = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name('sdirk_hairer_norsett_wanner_45'))
    times, solutions = solver_sdirk.solve(
        systeme,
        initial_step_size=0.01,
        adaptive_time_stepping=True,
        target_relative_error=1e-6,
        min_step_size=1e-6,
        max_step_size=1.0
    )

    # Compute analytical solution and errors
    analytical_solutions = np.array([solution_analytique(t, u_init) for t in times])
    error = np.linalg.norm(solutions - analytical_solutions, axis=1)

    # Plot solutions and errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(times, solutions[:, 0], 'b-', label='x(t) Numerical')
    ax1.plot(times, solutions[:, 1], 'r-', label='y(t) Numerical')
    ax1.plot(times, analytical_solutions[:, 0], 'k--', label='x(t) Analytical')
    ax1.plot(times, analytical_solutions[:, 1], 'g--', label='y(t) Analytical')
    ax1.set_title("Coupled Linear System: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, error, 'b-', label='L2 Norm Error')
    ax2.set_yscale('log')
    ax2.set_title("Error vs Analytical Solution")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    plt.savefig("examples/figures/quick_example.png")
    
    plt.show()
    