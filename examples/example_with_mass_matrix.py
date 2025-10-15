import numpy as np
import matplotlib.pyplot as plt

# Using the top-level imports from the installed package
from pyodys import ODEProblem, PyodysSolver

# Define coupled linear system
class SystemeEDO(ODEProblem):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)
        self.mass_matrix_is_constant = True
        self.jacobian_is_constant =True
    
    def evaluate_at(self, t, u):
        x, y = u
        Ay = np.array([y, -x], dtype=float)
        Ay_exact = np.array([np.cos(t), -np.sin(t),], dtype=float)
        My_prime_exact = np.array([2*np.cos(t) - np.sin(t), np.cos(t) - 2*np.sin(t)], dtype=float)
        ft = My_prime_exact - Ay_exact
        return Ay + ft
    
    def _compute_mass_matrix(self, t, state):
        x,y=state
        M = np.array([
            [2, 1],
            [1, 2]
        ], dtype=float)
        return M

# Analytical solution
def solution_analytique(t):
    x = np.sin(t)
    y = np.cos(t)
    return np.array([x, y])

if __name__ == '__main__':
    # Initial conditions
    t_init = 0.0
    t_final = 2.0*np.pi
    u_init = [0.0, 1.0]
    systeme = SystemeEDO(t_init, t_final, u_init)

    # Using a SDIRK solver
    solver_sdirk = PyodysSolver(
        method = "sdirk43",
        fixed_step=1e-3,                          
        first_step = None,
        adaptive = False,
        atol = 1e-10,
        rtol = 1e-10,
        min_step = 1e-6,
        max_step = 1.0,
        verbose=True
    )
    
    times, solutions = solver_sdirk.solve( systeme )

    # Compute analytical solution and errors
    analytical_solutions = np.array([solution_analytique(t) for t in times])
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
    
    #plt.savefig("results.png")
    
    plt.show()
    