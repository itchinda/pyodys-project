import numpy as np
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# Singular-mass DAE system
class SimpleIndex1DAE(ODEProblem):
    """
    DAE:
      M u' = f(t,u),  with M = [[0,0],[0,1]]
      f = [-x + y + sin t,  -x - y]
    Exact solution: y(t) = (y0 - 1/5) e^{-2t} - (2 sin t - cos t)/5, x = y + sin t
    """
    def __init__(self, y0=1.0, t_init=0.0, t_final=1.0):
        # note: mass_matrix_is_identity=False, mass_matrix_is_constant=True
        super().__init__(t_init, t_final, np.array([y0, y0]),  # x0 = y0 ensures consistency
                          mass_matrix_is_constant=True, jacobian_is_constant=True)
        self.M = np.array([[0.0, 0.0], [0.0, 1.0]])

    def _compute_mass_matrix(self, t, state):
        return self.M

    def evaluate_at(self, t, u):
        x, y = u
        return np.array([-x + y + np.sin(t), -x - y], dtype=float)

    def jacobian_at(self, t, u):
        # constant Jacobian
        return np.array([[-1.0, 1.0], [-1.0, -1.0]], dtype=float)

    def exact(self, times):
        y0 = self.initial_state[1]
        def y_exact(t):
            return (y0 - 0.2)*np.exp(-2.0*t) - (2.0*np.sin(t) - np.cos(t))/5.0
        ys = np.array([y_exact(t) for t in times])
        xs = ys + np.sin(times)
        return np.vstack([xs, ys]).T

if __name__ == "__main__":
    t_init, t_final = 0.0, 2.0 * np.pi
    u_init = [0.0, 1.0]  # consistent since x(0)=0, y(0)=1
    systeme = SimpleIndex1DAE()

    solver = PyodysSolver(
        method="sdirk211",   # implicit solver suitable for DAE
        first_step=None,
        adaptive=True,
        atol=1e-10,
        rtol=1e-10,
        min_step=1e-6,
        max_step=1.0,
        verbose=True,
        linear_solver="lu"
    )

    times, solutions = solver.solve(systeme)

    analytical = systeme.exact(times)
    error = np.linalg.norm(solutions - analytical, axis=1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(times, solutions[:, 0], 'b-', label='x(t) numerical')
    ax1.plot(times, solutions[:, 1], 'r-', label='y(t) numerical')
    ax1.plot(times, analytical[:, 0], 'k--', label='x(t) analytical')
    ax1.plot(times, analytical[:, 1], 'g--', label='y(t) analytical')
    ax1.set_title("Singular Mass Matrix System (x ≠ y)")
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
    plt.show()
