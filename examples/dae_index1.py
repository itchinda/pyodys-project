import numpy as np
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# --- New Index 1 DAE Example: Simple Circuit Constraint ---
class SimpleCircuitDAE(ODEProblem):
    """
    Index 1 DAE System:
      y' = -y - z + E(t)
      0  = -y + z + 1
      E(t) = sin(t)

    This is a semiexplicit Index 1 DAE.
    - Differential variable: y
    - Algebraic variable: z

    Exact solution (from constraint, z = y - 1):
    y' = -y - (y - 1) + sin(t)
    y' = -2y + 1 + sin(t)
    y(t) = C*e^{-2t} + (5 - 4*sin(t) - 2*cos(t)) / 10
    """
    def __init__(self, y0=1.0, t_init=0.0, t_final=2.0*np.pi):
        # Initial condition: y(0)=y0.
        # Initial z (algebraic) must satisfy the constraint: z0 = y0 - 1.0
        z0 = y0 - 1.0
        initial_state = np.array([y0, z0])
        
        # M = [[1, 0], [0, 0]]
        super().__init__(t_init, t_final, initial_state, 
                         mass_matrix_is_constant=True, jacobian_is_constant=False)
        self.M = np.array([[1.0, 0.0], [0.0, 0.0]])
        self.y0 = y0

    def _compute_mass_matrix(self, t, state):
        # M is constant: [[1, 0], [0, 0]]
        return self.M

    def evaluate_at(self, t, u):
        # f(t, u) = [y', 0]
        y, z = u
        E_t = np.sin(t) # External source
        
        dy_dt = -y - z + E_t        # Differential equation
        alg_constraint = -y + z + 1.0 # Algebraic equation
        
        return np.array([dy_dt, alg_constraint], dtype=float)

    def jacobian_at(self, t, u):
        # J = dF/du = [[d(y')/dy, d(y')/dz], [d(0)/dy, d(0)/dz]]
        # J = [[-1, -1], [-1, 1]]
        return np.array([[-1.0, -1.0], [-1.0, 1.0]], dtype=float)

    def exact(self, times):
        y0 = self.y0
        
        # C is determined from y(0)=y0: C = y0 - (5 - 2)/10 = y0 - 0.3
        C = y0 - 3/10
        
        def y_exact(t):
            return C * np.exp(-2.0*t) + (5.0 + 4.0*np.sin(t) - 2.0*np.cos(t)) / 10.0
            
        ys = np.array([y_exact(t) for t in times])
        # From constraint: z = y - 1
        zs = ys - 1.0
        
        return np.vstack([ys, zs]).T

if __name__ == "__main__":
    t_init, t_final = 0.0, 2.0 * np.pi
    y_init = 1.0
    
    # y(0)=1.0, z(0) must be 0.0 for consistency (z=y-1)
    systeme = SimpleCircuitDAE(y0=y_init, t_final=t_final)

    # *** Use a DIRK scheme (sdirk21) which is excellent for Index 1 DAEs ***
    solver = PyodysSolver(
        method="sdirk2",  # Diagonally Implicit Runge-Kutta
        adaptive=True,
        verbose=True,
        linear_solver="lu"
    )

    times, solutions = solver.solve(systeme)

    analytical = systeme.exact(times)
    error = np.linalg.norm(solutions - analytical, axis=1)

    # Plotting results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Solution Plot
    ax1.plot(times, solutions[:, 0], 'b-', label='y(t) numerical (Differential)')
    ax1.plot(times, solutions[:, 1], 'r-', label='z(t) numerical (Algebraic)')
    ax1.plot(times, analytical[:, 0], 'k--', label='y(t) analytical')
    ax1.plot(times, analytical[:, 1], 'g--', label='z(t) analytical')
    ax1.set_title("Index 1 DAE: Differential (y) and Algebraic (z) Variables")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # Error Plot
    ax2.plot(times, error, 'b-', label='L2 Norm Error')
    ax2.set_yscale('log')
    ax2.set_title(f"Error for DAE Solution (DIRK method)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()