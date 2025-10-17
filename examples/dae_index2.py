import numpy as np
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# --- Index 2 DAE Example: Manufactured Solution ---
class Index2DAE(ODEProblem):
    """
    Index 2 DAE System:
      y' = z                 (Differential Eq.)
      0  = y - sin(t)        (Algebraic Constraint)

    Manufactured Exact Solution:
      y_ex(t) = sin(t)
      z_ex(t) = cos(t)
    Initial conditions (consistent at t=0): y(0)=0, z(0)=1
    """
    def __init__(self, t_init=0.0, t_final=2.0 * np.pi):
        # Consistent initial state: y(0)=0, z(0)=1
        initial_state = np.array([np.sin(t_init), np.cos(t_init)])
        
        # M = [[1, 0], [0, 0]]
        super().__init__(t_init, t_final, initial_state, 
                         mass_matrix_is_constant=True, jacobian_is_constant=False)
        self.M = np.array([[1.0, 0.0], [0.0, 0.0]])

    def _compute_mass_matrix(self, t, state):
        # M is constant: [[1, 0], [0, 0]]
        return self.M

    def evaluate_at(self, t, u):
        # We need to return the right-hand side f(t, u) such that M u' = f(t, u)
        y, z = u
        
        f1 = z                  # -> y' = z
        f2 = y - np.sin(t)      # -> 0 = y - sin(t) (The algebraic constraint)
        
        return np.array([f1, f2], dtype=float)

    def jacobian_at(self, t, u):
        # J = dF/du = [[d(f1)/dy, d(f1)/dz], [d(f2)/dy, d(f2)/dz]]
        # J = [[0, 1], [1, 0]]
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    def exact(self, times):
        ys = np.sin(times)
        zs = np.cos(times)
        return np.vstack([ys, zs]).T

if __name__ == "__main__":
    t_init, t_final = 0.0, 2.0 * np.pi
    systeme = Index2DAE(t_final=t_final)

    # *** Using the DIRK scheme (sdirk21) on the Index 2 DAE ***
    print("--- Solving Index 2 DAE with DIRK (sdirk4) ---")
    
    # CRITICAL WARNING for the user, specific to high-index DAEs
    print("\n--- WARNING: Index-2 DAE Instability ---")
    print("Standard adaptive time stepping is generally unreliable or unstable for Index-2 DAEs.")
    print("Solving with a fixed, small step to avoid adaptive step size control failure.")
    print("-----------------------------------------\n")
    
    solver = PyodysSolver(
        method="sdirk4",       # DIRK method
        # Implementation of the user's warning/intention:
        adaptive=False,        # DISABLED to avoid instability issues common in adaptive DAE stepping
        fixed_step=1e-4,       # Using a fixed, small step instead
        verbose=True,
        linear_solver="lu",
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
    ax1.set_title("Index 2 DAE: Differential (y) and Algebraic (z) Variables")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # Error Plot
    ax2.plot(times, error, 'b-', label='L2 Norm Error')
    ax2.set_yscale('log')
    ax2.set_title(f"Error for Index 2 DAE (DIRK method)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()