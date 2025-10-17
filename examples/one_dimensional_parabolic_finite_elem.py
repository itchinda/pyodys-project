import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# ---------------------------------------------------------------------
# Finite Element Assembly
# ---------------------------------------------------------------------
def fem_matrices_1d(N, kappa=1.0):
    """
    Assemble FEM mass and stiffness matrices for 1D diffusion
    using N elements and N+1 nodes.
    Dirichlet BCs applied at both ends ===> system size = N-1.
    """
    h = 1.0 / N
    M_local = (h / 6.0) * np.array([[2, 1], [1, 2]])
    K_local = (kappa / h) * np.array([[1, -1], [-1, 1]])

    rows, cols, data_M, data_K = [], [], [], []
    for e in range(N):
        for a in range(2):
            for b in range(2):
                i = e + a
                j = e + b
                rows.append(i)
                cols.append(j)
                data_M.append(M_local[a, b])               # Crucial! when two elements share a node, this process 
                data_K.append(K_local[a, b])               # records two separate entries at the same global index (i,j).

    M = sp.coo_matrix((data_M, (rows, cols)), shape=(N + 1, N + 1)).tocsc()   # The key feature of the COO format here is that when multiple entries 
    K = sp.coo_matrix((data_K, (rows, cols)), shape=(N + 1, N + 1)).tocsc()   # are recorded for the same (i,j) index (i.e., when elements share 
                                                                              # a node), the final sparse matrix automatically sums these values.
    # Apply Dirichlet BCs (remove first and last rows/cols)
    M = M[1:-1, 1:-1]
    K = K[1:-1, 1:-1]

    x = np.linspace(0, 1, N + 1)[1:-1]  # interior points
    return M, K, x, h

# ---------------------------------------------------------------------
# Define the ODE Problem
# ---------------------------------------------------------------------
class HeatFEMProblem(ODEProblem):
    """1D Heat equation M du/dt + κ K u = 0."""
    def __init__(self, N, kappa=1.0):
        M, K, x, h = fem_matrices_1d(N, kappa)
        self.M = M
        self.K = K
        self.kappa = kappa
        self.h = h
        self.x = x
        
        # Initial condition is applied to the interior nodes
        u0 = np.sin(np.pi * x)

        super().__init__(
            t_init=0.0,
            t_final=1.0,
            initial_state=u0,
            mass_matrix_is_constant=True,   # Optional, but important for optimization. Store the mass matrix and avoid recomputing every steps
            jacobian_is_constant=True,      # Optional, store jacobian and avoid recomputing every steps
            jacobian_is_sparse=True
        )

    def _compute_mass_matrix(self, t, U):
        # M is the constant mass matrix for M du/dt = F(U)
        return self.M

    def evaluate_at(self, t, U):
        # F(U) = -κ K U (the right-hand side of the DAE: M du/dt = F(U))
        return -self.K.dot(U)

    def jacobian_at(self, t, U):
        # J(U) = dF/dU = -κ K (The Jacobian of the right-hand side)
        return -self.K

    def exact_solution(self, t):
        # u(x,t) = sin(PI x) * exp(-κ PI^2 t)
        return np.sin(np.pi * self.x) * np.exp(-self.kappa * np.pi**2 * t)

# ---------------------------------------------------------------------
# Run the Solver
# ---------------------------------------------------------------------
if __name__ == "__main__":
    N = 10000  # number of elements ===> N-1 = 49 DOFs
    problem = HeatFEMProblem(N, kappa=0.25)
    solver = PyodysSolver(
        method="sdirk43",
        atol=1e-10,
        rtol=1e-10,
        min_step=1e-8,
        linear_solver="lu"   # Will automatically select scipy sparse lu (splu)
    )

    times, U = solver.solve(problem)

    # Compare with exact solution
    U_exact = problem.exact_solution(problem.t_final)
    U_num = U[-1, :]
    err = np.linalg.norm(U_num - U_exact) / np.linalg.norm(U_exact)
    print(f"Number of DOFs (N-1): {problem.M.shape[0]}")
    print(f"Relative L2 error = {err:.2e}")

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1D snapshot at final time
    ax[0].plot(problem.x, U_num, "r-", label="Numerical")
    ax[0].plot(problem.x, U_exact, "k--", label="Exact")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("u(x)")
    ax[0].set_title(f"Heat equation at t={problem.t_final}")
    ax[0].legend()
    ax[0].grid(True)

    # 2D space-time map
    U_map = U.T
    im = ax[1].imshow(U_map, aspect="auto",
                      extent=[times[0], times[-1], problem.x[0], problem.x[-1]],
                      origin="lower", cmap="inferno")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("x")
    ax[1].set_title("u(x,t)")
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.show()