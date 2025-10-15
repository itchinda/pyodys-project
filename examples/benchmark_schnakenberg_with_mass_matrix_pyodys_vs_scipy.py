import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, bmat, block_diag
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import pyodys as pod

# ---------------------------------------------------------------------------
# Linearized 1D Schnakenberg with time-dependent non-diagonal mass
# ---------------------------------------------------------------------------
D_u, D_v = 1e-3, 5e-3
alpha, beta = -1.0, 2.0

def laplacian_1d(N, h):
    main_diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)
    return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csc") / h**2

class LinearSchnakenbergMass(pod.ODEProblem):
    """Linear Schnakenberg with time-dependent non-diagonal mass matrix M(t)."""
    def __init__(self, N, t_init=0.0, t_final=0.5):
        self.N = N
        self.h = 1.0 / (N + 1)
        u0 = np.sin(np.pi * np.linspace(self.h, 1 - self.h, N))
        v0 = np.cos(np.pi * np.linspace(self.h, 1 - self.h, N))
        initial_state = np.hstack([u0, v0])
        super().__init__(t_init, t_final, initial_state,
                         mass_matrix_is_constant=False,
                         jacobian_is_constant=True)
        I = sp.eye(N)
        L = laplacian_1d(N, self.h)
        self.A = bmat([[D_u*L + alpha*I, beta*I],
                       [beta*I, D_v*L + alpha*I]], format="csc")

    def _compute_mass_matrix(self, t, U):
        # Time-dependent 2x2 blocks, non-diagonal
        blocks = [np.array([[1 + 0.1*np.sin(t), 0.05*np.cos(t)],
                            [0.05*np.cos(t), 1 + 0.05*np.sin(t)]]) for _ in range(self.N)]
        return block_diag(blocks).tocsc()

    def evaluate_at(self, t, U):
        return self.A.dot(U)

    def jacobian_at(self, t, U):
        return self.A

# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
Ns = [25, 50, 100, 200, 400, 800]
solver_opts = dict(method="esdirk64",
                   first_step=1e-6,
                   adaptive=True,
                   min_step=1e-8,
                   max_step=0.5,
                   atol=1e-8,
                   rtol=1e-8,
                   verbose=False)

results = []

for N in Ns:
    prob = LinearSchnakenbergMass(N, t_init=0.0, t_final=0.5)

    # --- PyOdys solver ---
    solver = pod.PyodysSolver(auto_check_sparsity=True, **solver_opts)
    t0 = time.perf_counter()
    times, sol_pyodys = solver.solve(prob)
    t_pyodys = time.perf_counter() - t0
    U_pyodys = sol_pyodys[-1, :]

    # --- SciPy solver with sparse M(t) ---
    def fun_scipy_sparse(t, U):
        M = prob._compute_mass_matrix(t, U)
        F = prob.evaluate_at(t, U)
        return spsolve(M, F)

    t0 = time.perf_counter()
    sol_scipy_sparse = solve_ivp(
        fun_scipy_sparse,
        [prob.t_init, prob.t_final],
        prob.initial_state,
        method="BDF",
        rtol=solver_opts["rtol"],
        atol=solver_opts["atol"],
        first_step=solver_opts["first_step"],
        max_step=solver_opts["max_step"]
    )
    t_scipy = time.perf_counter() - t0
    U_scipy = sol_scipy_sparse.y[:, -1]

    # --- Reference solution using very small steps (SciPy Radau) ---
    def fun_ref(t, U):
        M = prob._compute_mass_matrix(t, U)
        F = prob.evaluate_at(t, U)
        return spsolve(M, F)

    sol_ref = solve_ivp(
        fun_ref,
        [prob.t_init, prob.t_final],
        prob.initial_state,
        method="Radau",
        rtol=1e-12,
        atol=1e-12
    )
    U_ref = sol_ref.y[:, -1]

    # --- Compute relative L2 errors ---
    err_pyodys = np.linalg.norm(U_pyodys - U_ref) / np.linalg.norm(U_ref)
    err_scipy  = np.linalg.norm(U_scipy - U_ref) / np.linalg.norm(U_ref)

    results.append((N, 2*N, t_pyodys, t_scipy, err_pyodys, err_scipy))
    print(f"N={N}, 2N={2*N}, t_pyodys={t_pyodys:.3f}, t_scipy={t_scipy:.3f}, "
          f"err_pyodys={err_pyodys:.2e}, err_scipy={err_scipy:.2e}")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
print("\\begin{table}[ht!]")
print("\\centering")
print("\\begin{tabular}{c|c|c|c|c|c}")
print("\\hline")
print("$N$ & DoF ($2N$) & PyOdys time & SciPy time & Rel. L2 err (PyOdys) & Rel. L2 err (SciPy) \\\\")
print("\\hline")
for N, dof, t_py, t_sp, err_p, err_s in results:
    print(f"{N} & {dof} & {t_py:.3f} & {t_sp:.3f} & {err_p:.2e} & {err_s:.2e} \\\\")
    print("\\hline")
print("\\end{tabular}")
print("\\caption{CPU runtimes and relative $L^2$ errors for 1D linearized Schnakenberg system with time-dependent non-diagonal mass matrix. PyOdys uses DIRK solver; SciPy uses sparse Radau solver.}")
print("\\label{tab:schnakenberg1d_mass}")
print("\\end{table}")
