import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, bmat
from scipy.integrate import solve_ivp

# from pyodys import ODEProblem, PyodysSolver
import pyodys as pod

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
a, b = 0.2, 1.3        # Schnakenberg reaction parameters
D_u, D_v = 1e-3, 5e-3  # Diffusion coefficients


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def laplacian_1d(N: int, h: float) -> sp.csc_matrix:
    """Construct 1D Laplacian with Dirichlet BC."""
    main_diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)
    return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csc") / (h**2)


# ---------------------------------------------------------------------------
# Schnakenberg 1D problem (PyOdys interface)
# ---------------------------------------------------------------------------
class Schnakenberg1D(pod.ODEProblem):
    """1D Schnakenberg reaction-diffusion system."""

    def __init__(self, N: int, t_init: float, t_final: float, force_dense: bool = False):
        self.N = N
        self.h = 1.0 / (N + 1)
        self.x = np.linspace(self.h, 1.0 - self.h, N)
        self.force_dense = force_dense

        # Steady-state solution with perturbation
        u0 = a + b
        v0 = b / (a + b) ** 2
        rng = np.random.default_rng(12345)
        perturb = 0.02 * rng.standard_normal(size=(N,))
        initial_state = np.hstack([u0 + perturb, v0 + perturb])

        self.L = laplacian_1d(N, self.h)

        super().__init__(
            t_init,
            t_final,
            initial_state,
            jacobian_is_constant=False
        )

    def evaluate_at(self, t: float, U: np.ndarray) -> np.ndarray:
        u = U[: self.N]
        v = U[self.N :]
        rhs_u = D_u * self.L.dot(u) + (a - u + u**2 * v)
        rhs_v = D_v * self.L.dot(v) + (b - u**2 * v)
        return np.concatenate([rhs_u, rhs_v])

    def jacobian_at(self, t: float, U: np.ndarray):
        u = U[: self.N]
        v = U[self.N :]
        n = self.N
        if self.force_dense:
            J = np.zeros((2 * n, 2 * n))
            J[:n, :n] = D_u * self.L.toarray() + np.diag(-1 + 2 * u * v)
            J[:n, n:] = np.diag(u**2)
            J[n:, :n] = np.diag(-2 * u * v)
            J[n:, n:] = D_v * self.L.toarray() + np.diag(-u**2)
            return J
        else:
            R_uu = diags(-1 + 2 * u * v, 0, format="csc")
            R_uv = diags(u**2, 0, format="csc")
            R_vu = diags(-2 * u * v, 0, format="csc")
            R_vv = diags(-u**2, 0, format="csc")
            A11 = D_u * self.L + R_uu
            A12 = R_uv
            A21 = R_vu
            A22 = D_v * self.L + R_vv
            return bmat([[A11, A12], [A21, A22]], format="csc")


# ---------------------------------------------------------------------------
# SciPy Problem Wrapper (uses same Laplacian)
# ---------------------------------------------------------------------------
def schnakenberg_scipy(t, U, L, N):
    u = U[:N]
    v = U[N:]
    rhs_u = D_u * L.dot(u) + (a - u + u**2 * v)
    rhs_v = D_v * L.dot(v) + (b - u**2 * v)
    return np.concatenate([rhs_u, rhs_v])


def schnakenberg_scipy_jac(t, U, L, N):
    u = U[:N]
    v = U[N:]
    R_uu = diags(-1 + 2 * u * v, 0, format="csc")
    R_uv = diags(u**2, 0, format="csc")
    R_vu = diags(-2 * u * v, 0, format="csc")
    R_vv = diags(-u**2, 0, format="csc")
    A11 = D_u * L + R_uu
    A12 = R_uv
    A21 = R_vu
    A22 = D_v * L + R_vv
    return bmat([[A11, A12], [A21, A22]], format="csc")


# ---------------------------------------------------------------------------
# CPU-Time + Accuracy vs N
# ---------------------------------------------------------------------------
Ns = [25, 50, 100, 200, 400, 800, 5000, 10000, 20000, 40000]
t_final = 0.5

solver_opts = {
    "method": "esdirk64",  # PyOdys RK scheme
    "first_step": 1e-6,
    "adaptive": True,
    "min_step": 1e-10,
    "max_step": 5e-1,
    "atol": 1e-10,
    "rtol": 1e-10,
    "verbose": False,
}

results = []

for N in Ns:
    prob_sparse = Schnakenberg1D(N, 0.0, t_final, force_dense=False)

    # --- PyOdys Sparse ---
    solver_sparse = pod.PyodysSolver(auto_check_sparsity=True, **solver_opts)
    t0 = time.perf_counter()
    _, sol_pyodys = solver_sparse.solve(prob_sparse)
    t_pyodys_sparse = time.perf_counter() - t0
    U_pyodys = sol_pyodys[-1,:]

    # --- SciPy Radau ---
    L = laplacian_1d(N, 1.0 / (N + 1))
    t0 = time.perf_counter()
    sol_scipy = solve_ivp(
        fun=lambda t, U: schnakenberg_scipy(t, U, L, N),
        t_span=[0, t_final],
        y0=prob_sparse.initial_state,
        method="Radau",
        rtol=solver_opts["rtol"],
        atol=solver_opts["atol"],
        first_step=solver_opts["first_step"],
        max_step=solver_opts["max_step"],
        jac=lambda t, U: schnakenberg_scipy_jac(t, U, L, N),
    )
    t_scipy = time.perf_counter() - t0
    U_scipy = sol_scipy.y[:, -1]

    # --- Reference solution (high-accuracy Radau) ---
    sol_ref = solve_ivp(
        fun=lambda t, U: schnakenberg_scipy(t, U, L, N),
        t_span=[0, t_final],
        y0=prob_sparse.initial_state,
        method="Radau",
        rtol=1e-12,
        atol=1e-12,
        jac=lambda t, U: schnakenberg_scipy_jac(t, U, L, N),
    )
    U_ref = sol_ref.y[:, -1]

    # --- Relative L2 errors (final state only) ---
    err_pyodys = np.linalg.norm(U_pyodys - U_ref) / np.linalg.norm(U_ref)
    err_scipy  = np.linalg.norm(U_scipy - U_ref) / np.linalg.norm(U_ref)

    results.append((N, 2 * N, t_pyodys_sparse, t_scipy, err_pyodys, err_scipy))
    print(f"N: {N}, 2N: {2 * N}, t_pyodys_sparse: {t_pyodys_sparse}, t_scipy: {t_scipy}, err_pyodys: {err_pyodys}, err_scipy: {err_scipy}")


# ---------------------------------------------------------------------------
# Print LaTeX table
# ---------------------------------------------------------------------------
print("\\begin{table}[ht!]")
print("\\centering")
print("\\begin{tabular}{c|c|c|c|c|c}")
print("\\hline")
print("$N$ & DoF ($2N$) & PyOdys time & SciPy time & Rel. L2 err (PyOdys) & Rel. L2 err (SciPy) \\\\")
print("\\hline")
for N, dof, t_pyodys, t_scipy, err_p, err_s in results:
    print(f"{N} & {dof} & {t_pyodys:.3f} & {t_scipy:.3f} & {err_p:.2e} & {err_s:.2e} \\\\")
    print("\\hline")
print("\\end{tabular}")
print("\\caption{CPU runtimes (in seconds) and relative $L^2$ errors for the 1D Schnakenberg reaction--diffusion system. PyOdys uses an explicit Runge--Kutta solver with sparse Jacobian handling; SciPy uses the implicit Radau method. Errors are measured at $t={t_final}$ against a Radau reference solution with $10^{-12}$ tolerances.}")
print("\\label{tab:schnakenberg1d_accuracy}")
print("\\end{table}")
