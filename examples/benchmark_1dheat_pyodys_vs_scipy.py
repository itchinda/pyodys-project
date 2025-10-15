#!/usr/bin/env python3
"""
Benchmark with exact solution: 1D heat equation u_t = D u_xx with u(x,0)=sin(pi x).
Compare PyOdys PyodysSolver vs SciPy solve_ivp (Radau).
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from pyodys import ODEProblem, PyodysSolver   # adjust import if class name differs

# problem parameter
D = 1.0e-3   # diffusion coefficient
pi = np.pi

def laplacian_1d(N, h):
    """N interior points, Dirichlet BCs (matrix is N x N)."""
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N-1)
    return diags([off, main, off], offsets=[-1,0,1], format='csr') / (h*h)

def exact_solution_on_grid(x, t):
    """Exact solution u(x,t) = sin(pi x) exp(-D pi^2 t)"""
    return np.sin(pi * x) * np.exp(-D * pi**2 * t)

class Heat1D(ODEProblem):
    """Semidiscrete 1D heat equation with Dirichlet BCs and sin(pi x) initial condition.

    - If force_dense=True, jacobian_at returns a dense ndarray.
    - Otherwise returns a scipy.sparse.csr_matrix.
    """
    def __init__(self, N, t0, tf, force_dense=False):
        self.N = N
        self.h = 1.0 / (N + 1)
        self.x = np.linspace(self.h, 1.0 - self.h, N)
        self.force_dense = force_dense
        # initial: sin(pi x) on interior points
        u0 = np.sin(pi * self.x)
        super().__init__(t0, tf, u0, jacobian_is_constant=True)
        # Precompute Laplacian operator (sparse)
        self.L = laplacian_1d(N, self.h)

        self.nb_jacobian_call = 0
        self.nb_evaluate_at = 0

    def evaluate_at(self, t, u):
        # u is length N
        self.nb_evaluate_at += 1
        return D * self.L.dot(u)

    def jacobian_at(self, t, u):
        # Jacobian is simply D * L (independent of u or t)
        self.nb_jacobian_call += 1
        if self.force_dense:
            return (D * self.L).toarray()
        return (D * self.L).tocsr()

# --- benchmark runner -------------------------------------------------------
def run_benchmark(N_list, t_final=1.0):
    # solver options (PyOdys)
    pyodys_opts = dict(
        method = "esdirk64",   # fill if RKSolver expects a tableau; adjust accordingly
        first_step = 1e-6,
        adaptive = True,
        min_step = 1e-8,
        max_step = 5e-1,
        atol = 1e-10,
        rtol = 1e-10,
        auto_check_sparsity = True,
        verbose = False
    )

    # SciPy options
    scipy_opts = dict(method='Radau', 
                      rtol=1e-10, 
                      atol=1e-10, 
                      first_step=1e-6, 
                      max_step=5e-1)

    results = []

    for N in N_list:
        print(f"\n--- N = {N} (dof = {N}) ---")
        prob_sparse_pyodys = Heat1D(N, 0.0, t_final, force_dense=False)
        prob_sparse_scipy = Heat1D(N, 0.0, t_final, force_dense=False)
        #prob_dense  = Heat1D(N, 0.0, t_final, force_dense=True)

        # --- PyOdys sparse run ---
        pyodys_solver = PyodysSolver(**pyodys_opts)

        t0 = time.perf_counter()
        times_p, sol_p = pyodys_solver.solve(prob_sparse_pyodys)   # sol_p shape (nt, N)
        time_pyodys = time.perf_counter() - t0
        
        print(f"Nb Jacobian call: {prob_sparse_pyodys.nb_jacobian_call}")
        print(f"Nb evaluate_at call: {prob_sparse_pyodys.nb_evaluate_at}")

        # compute error at final time (last solution row)
        u_num = sol_p[-1, :]
        u_ex = exact_solution_on_grid(prob_sparse_pyodys.x, t_final)
        rel_L2_error_py_sparse = np.linalg.norm(u_num - u_ex) / np.linalg.norm(u_ex)

        # --- SciPy sparse run (Radau with sparse jac) ---
        u0 = prob_sparse_scipy.initial_state.copy()
        t0 = time.perf_counter()
        sol_s = solve_ivp(fun=lambda t,y: prob_sparse_scipy.evaluate_at(t,y),
                          t_span=(0.0, t_final),
                          y0=u0,
                          jac=lambda t,y: prob_sparse_scipy.jacobian_at(t,y),
                          **scipy_opts)
        t_scipy = time.perf_counter() - t0
        u_scipy = sol_s.y[:, -1]
        rel_L2_error_scipy = np.linalg.norm(u_scipy - u_ex) / np.linalg.norm(u_ex)

        # record
        results.append({
            "N": N,
            "dof": N,
            "py_sparse_time": time_pyodys,
            "py_sparse_err": rel_L2_error_py_sparse,
            "scipy_time": t_scipy,
            "scipy_err": rel_L2_error_scipy
        })

        print(f"Py sparse: time={time_pyodys:.4f}s err={rel_L2_error_py_sparse:.2e}")
        print(f"SciPy Radau: time={t_scipy:.4f}s err={rel_L2_error_scipy:.2e}")

        print(f"Nb Jacobian call: {prob_sparse_scipy.nb_jacobian_call}")
        print(f"Nb evaluate_at call: {prob_sparse_scipy.nb_evaluate_at}")
    return results


if __name__ == "__main__":
    N_list = [50, 100, 200, 400, 800, 1600, 3200 , 10000, 100000, 1000000]
    r = run_benchmark(N_list, t_final=1.0)

    # print LaTeX table
    print("\n\\begin{tabular}{c|c|c|c|c}")
    print("N & PyOdys (sparse) &  SciPy (Radau) & rel. L2 error(PyOdys) & rel. L2 error(Scipy)  \\\\ \\hline")
    for row in r:
        print(f"{row['N']} & {row['py_sparse_time']:.3f} & {row['scipy_time']:.3f} & {row['py_sparse_err']:.2e} & {row['scipy_err']:.2e} \\\\")
    print("\\end{tabular}")

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.loglog(N_list, [d["py_sparse_time"] for d in r], 'b-', label = "PyOdys")
    ax1.loglog(N_list, [d["scipy_time"] for d in r], 'r-', label = "Scipy")
    ax1.set_title("CPU-Time: PyOdys vs Scipy")
    ax1.set_xlabel("N")
    ax1.set_ylabel("CPU-Time")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.loglog(N_list, [d["py_sparse_err"] for d in r], 'b-', label = "PyOdys")
    ax2.loglog(N_list, [d["scipy_err"] for d in r], 'r-', label = "Scipy")

    ax2.set_title("l2-error: PyOdys vs Scipy")
    ax2.set_xlabel("N")
    ax2.set_ylabel("l2-error")
    ax2.grid(True)
    ax2.legend()

    plt.show()

