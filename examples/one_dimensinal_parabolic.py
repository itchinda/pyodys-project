import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable
import time

from pyodys import ODEProblem
from pyodys import ButcherTableau
from pyodys import RKSolverWithButcherTableau

# ---------------- Laplacian ----------------
def build_1d_laplacian(N: int, h: float, dtype=float) -> csr_matrix:
    main_diag = -2.0 * np.ones(N, dtype=dtype)
    off_diag = np.ones(N-1, dtype=dtype)
    L = sp.diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr",
        dtype=dtype
    )
    return L

# ---------------- Forcing ----------------
def forcing_vector(x_nodes: np.ndarray, t: float) -> np.ndarray:
    return np.zeros(len(x_nodes))

# ---------------- Parabolic system ----------------
class ParabolicProblem(ODEProblem):
    def __init__(self, N, t_init, t_final, u0_function: Callable[[np.ndarray], np.ndarray],
                 forcing_func: Callable[[np.ndarray, float], np.ndarray] = None):
        self.N = N
        self.h = 1.0 / (N + 1)
        self.x = np.linspace(self.h, 1.0 - self.h, N)  # interior nodes only
        self.L = build_1d_laplacian(N, self.h)
        self.A = (1.0 / self.h ** 2) * self.L
        self.forcing_func = forcing_func
        self.initial_state = np.asarray(u0_function(self.x), dtype=float).reshape(N,)
        super().__init__(t_init=t_init, t_final=t_final, initial_state=self.initial_state)

    def evalue(self, t: float, u: np.ndarray) -> np.ndarray:
        Au = self.A.dot(u)
        if self.forcing_func is None:
            return Au
        return Au + self.forcing_func(self.x, t)

    def jacobien(self, t: float, u: np.ndarray):
        return self.A  # already sparse

# ---------------- Exact solution ----------------
def u_exact(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

# ---------------- Initial condition ----------------
def u0_func(x):
    return np.sin(np.pi * x)

# ---------------- Parameters ----------------
Nx = 10000
t0, tf = 0.0, 1.0
L = 1.0

x_interior = np.linspace(1/(Nx+1), 1-1/(Nx+1), Nx)
parabolic_problem = ParabolicProblem(N=Nx, t_init=t0, t_final=tf,
                                     u0_function=u0_func,
                                     forcing_func=forcing_vector)

solver = RKSolverWithButcherTableau(
    butcher_tableau=ButcherTableau.from_name("sdirk_norsett_thomson_23"),
    initial_step_size=1e-5,
    adaptive_time_stepping=True,
    target_relative_error=1e-5,
    min_step_size=1e-8,
    max_step_size=1e-1,
    auto_sparse_jacobian=True
)

# ---------------- Solve ----------------
start = time.time()
times, numerical_solutions = solver.solve(parabolic_problem)
Elapsed = time.time() - start
print(f"Elapsed time = {Elapsed}")

analytical_solutions = np.array([u_exact(x_interior, t) for t in times])

# ---------------- Animation ----------------
fig, ax = plt.subplots()
line_num, = ax.plot([], [], 'r-', lw=2, label="Numerical")
line_exact, = ax.plot([], [], 'b--', lw=2, label="Exact")
ax.set_xlim(0, L)
ax.set_ylim(0.0, 1.2)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('1D Parabolic Problem: Numerical vs Exact')
ax.legend()
ax.grid(True)

def init():
    line_num.set_data([], [])
    line_exact.set_data([], [])
    return line_num, line_exact

# Subsample frames if too many
n_frames = len(times)
skip = max(1, n_frames // 200)

def update(frame):
    line_num.set_data(x_interior, numerical_solutions[frame])
    line_exact.set_data(x_interior, analytical_solutions[frame])
    ax.set_title(f"t = {times[frame]:.5f}")
    return line_num, line_exact

total_duration = 5.0
interval = total_duration / n_frames * 1000
anim = FuncAnimation(fig, update, frames=range(0, n_frames, skip),
                     init_func=init, blit=True, interval=interval*skip)

plt.show()
