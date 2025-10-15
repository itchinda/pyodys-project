[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![Build](https://github.com/itchinda/pyodys-project/actions/workflows/ci.yml/badge.svg)]()



# PyOdys – Numerical ODE Solvers for Large and Stiff Systems

PyOdys is a robust and flexible Python package for solving **ordinary differential equations (ODEs)** of the form: 

$$ 
M(t, y)\frac{dy}{dt} = F(t, y).
$$.

where `M(t, y)` is the **mass matrix**. This can be the identity (classical ODEs) or a general, possibly **time-dependent matrix**.  

It supports both **Runge–Kutta schemes** (explicit, DIRK) and **BDF multistep methods**, with adaptive time-stepping and strong support for **sparse Jacobians**—making it well-suited for large-scale and stiff problems. It also includes a **numerical Jacobian** for convenience.

---

## Features

- **Unified Solver Interface**:  
  The `PyodysSolver` class provides a single entry point. You just specify the method name (e.g. `"erk4"`, `"esdirk64"`, `"bdf4"`) and PyOdys automatically selects the correct solver backend: RK or BDF (**planned**).

- **Wide Range of Methods**:
  - **Explicit Runge–Kutta**: classic schemes like RK4 (`erk4`) and Dormand–Prince (`dopri54`).
  - **Implicit Runge-Kutta**: DIRK, SDIRK and ESDIRK methods for stiff problems.
  - **BDF methods (planned)**: multistep implicit solvers for highly stiff systems. A `BDFSolver` class is included in the design, but support for BDF methods is still **under development** and not yet available in this release.

- **Mass Matrix Support**:  
  PyOdys now fully supports **general mass matrices**:
  - Constant or **time-dependent**.
  - Non-diagonal, block-diagonal, or full sparse matrices.
  - User can provide `M(t, y)` and the solver will correctly handle the linear solve at each step.
  - Works seamlessly with **sparse Jacobians** for large systems.
  
- **PyOdys is designed to be highly extensible**:
  - Users may plug in custom Runge–Kutta schemes through the `RKScheme` class.
  - Support for custom BDF schemes will be available through the `BDFScheme` class once the BDF solver is finalized.

- **Adaptive Time-Stepping**: 
  Automatic control of time step size based on local error estimates. Balances accuracy and efficiency, crucial for multiscale dynamics.

- **Implicit Method Support**:  
  Nonlinear systems are solved with **Newton iterations**. Linear solves exploit sparse Jacobians (`scipy.sparse.linalg`).

- **Flexible Problem Definition:**:  
  Define any ODE system by inheriting from the `ODEProblem` abstract class. A fallback **numerical Jacobian** (central finite differences) is provided automatically.
  - Default: numerical Jacobian(central finite differences) is provided automatically.
  - Optional: user-supplied analytical/sparse Jacobian for efficiency.

- **Example Systems Included**:
  - **Lorenz System**: Demonstrates handling of chaotic dynamics and generates the famous butterfly attractor.  
  - **Simple Linear System**: With a known analytical solution, perfect for accuracy testing.
  - **Robertson**: A classic stiff problem that showcases the power of implicit solvers.
  - **1D parabolic problem**: Demonstrates solving a 1D parabolic PDE with PyOdys using a sparse Jacobian for efficient large-scale computation, and visualizes the animated solution against the exact result.
---

## Getting Started

### Prerequisites

You will need **Python** (version $\geq$ 3.8) and the following packages:

- `numpy`  
- `scipy`  
- `matplotlib` (for visualization)

### Installation

Clone the repository and install the package in "editable" mode:


```bash
git clone https://github.com/itchinda/pyodys-project.git
cd pyodys-project
pip install -e .

```

The `-e` flag allows you to run the package from any directory while still being able to edit the source code.

## Usage

### Listing Available Schemes

You can list all the available Runge-Kutta schemes directly from the command line:

```bash
python -m pyodys --list-schemes
```

### Running a Quick Example

To solve the Lorenz System with a simple command, you can use one of the provided examples The script will automatically handle the initial conditions and visualization.

```bash
python examples/lorenz_system.py --method dopri5 --final-time 50.0
```

You can customize the simulation by changing parameters like the method (`--method`), adaptive stepping (`--adaptive`), final time (`--final-time`), initial step (`--first-step`), minimal step (`--min-step`), maximal step (`--max-step`), adaptive (`--atol`) and relative (`--rtol`) tolerances.

## Code Example: Coupled Linear System

This example solves the coupled system:

$$ 
x'(t) = -x(t) + y(t),
$$

$$ 
y'(t) = -y(t),
$$

with 

$$ 
x(0) = 1, y(0) = 1, 
$$

using **RK4** solver, and plot the solution:

$$
x(t) = e^{-t}  (1 + t),  
$$

$$
y(t) = e^{-t}
$$

---

```python
import numpy as np
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# Define coupled system
class CoupledLinearSystem(ODEProblem):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)
    def evaluate_at(self, t, u):
        x, y = u
        return np.array([-x + y, -y])

# Analytical solution
def analytical_solution(t, u0):
    tau = t - 0.0
    x0, y0 = u0
    x = np.exp(-tau) * (x0 + y0 * tau)
    y = y0 * np.exp(-tau)
    return np.array([x, y])

if __name__ == "__main__":
    t_init, t_final = 0.0, 10.0
    u_init = [1.0, 1.0]
    problem = CoupledLinearSystem(t_init, t_final, u_init)

    solver = PyodysSolver(
      method = 'sdirk43',
      first_step = 1e-2,
      adaptive = True,
      min_step = 1e-6,
      max_step = 1.0,
      atol = 1e-10,
      rtol = 1e-8
    )

    times, U = solver.solve(problem)

    # Analytical
    U_exact = np.array([analytical_solution(t, u_init) for t in times])
    error = np.linalg.norm(U - U_exact, axis=1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(times, U[:, 0], "b-", label="x(t) Numerical")
    ax1.plot(times, U[:, 1], "r-", label="y(t) Numerical")
    ax1.plot(times, U_exact[:, 0], "k--", label="x(t) Analytical")
    ax1.plot(times, U_exact[:, 1], "r-.", label="y(t) Analytical")
    ax1.set_title("Coupled Linear System")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, error, "b-", label="L2 Error")
    ax2.set_yscale("log")
    ax2.set_title("Error vs Analytical Solution")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

```
![Quick Example Output Figures](src/pyodys/utils/figures/quick_example.png)

## 1D Heat equation discretization using finite element

```python
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
        method="sdirk54",
        adaptive=True,
        first_step=1e-3,
        atol=1e-8,
        rtol=1e-8,
        min_step=1e-8,
        max_step=1,
        verbose=True,
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
```
![Quick Example Output Figures](src/pyodys/utils/figures/1d_heat_eq_finite_elem.png)

###  Defining a Custom Runge–Kutta Scheme

PyOdys allows users to define their **own Runge–Kutta method** via the `RKScheme` class. This is useful if you want to experiment with new schemes, test variants from the literature, or reproduce methods from papers.

#### Example:

```python
import numpy as np
from pyodys import RKScheme

# Define Butcher tableau
A = np.array([
    [0.0, 0.0],
    [0.5, 0.0]
])
B = np.array([0.0, 1.0])   # weights
C = np.sum(A, axis=1)      # nodes

# Create scheme
midpoint = RKScheme(A, B, C, order=2)

print(midpoint)
print(midpoint.info())
```
Output:
```vbnet
Runge-Kutta method of order 2

  0 |   0   0
0.5 | 0.5   0
----+---------
        0   1

Type: Explicit RK
Stages: 2
Order: 2
Embedded: No
```
You can then use it in the solver configuration as follows:
```python
solver = PyodysSolver(
      method = midpoint,
      first_step = 1e-2,
      adaptive = True,
      min_step = 1e-6,
      max_step = 1.0,
      atol = 1e-10,
      rtol = 1e-8
    )

``` 


