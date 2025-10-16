import numpy as np
import pytest

from pyodys import RKScheme, ODEProblem
from pyodys.solvers.RKSolver import RKSolver
import pyodys.utils.pyodys_utils as utils 

# To execute the tests, run python -m pytest -v, from the working directory edo/

class ExponentialDecay(ODEProblem):
    """
    Simple test system: u'(t) = -u, solution u(t) = exp(-t).
    """
    def __init__(self, u0=1.0, t_init=0.0, t_final=1.0):
        super().__init__(t_init, t_final, [u0])

    def evaluate_at(self, t, u):
        return -u

    def jacobian_at(self, t, u):
        return np.array([[-1.0]])


@pytest.mark.parametrize("method", RKScheme.available_schemes())
def test_solver_runs_and_matches_exact_solution(method):
    system = ExponentialDecay()
    tableau = RKScheme.from_name(method)
    solver = RKSolver(
        method=tableau, 
        fixed_step=0.01,
        adaptive=False
    )

    temps, solutions = solver.solve(system)
    exact = np.exp(-system.t_final)

    assert np.isclose(solutions[-1][0], exact, rtol=1e-2), \
        f"{method} failed: got {solutions[-1][0]}, expected {exact}"


@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
                                          #if RKScheme.from_name(m).with_prediction])
def test_solver_adaptive_step_runs(method_name):
    """Test adaptive stepping for schemes that support it."""
    tableau = RKScheme.from_name(method_name)
    solver = RKSolver(
        tableau,
        first_step=1.0e-1,
        adaptive=True,
        min_step=1e-6,
        max_step=0.5,
        atol=1e-6,
        rtol=1e-6
    )
    system = ExponentialDecay()

    temps, solutions = solver.solve(system)

    # Check shapes
    assert solutions.shape[0] == len(temps)
    assert solutions.shape[1] == system.initial_state.size

    # Check solution is monotonic decay
    assert np.all(np.diff(solutions.flatten()) <= 0)

# Define a stiff problem
class StiffProblem(ODEProblem):
    def __init__(self, t_init, t_final, initial_state):
        super().__init__(t_init, t_final, initial_state)
        
    def evaluate_at(self, t, u):
        x, y = u
        dxdt = -100.0*x + 99.0*y
        dydt = -y
        return np.array([dxdt, dydt])
    
    def jacobian_at(self, t, u):
        x, y = u
        jacobian = np.array([
            [-100.0, 99.0],
            [ 0.0, -1.0]
        ])
        return jacobian

def exact_solution(t):
    return np.array([2.0*np.exp(-t) - np.exp(-100.0 * t), 2.0 * np.exp(-t)])

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
def test_step_size_adjustment_time_limits(method_name):
    """Test that step size is clipped to min/max limits."""
    tableau = RKScheme.from_name(method_name)
    solver = RKSolver(
        method=tableau,
        first_step=1e-4, 
        adaptive=True,
        min_step=1e-8, 
        max_step=1.0,
        atol=1e-8,
        rtol=1e-8, 
        progress_interval_in_time=1.0, 
        max_jacobian_refresh=1
    )
    

    system = StiffProblem(t_init=0.0, t_final=1.0, initial_state=[1.0,2.0])
    temps, solutions = solver.solve(system)

    steps = np.diff(temps)
    assert np.all(steps >= 1e-8)
    assert np.all(steps <= 1.0)

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
def test_solver_adaptive_step_runs_and_matches_exact_solution(method_name):
    """Test that step size is clipped to min/max limits."""
    tableau = RKScheme.from_name(method_name)
    solver = RKSolver(
        method=tableau,
        first_step=1e-4,
        adaptive=True,
        min_step=1e-8, 
        max_step=1.0,
        atol=1e-8, 
        rtol=1e-8, 
        progress_interval_in_time=1.0, 
        max_jacobian_refresh=1
    )
    

    system = StiffProblem(t_init=0.0, t_final=1.0, initial_state=[1.0,2.0])
    temps, solutions = solver.solve(system)

    for i, t in enumerate(temps):
            numerical_solution = solutions[i]
            exact = exact_solution(t)
            assert np.allclose(numerical_solution, exact, rtol=1e-4, atol=1e-8)

def test_invalid_tableau_name_raises():
    with pytest.raises(ValueError):
        RKSolver(method="not_a_tableau")

def test_missing_fixed_step_raises():
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    with pytest.raises(ValueError):
        RKSolver(method=bt, adaptive = False, fixed_step=None)

def test_min_step_too_large_raises():
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    with pytest.raises(ValueError):
        tableau = RKScheme.from_name("erk4")
        min_step = 1.2
        solver = RKSolver(
            method=tableau,
            adaptive=True,
            min_step=min_step
        )

        system = StiffProblem(t_init=0.0, t_final=min_step/2, initial_state=[1.0,2.0])
        temps, solutions = solver.solve(system)

def test_export_creates_csv(tmp_path):
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    prefix = str(tmp_path / "results/out")
    solver = RKSolver(
        method=bt,
        adaptive = False,
        fixed_step=0.1,
        export_prefix=prefix,
        export_interval=1
    )
    times = np.array([0.0, 0.1, 0.2])
    sol = np.array([[1.0], [0.9], [0.81]])
    solver._export(times, sol)
    file = f"{prefix}_00001.csv"
    assert tmp_path.joinpath("results/out_00001.csv").exists()
    with open(file) as f:
        header = f.readline().strip().split(",")
    assert header[0] == "t"

class NonlinearProblem(ODEProblem):
    """Pathological nonlinear system that makes Newton iterations struggle."""
    def __init__(self):
        super().__init__(0.0, 1.0, np.array([1.0]))

    def evaluate_at(self, t, u):
        return np.array([np.sin(u[0]) + 10.0])

    def jacobian_at(self, t, u):
        return np.array([[np.cos(u[0])]])

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()
                                          if RKScheme.from_name(m).is_implicit])
def test_newton_failure_flag_triggered(method_name):
    """Force Newton failure by limiting max iterations to 1."""
    tableau = RKScheme.from_name(method_name)
    
    solver = RKSolver(
        method=tableau,
        fixed_step=0.1,
        adaptive=False,
        newton_nmax=1,  # ensures Newton fails
        atol=1e-10,
        rtol=1e-10,
        max_jacobian_refresh=0
    )
    try:
        system = NonlinearProblem()
        temps, solutions = solver.solve(system)
    except utils.PyodysError:
        pass

    assert solver.newton_failed is True

class ConstantMassMatrixProblem(ODEProblem):
    """Test system with a constant, non-identity mass matrix: M u' = -u"""
    def __init__(self):
        super().__init__(t_init=0.0, t_final=1.0, initial_state=np.array([1.0, 2.0]),
                          mass_matrix_is_constant=True)
        self._M = np.array([[2.0, 0.0], [0.0, 1]])

    def evaluate_at(self, t, u):
        return -u  # f = -u, note: solver multiplies by M^{-1}

    def _compute_mass_matrix(self, t, u):
        return self._M

    def jacobian_at(self, t, u):
        return -np.eye(2)  # df/du
    
    def exact_sol(self, t):
        #x_exact = C_x/(1+t), y_exact = C_y(2-t)
        # x(0) = 1 ==> C_x = 1 et y(0) = 2 ==> C_y = 1
        return np.array([np.exp(-0.5 * t), 2*np.exp(-t)], dtype=float)

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
def test_solver_constant_mass_matrix(method_name):
    tableau = RKScheme.from_name(method_name)
    system = ConstantMassMatrixProblem()
    solver = RKSolver(method=tableau, fixed_step=0.01, adaptive=False)
    times, solutions = solver.solve(system)
    # Since exact solution satisfies M u' = -u, multiply by M^{-1} to get u' = -M^{-1} u
    exact = np.array([system.exact_sol(t) for t in times]) 
    assert solutions.shape[0] == len(times)
    assert solutions.shape[1] == 2
    # Rough check: values decrease
    assert np.allclose(solutions, exact, 1e-2)

@pytest.mark.parametrize("method_name", RKScheme.available_schemes())
def test_constant_mass_matrix_adaptive(method_name):
    """Check solver with a constant non-identity mass matrix using adaptive stepping."""
    tableau = RKScheme.from_name(method_name)
    system = ConstantMassMatrixProblem()
    solver = RKSolver(
        method=tableau,
        first_step=0.01,
        adaptive=True,
        min_step=1e-6,
        max_step=0.1,
        atol=1e-8,
        rtol=1e-6
    )
    times, solutions = solver.solve(system)
    exact = np.array([system.exact_sol(t) for t in times]) 
    assert np.allclose(solutions, exact, rtol=1e-3), \
        f"Adaptive step failed for {method_name}, got {solutions[-1][0]}, expected {exact}"

class TimeVaryingMassProblem(ODEProblem):
    """System with time-dependent mass matrix: M(t) u' = -u"""
    def __init__(self):
        super().__init__(0.0, 1.0, np.array([1.0, 2.0]),
                          mass_matrix_is_constant=False)

    def evaluate_at(self, t, u):
        return -u

    def _compute_mass_matrix(self, t, u):
        return np.array([[1.0 + t, 0.0], [0.0, 2.0 - t]])

    def jacobian_at(self, t, u):
        return -np.eye(2)
    
    def exact_sol(self, t):
        #x_exact = C_x/(1+t), y_exact = C_y(2-t)
        # x(0) = 1 ==> C_x = 1 et y(0) = 2 ==> C_y = 1
        return np.array([1.0/(1+t), 2-t], dtype=float)

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
def test_solver_time_varying_mass(method_name):
    tableau = RKScheme.from_name(method_name)
    system = TimeVaryingMassProblem()
    solver = RKSolver(method=tableau, adaptive = False, fixed_step=0.01)
    times, solutions = solver.solve(system)
    assert solutions.shape[0] == len(times)
    assert solutions.shape[1] == 2
    exact = np.array([system.exact_sol(t) for t in times])
    assert np.allclose(solutions, exact, rtol=1e-2, atol=1e-5)

@pytest.mark.parametrize("method_name", RKScheme.available_schemes())
def test_time_varying_mass_matrix_adaptive(method_name):
    """Check solver with a time-varying mass matrix using adaptive stepping."""
    tableau = RKScheme.from_name(method_name)
    system = TimeVaryingMassProblem()
    solver = RKSolver(
        method=tableau,
        first_step=0.01,
        adaptive=True,
        min_step=1e-6,
        max_step=0.1,
        atol=1e-8,
        rtol=1e-6
    )
    times, solutions = solver.solve(system)
    exact = np.array([system.exact_sol(t) for t in times], dtype=float)
    assert np.allclose(solutions, exact, rtol=1e-3, atol=1e-6)

class NonDiagonalMassProblem(ODEProblem):
    """
    Linear system with non-diagonal mass matrix:
        M u' = -u,  M = [[2, 1], [1, 2]]
    Exact solution: u(t) = exp(-t/3) * [1, 1]
    """
    def __init__(self):
        super().__init__(t_init=0.0, t_final=1.0, initial_state=np.array([1.0,1.0]), mass_matrix_is_constant=True,  jacobian_is_constant=True)
        self.M = np.array([[2.0, 1.0], [1.0, 2.0]])

    def evaluate_at(self, t, u):
        return -u

    def jacobian_at(self, t, u):
        return -np.eye(2)
    
    def _compute_mass_matrix(self, t, state):
        return self.M
    
    def exact_sol(self, t):
        return np.array([np.exp(-t/3), np.exp(-t/3)], dtype=float)

@pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()])
def test_non_diagonal_mass_adaptive(method_name):
    tableau = RKScheme.from_name(method_name)
    solver = RKSolver(
        method=tableau,
        first_step=0.1,
        adaptive=True,
        min_step=1e-6,
        max_step=0.5,
        atol=1e-8,
        rtol=1e-8
    )

    system = NonDiagonalMassProblem()
    times, solutions = solver.solve(system)

    exact = np.array([system.exact_sol(t) for t in times], dtype=float)
    assert np.allclose(solutions, exact, rtol=1e-4, atol=1e-8)

def test_user_defined_linear_solver_called():
    """Check that a user-defined linear solver is called for DIRK stages."""
    called = {"flag": False}
    
    def my_solver(A, rhs):
        called["flag"] = True
        # simple solver: assume A is invertible and small
        return np.linalg.solve(A, rhs)
    
    tableau = RKScheme.from_name("SDIRK2")  # choose a DIRK/SDIRK scheme
    solver = RKSolver(method=tableau, adaptive = False, fixed_step=0.01, linear_solver=my_solver)
    system = ExponentialDecay()
    
    solver.solve(system)
    assert called["flag"] is True


class DummyProblem(ODEProblem):
    """Simple linear system for testing."""
    def __init__(self):
        super().__init__(0.0, 1.0, np.array([1.0]))
    def evaluate_at(self, t, u):
        return -u
    def jacobian_at(self, t, u):
        return np.array([[-1.0]])

def test_random_wrong_linear_solver_triggers_pyodys_error():
    """Randomly wrong linear solver should make Newton fail and raise PyodysError."""

    def random_solver(A, rhs):
        # return a random vector of the same shape
        return np.random.rand(*rhs.shape)

    tableau = RKScheme.from_name("SDIRK2")
    solver = RKSolver(
        method=tableau,
        adaptive = False,
        fixed_step=0.1,
        linear_solver=random_solver,
        newton_nmax=5,  # low iteration to speed up failure
        atol=1e-12,
        rtol=1e-12,
    )

    system = DummyProblem()
    with pytest.raises(utils.PyodysError):
        solver.solve(system)

def test_user_defined_solver_with_constant_mass():
    called = {"flag": False}
    def custom_solver(A, rhs):
        called["flag"] = True
        return np.linalg.solve(A, rhs)
    
    tableau = RKScheme.from_name("SDIRK2")
    system = ConstantMassMatrixProblem()
    solver = RKSolver(method=tableau, adaptive = False, fixed_step=0.01, linear_solver=custom_solver)
    
    times, solutions = solver.solve(system)
    assert called["flag"] is True
    # Check rough correctness of solution
    exact = np.array([system.exact_sol(t) for t in times])
    assert np.allclose(solutions, exact, rtol=1e-2)

def test_user_defined_solver_time_varying_mass():
    called = {"flag": False}
    def custom_solver(A, rhs):
        called["flag"] = True
        return np.linalg.solve(A, rhs)
    
    tableau = RKScheme.from_name("SDIRK2")
    system = TimeVaryingMassProblem()
    solver = RKSolver(method=tableau, adaptive = False, fixed_step=0.01, linear_solver=custom_solver)
    
    times, solutions = solver.solve(system)
    assert called["flag"] is True
    exact = np.array([system.exact_sol(t) for t in times])
    assert np.allclose(solutions, exact, rtol=1e-2)