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
        super().__init__(t_init, t_final, [u0], mass_matrix_is_identity=True)

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
        super().__init__(t_init, t_final, initial_state, mass_matrix_is_identity=True)
        
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

def test_missing_first_step_raises():
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    with pytest.raises(ValueError):
        RKSolver(method=bt, fixed_step=None)

def test_adaptive_missing_args_raise():
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    # missing min/max
    with pytest.raises(TypeError):
        RKSolver(
            method=bt, 
            first_step=0.1,
            adaptive=True, 
            min_step=None,
            max_step=0.1, 
            rtol=1e-3
        )

def test_export_creates_csv(tmp_path):
    bt = RKScheme.from_name(RKScheme.available_schemes()[0])
    prefix = str(tmp_path / "results/out")
    solver = RKSolver(
        method=bt,
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

# def test_detect_jacobian_sparsity_dense_and_sparse():
#     bt = next((RKScheme.from_name(name) for name in  RKScheme.available_schemes() if RKScheme.from_name(name).is_implicit), None)

#     solver = RKSolver(bt, fixed_step=0.1)
#     system = ExponentialDecay()
#     solver._detect_sparsity(system, 0.0, system.initial_state)
#     assert solver._jacobian_is_sparse is False

#     # force sparse Jacobian
#     import scipy.sparse as sp
#     class SparseSystem(ExponentialDecay):
#         def jacobian_at(self, t, u):
#             return sp.csr_matrix((1,1))
#     solver._detect_sparsity(SparseSystem(), 0.0, np.array([1.0]))
#     assert solver._jacobian_is_sparse is True

# def test_check_step_size_rejects_large_error():
#     bt = RKScheme.from_name(RKScheme.available_schemes()[0])
#     solver = RKSolver(bt, fixed_step=0.1)
#     U_approx = np.array([1.0])
#     U_pred = np.array([0.0])  # huge error
#     new_step, accepted = solver._check_step_size(U_approx, 
#                                                  U_pred,
#                                                  step_size=0.1,
#                                                  min_step=1e-6,
#                                                  max_step=1.0,
#                                                  current_time=0.0,
#                                                  t_final=1.0)
#     assert accepted == False
#     assert new_step >= 1e-6

class NonlinearProblem(ODEProblem):
    """Pathological nonlinear system that makes Newton iterations struggle."""
    def __init__(self):
        super().__init__(0.0, 1.0, np.array([1.0]), mass_matrix_is_identity=True)

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