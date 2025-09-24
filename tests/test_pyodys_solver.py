import numpy as np
import pytest
from pyodys import PyodysSolver, RKScheme, ODEProblem
import pyodys.utils.pyodys_utils as utils


# --- Simple test ODE ---
class ExponentialDecay(ODEProblem):
    """u'(t) = -u, solution u(t) = exp(-t)."""
    def __init__(self, u0=1.0, t_init=0.0, t_final=1.0):
        super().__init__(t_init, t_final, [u0], mass_matrix_is_identity=True)

    def evaluate_at(self, t, u):
        return -u

    def jacobian_at(self, t, u):
        return np.array([[-1.0]])


@pytest.mark.parametrize("method_name", RKScheme.available_schemes())
def test_pyodys_solver_runs_and_matches_exact_solution(method_name):
    system = ExponentialDecay()
    solver = PyodysSolver(method=method_name, fixed_step=0.01)
    t, sol = solver.solve(system)
    exact = np.exp(-system.t_final)
    assert np.isclose(sol[-1][0], exact, rtol=1e-2)


@pytest.mark.parametrize("method_name", RKScheme.available_schemes())
def test_pyodys_solver_adaptive_runs(method_name):
    tableau = RKScheme.from_name(method_name)
    system = ExponentialDecay()
    solver = PyodysSolver(
        method=tableau,
        adaptive=True,
        first_step=0.01,
        min_step=1e-6,
        max_step=0.1,
        atol=1e-6,
        rtol=1e-6
    )
    t, sol = solver.solve(system)
    assert sol.shape[0] == len(t)
    assert np.all(np.diff(sol.flatten()) <= 0)


# --- Nonlinear problem to trigger Newton failure ---
class NonlinearProblem(ODEProblem):
    def __init__(self):
        super().__init__(0.0, 1.0, np.array([1.0]), mass_matrix_is_identity=True)

    def evaluate_at(self, t, u):
        return np.array([np.sin(u[0]) + 10.0])

    def jacobian_at(self, t, u):
        return np.array([[np.cos(u[0])]])


# @pytest.mark.parametrize("method_name", [m for m in RKScheme.available_schemes()
#                                           if RKScheme.from_name(m).is_implicit])
# def test_pyodys_solver_newton_failure_flag(method_name):
#     tableau = RKScheme.from_name(method_name)
#     solver = PyodysSolver(
#         method=tableau,
#         fixed_step=0.1,
#         adaptive=False,
#         newton_nmax=1,  # ensure Newton fails
#         atol=1e-10,
#         rtol=1e-10,
#         max_jacobian_refresh=0
#     )
#     try:
#         system = NonlinearProblem()
#         t, sol = solver.solve(system)
#     except utils.PyodysError:
#         pass
#     assert solver.newton_failed is True


def test_invalid_method_type_raises():
    with pytest.raises(TypeError):
        PyodysSolver(method=42)

def test_unknown_method_name_raises():
    with pytest.raises(ValueError):
        PyodysSolver(method="not_a_tableau")
