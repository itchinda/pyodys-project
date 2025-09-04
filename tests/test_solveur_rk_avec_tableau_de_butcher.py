import numpy as np
import pytest

from pyodys import SolveurRKAvecTableauDeButcher
from pyodys import TableauDeButcher
from pyodys import EDOs

# To execute the tests, run python -m pytest -v, from the working directory edo/

class ExponentialDecay(EDOs):
    """
    Simple test system: u'(t) = -u, solution u(t) = exp(-t).
    """
    def __init__(self, u0=1.0, t_init=0.0, t_final=1.0):
        self.initial_state = np.array([u0])
        self.t_init = t_init
        self.t_final = t_final

    def evalue(self, t, u):
        return -u

    def jacobien(self, t, u):
        return np.array([[-1.0]])


@pytest.mark.parametrize("method", TableauDeButcher.SCHEMAS_DISPONIBLES)
def test_solver_runs_and_matches_exact_solution(method):
    system = ExponentialDecay()
    tableau = TableauDeButcher.par_nom(method)
    solver = SolveurRKAvecTableauDeButcher(tableau_de_butcher=tableau, 
                                           initial_step_size=0.01)

    temps, solutions = solver.resoud(system)
    exact = np.exp(-system.t_final)

    assert np.isclose(solutions[-1][0], exact, rtol=1e-2), \
        f"{method} failed: got {solutions[-1][0]}, expected {exact}"


@pytest.mark.parametrize("method_name", [m for m in TableauDeButcher.SCHEMAS_DISPONIBLES
                                          if TableauDeButcher.par_nom(m).est_avec_prediction])
def test_solver_adaptive_step_runs(method_name):
    """Test adaptive stepping for schemes that support it."""
    tableau = TableauDeButcher.par_nom(method_name)
    solver = SolveurRKAvecTableauDeButcher(tableau,
                                           initial_step_size=1.0e-5,
                                           adaptive_time_stepping=True,
                                           min_step_size=1e-6,
                                           max_step_size=0.5,
                                           target_relative_error=1e-4)
    system = ExponentialDecay()

    temps, solutions = solver.resoud(system)

    # Check shapes
    assert solutions.shape[0] == len(temps)
    assert solutions.shape[1] == system.initial_state.size

    # Check solution is monotonic decay
    assert np.all(np.diff(solutions.flatten()) <= 0)

def test_invalid_tableau_type_raises():
    """Check that passing a wrong type to the solver raises TypeError."""
    with pytest.raises(TypeError):
        SolveurRKAvecTableauDeButcher(tableau_de_butcher="not_a_tableau")

# Define a stiff problem
class StiffProblem(EDOs):
    def __init__(self, t_init, t_final, initial_state):
        super().__init__(t_init, t_final, initial_state)
        
    def evalue(self, t, u):
        x, y = u
        dxdt = -100.0*x + 99.0*y
        dydt = -y
        return np.array([dxdt, dydt])
    
    def jacobien(self, t, u):
        x, y = u
        Jacobien = np.array([
            [-100.0, 99.0],
            [ 0.0, -1.0]
        ])
        return Jacobien

def exact_solution(t):
    return np.array([2.0*np.exp(-t) - np.exp(-100.0 * t), 2.0 * np.exp(-t)])

@pytest.mark.parametrize("method_name", [m for m in TableauDeButcher.SCHEMAS_DISPONIBLES
                                          if TableauDeButcher.par_nom(m).est_avec_prediction])
def test_step_size_adjustment_time_limits(method_name):
    """Test that step size is clipped to min/max limits."""
    tableau = TableauDeButcher.par_nom(method_name)
    solver = SolveurRKAvecTableauDeButcher(tableau_de_butcher=tableau,
                                           initial_step_size=1e-4, 
                                           adaptive_time_stepping=True,
                                           min_step_size=1e-8, 
                                           max_step_size=1.0,
                                           target_relative_error=1e-8, 
                                           progress_interval_in_time=1.0, 
                                           max_jacobian_refresh=1)
    

    system = StiffProblem(t_init=0.0, t_final=1.0, initial_state=[1.0,2.0])
    temps, solutions = solver.resoud(system)

    steps = np.diff(temps)
    assert np.all(steps >= 1e-8)
    assert np.all(steps <= 1.0)

@pytest.mark.parametrize("method_name", [m for m in TableauDeButcher.SCHEMAS_DISPONIBLES
                                          if TableauDeButcher.par_nom(m).est_avec_prediction])
def test_solver_adaptive_step_runs_and_matches_exact_solution(method_name):
    """Test that step size is clipped to min/max limits."""
    tableau = TableauDeButcher.par_nom(method_name)
    solver = SolveurRKAvecTableauDeButcher(tableau,
                                           initial_step_size=1e-4,
                                           adaptive_time_stepping=True,
                                           min_step_size=1e-8, 
                                           max_step_size=1.0,
                                           target_relative_error=1e-8, 
                                           progress_interval_in_time=1.0, 
                                           max_jacobian_refresh=1)
    

    system = StiffProblem(t_init=0.0, t_final=1.0, initial_state=[1.0,2.0])
    temps, solutions = solver.resoud(system)

    for i, t in enumerate(temps):
            numerical_solution = solutions[i]
            exact = exact_solution(t)
            assert np.allclose(numerical_solution, exact, rtol=1e-5, atol=1e-8)

def test_invalid_tableau_raises():
    with pytest.raises(TypeError):
        SolveurRKAvecTableauDeButcher(tableau_de_butcher="not_a_tableau")
