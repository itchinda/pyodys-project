import numpy as np
import pytest
from pyodys import ODEProblem


# Dummy subclass for testing
class LinearSystem(ODEProblem):
    """Simple linear system: dx/dt = Ax."""

    def __init__(self, t_init, t_final, initial_state, A, **kwargs):
        super().__init__(t_init, t_final, initial_state, **kwargs)
        self.A = np.array(A, dtype=np.float64)

    def evalue(self, t, state):
        return self.A @ state


# ---------- Constructor Validation ----------

def test_valid_construction():
    sys = LinearSystem(0.0, 1.0, [1.0, 2.0], np.eye(2))
    assert sys.t_init == 0.0
    assert sys.t_final == 1.0
    assert np.allclose(sys.initial_state, [1.0, 2.0])


@pytest.mark.parametrize("t_init,t_final", [(1.0, 1.0), (2.0, 1.0)])
def test_invalid_time_order(t_init, t_final):
    with pytest.raises(ValueError, match="t_final must be strictly greater"):
        LinearSystem(t_init, t_final, [1.0], np.eye(1))


def test_invalid_initial_state_empty():
    with pytest.raises(ValueError, match="non-empty"):
        LinearSystem(0.0, 1.0, [], np.eye(1))


def test_invalid_initial_state_ndim():
    with pytest.raises(ValueError, match="1D array"):
        LinearSystem(0.0, 1.0, [[1.0, 2.0]], np.eye(2))


@pytest.mark.parametrize("value", ["abc", None, [1, 2]])
def test_invalid_t_init_type(value):
    with pytest.raises(ValueError, match="t_init must be a real numeric scalar"):
        LinearSystem(value, 1.0, [1.0], np.eye(1))


@pytest.mark.parametrize("value", ["abc", None, [1, 2]])
def test_invalid_t_final_type(value):
    with pytest.raises(ValueError, match="t_final must be a real numeric scalar"):
        LinearSystem(0.0, value, [1.0], np.eye(1))


# ---------- Jacobian Tests ----------

def test_jacobian_linear_system_identity():
    A = np.eye(2)
    sys = LinearSystem(0.0, 1.0, [1.0, 0.0], A)
    J = sys.jacobien(0.0, np.array([1.0, 0.0]))
    assert np.allclose(J, A, atol=1e-8)


def test_jacobian_linear_system_nontrivial():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A, delta=1e-6)
    J = sys.jacobien(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J, A, atol=1e-6)


# ---------- Abstract Class Enforcement ----------

def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        ODEProblem(0.0, 1.0, [1.0])  # should fail since abstract
