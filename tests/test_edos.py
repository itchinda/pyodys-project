import numpy as np
import pytest
from abc import ABC

from pyodys import EDOs


# -------------------------------------------------------------------
# Dummy subclass for testing
# dx/dt = -x
class ExponentialDecay(EDOs):
    def evalue(self, t: float, state: np.ndarray) -> np.ndarray:
        return -state


# dx/dt = A x, with A = [[0, 1], [-2, -3]]
class LinearSystem(EDOs):
    A = np.array([[0.0, 1.0],
                  [-2.0, -3.0]])

    def evalue(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.A @ state


# -------------------------------------------------------------------
def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        EDOs(0.0, 1.0, [1.0])  # should fail: abstract method not implemented

@pytest.mark.parametrize("t_init, t_final", [
    ("a", 1.0),         # string
    (None, 1.0),        # None
    ([0], 1.0),         # list
    (0.0, "b"),         # string for t_final
    (0.0, None),        # None for t_final
])
def test_t_init_t_final_must_be_numeric(t_init, t_final):
    with pytest.raises(ValueError, match="numeric"):
        ExponentialDecay(t_init, t_final, [1.0])
        
def test_invalid_time_interval():
    with pytest.raises(ValueError):
        ExponentialDecay(1.0, 1.0, [1.0])  # t_final == t_init
    with pytest.raises(ValueError):
        ExponentialDecay(2.0, 1.0, [1.0])  # t_final < t_init


def test_empty_initial_state():
    with pytest.raises(ValueError):
        ExponentialDecay(0.0, 1.0, [])  # empty state not allowed


def test_attributes_are_set_correctly():
    ode = ExponentialDecay(0.0, 10.0, [1.0, 2.0])
    assert ode.t_init == 0.0
    assert ode.t_final == 10.0
    np.testing.assert_array_equal(ode.initial_state, np.array([1.0, 2.0]))


def test_evalue_in_subclass():
    ode = ExponentialDecay(0.0, 5.0, [1.0])
    val = ode.evalue(0.0, np.array([2.0]))
    np.testing.assert_array_equal(val, np.array([-2.0]))


def test_jacobian_scalar_case():
    ode = ExponentialDecay(0.0, 5.0, [1.0])
    state = np.array([1.0])
    J = ode.jacobien(0.0, state)
    # Analytical Jacobian = [-1]
    assert J.shape == (1, 1)
    np.testing.assert_allclose(J, np.array([[-1.0]]), atol=1e-6)


def test_jacobian_matrix_case():
    ode = LinearSystem(0.0, 5.0, [1.0, 0.0])
    state = np.array([1.0, -1.0])
    J = ode.jacobien(0.0, state)
    # Analytical Jacobian = A
    np.testing.assert_allclose(J, LinearSystem.A, atol=1e-6)


def test_multiple_dimensions_evalue_and_jacobian():
    ode = LinearSystem(0.0, 5.0, [1.0, 1.0])
    t = 1.23
    state = np.array([2.0, -1.0])
    f = ode.evalue(t, state)
    assert f.shape == state.shape

    J = ode.jacobien(t, state)
    assert J.shape == (2, 2)
