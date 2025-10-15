import numpy as np
import pytest
from pyodys import ODEProblem


# ============================================================
# Dummy subclasses for testing
# ============================================================

class LinearSystem(ODEProblem):
    """Simple linear system: dx/dt = A x"""

    def __init__(self, t_init, t_final, initial_state, A, **kwargs):
        super().__init__(t_init=t_init, t_final=t_final, initial_state=initial_state, **kwargs)
        self.A = np.array(A, dtype=np.float64)

    def evaluate_at(self, t, state):
        return self.A @ state

class TimeDependentSystem(ODEProblem):
    """System whose Jacobian depends explicitly on time."""

    def __init__(self, t_init, t_final, initial_state, **kwargs):
        super().__init__(t_init=t_init, t_final=t_final, initial_state=initial_state, **kwargs)

    def evaluate_at(self, t, state):
        # dx/dt = (t + 1) * x
        return (t + 1.0) * state

class LorenzSystem(ODEProblem):
    """Nonlinear Lorenz system for testing Jacobian correctness."""
    def __init__(self, t_init, t_final, initial_state, sigma=10.0, rho=28.0, beta=8/3, **kwargs):
        super().__init__(t_init=t_init, t_final=t_final, initial_state=initial_state, **kwargs)
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def evaluate_at(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def analytical_jacobian(self, state):
        x, y, z = state
        return np.array([
            [-self.sigma,  self.sigma,       0.0],
            [self.rho - z, -1.0,           -x  ],
            [y,             x,             -self.beta]
        ])
    
class AnalyticJacobianSystem(LinearSystem):
    """System providing its own analytical Jacobian."""

    def jacobian_at(self, t, state):
        return np.copy(self.A)



# ============================================================
# Abstract class enforcement
# ============================================================

def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        ODEProblem(t_init=0.0, t_final=1.0, initial_state=[1.0])

# ============================================================
# Constructor validation tests
# ============================================================

def test_valid_construction():
    sys = LinearSystem(0.0, 1.0, [1.0, 2.0], np.eye(2))
    assert sys.t_init == 0.0
    assert sys.t_final == 1.0
    assert np.allclose(sys.initial_state, [1.0, 2.0])

@pytest.mark.parametrize("t_init,t_final", [(1.0, 1.0), (2.0, 1.0)])
def test_invalid_time_order(t_init, t_final):
    with pytest.raises(ValueError, match="t_final must be strictly greater than t_init."):
        LinearSystem(t_init, t_final, [1.0], np.eye(1))

@pytest.mark.parametrize("value", ["abc", None, [1, 2]])
def test_invalid_t_init_type(value):
    with pytest.raises(ValueError, match="t_init must be a real numeric scalar."):
        LinearSystem(value, 1.0, [1.0], np.eye(1))

@pytest.mark.parametrize("value", ["abc", None, [1, 2]])
def test_invalid_t_final_type(value):
    with pytest.raises(ValueError, match="t_final must be a real numeric scalar."):
        LinearSystem(0.0, value, [1.0], np.eye(1))

def test_invalid_initial_state_empty():
    with pytest.raises(ValueError, match="initial_state must be a non-empty 1D array."):
        LinearSystem(0.0, 1.0, [], np.eye(1))

def test_invalid_initial_state_ndim():
    with pytest.raises(ValueError, match="initial_state must be a non-empty 1D array."):
        LinearSystem(0.0, 1.0, [[1.0, 2.0]], np.eye(2))


# ============================================================
# Jacobian tests (dense mode)
# ============================================================

def test_jacobian_linear_system_identity():
    A = np.eye(2)
    sys = LinearSystem(0.0, 1.0, [1.0, 0.0], A)
    J = sys.jacobian_at(0.0, np.array([1.0, 0.0]))
    assert np.allclose(J, A, atol=1e-8)


def test_jacobian_linear_system_nontrivial():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A)
    J = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J, A, atol=1e-6)


@pytest.mark.parametrize("scheme", ["forward", "backward", "central"])
def test_jacobian_all_schemes(scheme):
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A,
                       finite_difference_scheme=scheme)
    J = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J, A, atol=1e-6)


def test_jacobian_zero_system():
    A = np.zeros((2, 2))
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A)
    J = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J, np.zeros((2, 2)), atol=1e-12)


def test_user_provided_jacobian_overrides_numerical():
    A = np.array([[0.0, 2.0], [-1.0, -3.0]])
    sys = AnalyticJacobianSystem(0.0, 1.0, [1.0, 1.0], A)
    J = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J, A) 
    assert J is not A

def test_constant_jacobian_caching():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A,
                       jacobian_is_constant=True)
    J1 = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    J2 = sys.jacobian_at(0.5, np.array([2.0, 3.0]))
    assert J1 is J2  # same object -> cached
    assert np.allclose(J1, A)


# ============================================================
# Sparse Jacobian tests
# ============================================================

def test_sparse_jacobian_linear_system():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A, jacobian_is_sparse=True)
    J_sparse = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J_sparse.toarray(), A, atol=1e-6)


def test_jacobian_with_sparsity_pattern():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    pattern = np.abs(A) > 0
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A,
                       jacobian_is_sparse=True)
    sys._get_jacobian_sparsity_pattern = lambda: np.nonzero(pattern)
    J_sparse = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    assert np.allclose(J_sparse.toarray(), A, atol=1e-6)


def test_sparse_constant_jacobian_caching():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    sys = LinearSystem(0.0, 1.0, [1.0, 1.0], A,
                       jacobian_is_sparse=True, jacobian_is_constant=True)
    J1 = sys.jacobian_at(0.0, np.array([1.0, 1.0]))
    J2 = sys.jacobian_at(0.5, np.array([2.0, 3.0]))
    assert J1 is J2  # Cached reuse
    assert np.allclose(J1.toarray(), A, atol=1e-6)

# ============================================================
# Non-constant Jacobian recomputation tests
# ============================================================

def test_nonconstant_jacobian_is_recomputed_each_time_time_dependent():
    sys = TimeDependentSystem(0.0, 2.0, [1.0], jacobian_is_constant=False)
    x = np.array([1.0])
    J1 = sys.jacobian_at(0.0, x)
    J2 = sys.jacobian_at(1.0, x)
    assert not np.allclose(J1, J2)
    assert J1 is not J2


def test_nonconstant_jacobian_is_recomputed_each_time_state_dependent():
    sys = LorenzSystem(0.0, 1.0, [1.0, 1.0, 1.0], jacobian_is_constant=False)
    J1 = sys.jacobian_at(0.0, np.array([1.0, 1.0, 1.0]))
    J2 = sys.jacobian_at(0.0, np.array([2.0, 2.0, 2.0]))
    assert not np.allclose(J1, J2)
    assert J1 is not J2

# ============================================================
# Finite difference schemes validation
# ============================================================
@pytest.mark.parametrize("finite_difference_scheme", ['forward', 'backward', 'central'])
def test_nonconstant_jacobian_is_recomputed_each_time_state_dependent(finite_difference_scheme):
    kwargs = dict(finite_difference_scheme = finite_difference_scheme)
    sys = LorenzSystem(0.0, 1.0, [1.0, 1.0, 1.0], jacobian_is_constant=False, **kwargs)
    J1 = sys.jacobian_at(0.0, np.array([1.0, 2.0, 3.0]))
    J_anal = sys.analytical_jacobian(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(J1, J_anal, rtol=1e-6)

import re
def test_invalid_finite_difference_scheme():
    """Ensure invalid finite difference scheme raises ValueError."""
    with pytest.raises(ValueError, match=re.escape(f"Invalid finite difference scheme. Available schemes are: {ODEProblem.AVAILABLE_FINITE_DIFFERENCE_SCHEMES}")):
        LorenzSystem(
            t_init=0.0,
            t_final=1.0,
            initial_state=[1.0, 1.0, 1.0],
            finite_difference_scheme="invalid_scheme",
        )

# ============================================================
# Nonlinear system correctness (Lorenz)
# ============================================================

def test_lorenz_jacobian_matches_analytical():
    sys = LorenzSystem(0.0, 1.0, [1.0, 2.0, 3.0], jacobian_is_constant=False)
    state = np.array([1.0, 2.0, 3.0])
    J_numerical = sys.jacobian_at(0.0, state)
    J_exact = sys.analytical_jacobian(state)
    assert np.allclose(J_numerical, J_exact, rtol=1e-5)

# ============================================================
# Mass matrix tests
# ============================================================

def test_mass_matrix_identity():
    sys = LinearSystem(0.0, 1.0, [1.0, 0.0], np.eye(2))
    M = sys.mass_matrix_at(0.0, np.array([1.0, 0.0]))
    assert np.allclose(M, np.eye(2))


def test_mass_matrix_constant():
    sys = LinearSystem(0.0, 1.0, [1.0], np.eye(1),
                       mass_matrix_is_constant=True)
    M1 = sys.mass_matrix_at(0.0, np.array([1.0]))
    M2 = sys.mass_matrix_at(0.5, np.array([2.0]))
    assert M1 is M2
    assert np.allclose(M1, np.eye(1))


def test_mass_matrix_variable_default_identity():
    sys = LinearSystem(0.0, 1.0, [1.0], np.eye(1),
                       mass_matrix_is_constant=False)
    M = sys.mass_matrix_at(0.0, np.array([1.0]))
    assert np.allclose(M, np.eye(1))


def test_illegal_mass_matrix_override():
    """Ensure metaclass protection prevents overriding @final method."""
    with pytest.raises(TypeError):
        class FakeOverrideMassMatrix(LinearSystem):
            """Illegally tries to override mass_matrix_at (should fail)."""
            def mass_matrix_at(self, t, state):  
                return np.eye(len(state))

