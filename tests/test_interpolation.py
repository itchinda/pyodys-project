import numpy as np
import pytest
from pyodys import ODEProblem, hermite_interpolate  # replace with actual import path

# Minimal example ODE system
class Systeme(ODEProblem):
    def __init__(self, t_init, t_final, u_init):
        super().__init__(t_init, t_final, u_init)
    
    def evaluate_at(self, t, u):
        x, y = u
        return np.array([-x + y, -y])

@pytest.fixture
def simple_system():
    return Systeme(0.0, 10.0, [1.0, 1.0])

def solution_analytique(t):
    x = np.exp(-t) * (1 +  t)
    y =  np.exp(-t)
    return np.column_stack((x, y))  


@pytest.fixture
def example_data():
    t_i = np.linspace(0, 2, 5)
    y_i = solution_analytique(t_i)
    t_new = np.linspace(0, 1, 10)
    return t_i, y_i, t_new

def test_hermite_interpolate_shape(simple_system, example_data):
    t_i, y_i, t_new = example_data
    y_new = hermite_interpolate(t_i, y_i, simple_system, t_new)
    assert y_new.shape == (len(t_new), y_i.shape[1]), "Output shape mismatch"

def test_hermite_interpolate_known_values(simple_system, example_data):
    t_i, y_i, t_new = example_data
    y_new = hermite_interpolate(t_i, y_i, simple_system, t_new)

    # Check that interpolation recovers original points (t_i)
    y_interp_at_ti = hermite_interpolate(t_i, y_i, simple_system, t_i)
    np.testing.assert_allclose(y_interp_at_ti, y_i, rtol=1e-5, atol=1e-8)

def test_hermite_interpolate_monotonicity(simple_system):
    # Simple monotone scalar example
    t_i = np.array([0, 1, 2])
    y_i = np.array([[0], [1], [2]])
    t_new = np.linspace(0, 2, 5)
    y_new = hermite_interpolate(t_i, y_i, lambda t, y: np.array([1.0]), t_new)
    assert np.all(np.diff(y_new[:,0]) > 0), "Interpolation should be monotone decreasing"

def test_hermite_interpolate_exact_cubic():
    a, b, c, d = 2.0, -3.0, 1.0, 5.0
    def cubic_function(t):
        return a*t**3 + b*t**2 + c*t + d

    def cubic_derivative(t):
        return 3*a*t**2 + 2*b*t + c

    t_i = np.array([0.0, 0.5, 1.0, 1.5])
    y_i = cubic_function(t_i).reshape(-1, 1)
    d_i = cubic_derivative(t_i).reshape(-1, 1)

    t_new = np.linspace(0.0, 1.5, 20)

    # Using a lambda that returns derivative
    y_new = hermite_interpolate(t_i, y_i, lambda t, y: cubic_derivative(t), t_new)

    # True values
    y_true = cubic_function(t_new).reshape(-1, 1)

    np.testing.assert_allclose(y_new, y_true, rtol=1e-12, atol=1e-12)