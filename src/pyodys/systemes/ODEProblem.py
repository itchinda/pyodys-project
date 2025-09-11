from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import identity

class ODEProblem(ABC):
    """Abstract base class for systems of Ordinary Differential Equations (ODEs). General form: M(t, u) dudt = F(t, u)

    Any subclass must implement the :meth:`evaluate_at` method, which defines the ODE system.

    Attributes:
        t_init (float): Initial simulation time.
        t_final (float): Final simulation time.
        initial_state (np.ndarray): Initial state vector of the system.
        delta (float): Perturbation used for numerical Jacobian approximation.
    """

    def __init__(self, t_init: float, 
                 t_final: float, 
                 initial_state: ArrayLike, 
                 delta: float = 1e-5, 
                 jacobian_is_constant: bool = False,
                 mass_matrix_is_constant: bool = True,
                 mass_matrix_is_identity: bool = False,
                 mass_matrix_jacobian_is_null: bool = True):
        """Initialize an ODE system.

        Args:
            t_init (float): Initial simulation time. Must be strictly less than `t_final`.
            t_final (float): Final simulation time. Must be strictly greater than `t_init`.
            initial_state (ArrayLike): Initial state vector of the system.
                                Must be convertible to a 1D NumPy array of floats.
            delta (float, optional): Perturbation for finite differences. Defaults to 1e-5.
            jacobian_is_constant (bool, optional): Flag to indicate if the Jacobian is constant. Defaults to False.

        Raises:
            ValueError: If `t_final <= t_init`.
            ValueError: If `initial_state` is empty or not 1D.
            ValueError: If `t_init` or `t_final` are not real scalars.
        """
        # Validate types for t_init and t_final
        if not np.isscalar(t_init) or not np.isreal(t_init):
            raise ValueError("t_init must be a real numeric scalar.")
        if not np.isscalar(t_final) or not np.isreal(t_final):
            raise ValueError("t_final must be a real numeric scalar.")
        if t_final <= t_init:
            raise ValueError("t_final must be strictly greater than t_init.")

        # Validate initial_state
        self.initial_state = np.atleast_1d(np.array(initial_state, dtype=np.float64))
        if self.initial_state.ndim != 1:
            raise ValueError("initial_state must be a 1D array.")
        if self.initial_state.size == 0:
            raise ValueError("initial_state must be a non-empty array.")

        # Store parameters
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.delta = float(delta)

        self.jacobian_is_constant = jacobian_is_constant
        self._cached_jacobian = None

        self.mass_matrix_is_constant = mass_matrix_is_constant
        self._cached_mass_matrix = None

        self.mass_matrix_is_identity = mass_matrix_is_identity

        self.number_of_equations = len(initial_state)
        self.mass_matrix_jacobian_is_null = mass_matrix_jacobian_is_null 

    @abstractmethod
    def evaluate_at(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the system at time `t`.

        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector (1D array).

        Returns:
            np.ndarray: Derivative vector (same shape as `state`).

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError(
            "Each subclass must implement the `evaluate_at` method."
        )

    def jacobian_at(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the numerical Jacobian matrix of the ODE system.

        The Jacobian is approximated using central finite differences.
        If the Jacobian is constant, it is computed only once and cached.

        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector (1D array).

        Returns:
            np.ndarray: Jacobian matrix of shape (n, n), where n is the dimension of `state`.
        """
        if self.jacobian_is_constant:
            if self._cached_jacobian is None:
                self._cached_jacobian = self._compute_jacobian(t, state)
            return self._cached_jacobian
        else:
            return self._compute_jacobian(t, state)

    def _compute_jacobian(self, t: float, state: np.ndarray) -> np.ndarray:
        """Helper method to compute the numerical Jacobian."""
        n = len(state)
        Jacobian = np.zeros((n, n), dtype=np.float64)
        h = self.delta

        perturbed_state = state.copy()

        for j in range(n):
            perturbed_state[j] += h
            f_right = self.evaluate_at(t, perturbed_state)

            perturbed_state[j] -= 2 * h
            f_left = self.evaluate_at(t, perturbed_state)

            # Central difference approximation for the j-th column
            Jacobian[:, j] = (f_right - f_left) / (2 * h)

            # Restore the original value for the next iteration
            perturbed_state[j] = state[j]

        return Jacobian
    
    def _compute_mass_matrix(self, t: float, state: np.ndarray):
        """
        Helper method for subclasses to implement.

        This method must be implemented if mass_matrix_is_identity is False
        and mass_matrix_is_constant is False.
        """
        raise NotImplementedError(
            "This method must be implemented by the subclass because `mass_matrix_is_identity` "
            "is set to False and the mass matrix is not constant or not provided."
        )

    def mass_matrix_at(self, t: float, state: np.ndarray):
        if self.mass_matrix_is_identity:
            return identity(self.number_of_equations, format='csr') if self.number_of_equations else np.eye(self.number_of_equations)

        if self.mass_matrix_is_constant:
            if self._cached_mass_matrix is None:
                self._cached_mass_matrix = self._compute_mass_matrix(t, state)
            return self._cached_mass_matrix

        # If not constant or identity, we must compute it at each step
        return self._compute_mass_matrix(t, state)
