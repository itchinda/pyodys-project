from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class ODEProblem(ABC):
    """Abstract base class for systems of Ordinary Differential Equations (ODEs).

    Any subclass must implement the :meth:`evaluate_at` method, which defines the ODE system.

    Attributes:
        t_init (float): Initial simulation time.
        t_final (float): Final simulation time.
        initial_state (np.ndarray): Initial state vector of the system.
        delta (float): Perturbation used for numerical Jacobian approximation.
    """

    def __init__(self, t_init: float, t_final: float, initial_state: ArrayLike, delta: float = 1e-5, jacobian_is_constant: bool = False):
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