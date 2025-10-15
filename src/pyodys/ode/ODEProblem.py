from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import identity, csc_matrix
import sys

try:
    from ..utils.pyodys_utils import _DEFAULT_ZERO_TOL
except ImportError:
    _DEFAULT_ZERO_TOL = 1e-15

# ====================================================================
# FINAL METHOD ENFORCEMENT UTILITIES
# ====================================================================

def final_method(method):
    """
    Marks a method as final, preventing its override in subclasses.

    Adds a private attribute '__is_final' to the method. 
    The 'FinalChecker' metaclass inspects this attribute to enforce the finality.
    
    Parameters
    ----------
    method : callable
        The method to be marked as final.

    Returns
    -------
    callable
        The same method with the '__is_final' attribute set.
    """
    method.__is_final = True
    return method

def check_final_methods(cls):
    """
    Verifies if the subclass `cls` attempts to override a method marked as final in a base class.

    Raises
    ------
    TypeError
        If a subclass overrides a method that is marked as final in any of its base classes.

    Parameters
    ----------
    cls : type
        Subclass to check for illegal overrides.

    Returns
    -------
    type
        The same class if no violations are found.
    """
    
    # Check all attributes defined in the new subclass
    for name, value in cls.__dict__.items():
        if not callable(value) or name.startswith('__'):
            continue
            
        # Iterate over the inheritance hierarchy (MRO)
        for base in cls.__mro__[1:]:
            base_method = getattr(base, name, None)
            
            # 1. Check if the base method is marked as final
            if base_method is not None and getattr(base_method, '__is_final', False):
                
                # 2. Check if the method has been redefined (by comparing qualified names)
                try:
                    is_redefined = getattr(value, '__qualname__') != getattr(base_method, '__qualname__')
                except AttributeError:
                    is_redefined = True 

                if is_redefined:
                     raise TypeError(
                        f"Final method '{name}' from base class '{base.__name__}' "
                        f"cannot be overridden in '{cls.__name__}'. "
                        f"Mass matrix definition must be done in the '_compute_mass_matrix' helper."
                    )
    return cls

class FinalChecker(type(ABC)):
    """
    Metaclass that enforces 'final' method rules.

    During class creation, it inspects all callable attributes of the subclass 
    and raises an error if a 'final' method from a base class is overridden.
    """
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        # Only check if it's a subclass (not the base class itself)
        if name != 'ODEProblem': 
            cls = check_final_methods(cls)
        return cls

# ====================================================================
# MAIN CLASS: ODEProblem
# ====================================================================

class ODEProblem(ABC, metaclass=FinalChecker):
    """
    Abstract base class for systems of ODEs or DAEs of the form:

        M(t, u) * du/dt = F(t, u)

    where `M` is the mass matrix (can be the identity) and `F` is the system RHS.

    Subclasses must implement:
    - `evaluate_at(t, state)` : returns F(t, u)
    - `_compute_mass_matrix(t, state)` if the mass matrix is not identity.

    Attributes
    ----------
    AVAILABLE_FINITE_DIFFERENCE_SCHEMES : list[str]
        Allowed schemes for numerical Jacobian computation: 'central', 'forward', 'backward'.
    """

    AVAILABLE_FINITE_DIFFERENCE_SCHEMES = ['central', 'forward', 'backward']

    def __init__(self, t_init: float, 
                 t_final: float, 
                 initial_state: ArrayLike, 
                 finite_difference_scheme:str = "central",
                 jacobian_is_sparse = False,
                 jacobian_is_constant: bool = False,
                 mass_matrix_is_constant: bool = False):
        """
        Initialize an ODE/DAE system.
    
        Parameters
        ----------
        t_init : float
            Initial simulation time.
        t_final : float
            Final simulation time (must be strictly greater than `t_init`).
        initial_state : ArrayLike
            Initial state vector (1D array, non-empty).
        finite_difference_scheme : str, optional
            Scheme for numerical Jacobian: 'central', 'forward', or 'backward'.
        jacobian_is_sparse : bool, optional
            Indicates if the Jacobian should be stored in sparse format.
        jacobian_is_constant : bool, optional
            If True, the Jacobian is computed once and cached.
        mass_matrix_is_constant : bool, optional
            If True, the mass matrix is constant and cached.
        
        Raises
        ------
        ValueError
            If validation of input arguments fails (time ordering, array shape, or finite difference scheme).
        """
        
        # --- Validation ---
        if not np.isscalar(t_init) or not np.isreal(t_init):
            raise ValueError("t_init must be a real numeric scalar.")
        if not np.isscalar(t_final) or not np.isreal(t_final):
            raise ValueError("t_final must be a real numeric scalar.")
        if t_final <= t_init:
            raise ValueError("t_final must be strictly greater than t_init.")
        
        if finite_difference_scheme not in self.AVAILABLE_FINITE_DIFFERENCE_SCHEMES:
            raise ValueError(f"Invalid finite difference scheme. Available schemes are: {self.AVAILABLE_FINITE_DIFFERENCE_SCHEMES}")

        self.initial_state = np.atleast_1d(np.array(initial_state, dtype=np.float64))
        if self.initial_state.ndim != 1 or self.initial_state.size == 0:
            raise ValueError("initial_state must be a non-empty 1D array.")

        # --- Parameter Storage ---
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.finite_difference_scheme = finite_difference_scheme
        self.jacobian_is_constant = jacobian_is_constant
        self.mass_matrix_is_constant = mass_matrix_is_constant
        self.number_of_equations = len(initial_state)
        self.jacobian_is_sparse = jacobian_is_sparse
        self._cached_jacobian = None
        self._cached_mass_matrix = None
        self._delta = np.sqrt(np.finfo(float).eps) if self.finite_difference_scheme in ['forward', 'backward'] else (np.finfo(float).eps)**(1.0/3.0)

        # --- Automatic Identity Detection (Introspection) ---
        # Checks if _compute_mass_matrix is the one inherited from the base class (ODEProblem).
        is_base_method = (self._compute_mass_matrix.__qualname__.split('.')[0] == 'ODEProblem')
        self.mass_matrix_is_identity = is_base_method
        
        if self.mass_matrix_is_identity:
            # If M=I, it is implicitly constant
            self.mass_matrix_is_constant = True


    @abstractmethod
    def evaluate_at(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        REQUIRED: Evaluate the right-hand side of the system, F(t, u).

        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Derivative vector F(t, u).
        """
        pass

    def jacobian_at(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the numerical or cached Jacobian matrix of F with respect to u, J = dF/du.
        Subclasses may override this method for an analytical Jacobian.
        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector.

         Returns:
            np.ndarray: The Jacobian of F(t, u).
        """
        if self.jacobian_is_constant:
            if self._cached_jacobian is None:
                self._cached_jacobian = self._compute_jacobian(t, state) 
            return self._cached_jacobian
        else:
            return self._compute_jacobian(t, state)

    def _get_jacobian_sparsity_pattern(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Optional helper method to return the structural non-zero pattern of the Jacobian.
        Subclasses should return a tuple of (row_indices, col_indices) for the 
        non-zero entries if self.jacobian_is_sparse is True.
        """
        return None
    
    def _compute_jacobian(self, t: float, state: np.ndarray) -> np.ndarray | csc_matrix:
        """
        Helper method that dispatches the Jacobian computation based on the selected scheme 
        and sparsity flag.
        """
        if self.finite_difference_scheme == 'central':
            if self.jacobian_is_sparse:
                return self._compute_jacobian_central_sparse(t, state)
            else:
                return self._compute_jacobian_central_dense(t, state)
        
        elif self.finite_difference_scheme == 'forward':
            if self.jacobian_is_sparse:
                return self._compute_jacobian_forward_sparse(t, state)
            else:
                return self._compute_jacobian_forward_dense(t, state)

        elif self.finite_difference_scheme == 'backward':
            if self.jacobian_is_sparse:
                return self._compute_jacobian_backward_sparse(t, state)
            else:
                return self._compute_jacobian_backward_dense(t, state)
        
        # This should not be reached due to validation in __init__
        raise NotImplementedError("Selected finite difference scheme not implemented.")

    def _compute_jacobian_forward_dense(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Forward difference, dense output (O(h) accuracy).
        """
        n = len(state)
        Jacobian = np.zeros((n, n), dtype=np.float64)
        perturbed_state = state.copy()
        
        f_unperturbed = self.evaluate_at(t, state) # Base F(u)
        
        for j in range(n):
            h = self._delta*max(abs(state[j]), 1.0)
            # F(u + h)
            perturbed_state[j] += h
            f_right = self.evaluate_at(t, perturbed_state)
            
            # Forward difference: [F(u+h) - F(u)] / h
            Jacobian[:, j] = (f_right - f_unperturbed) / h
            
            # Restore state for next column
            perturbed_state[j] = state[j] 
        return Jacobian
    
    def _compute_jacobian_backward_dense(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Backward difference, dense output (O(h) accuracy).
        """
        n = len(state)
        Jacobian = np.zeros((n, n), dtype=np.float64)
        perturbed_state = state.copy()
        
        f_unperturbed = self.evaluate_at(t, state) # Base F(u)
        
        for j in range(n):
            h = self._delta*max(abs(state[j]), 1.0)
            # F(u + h)
            perturbed_state[j] -= h
            f_left = self.evaluate_at(t, perturbed_state)
            
            # Backward difference: [F(u) - F(u-h) ] / h
            Jacobian[:, j] = (f_unperturbed - f_left) / h
            
            # Restore state for next column
            perturbed_state[j] = state[j] 
        return Jacobian
    
    def _compute_jacobian_central_dense(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Central difference, dense output (O(h^2) accuracy).
        """
        n = len(state)
        Jacobian = np.zeros((n, n), dtype=np.float64)
        perturbed_state = state.copy()
                
        for j in range(n):
           h = self._delta*max(abs(state[j]), 1.0)
           # F(u + h)
           perturbed_state[j] += h
           f_right = self.evaluate_at(t, perturbed_state)

           # F(u - h)
           perturbed_state[j] -= 2*h
           f_left = self.evaluate_at(t, perturbed_state)

           # Central difference approximation: [F(u+h) - F(u-h)] / (2h)
           Jacobian[:, j] = (f_right - f_left) / (2 * h)

           # Restore for the next column
           perturbed_state[j] = state[j]

        return Jacobian
    
    def _compute_jacobian_forward_sparse(self, t: float, state: np.ndarray) -> np.ndarray | csc_matrix:
        """
        Forward difference, sparse output (O(h) accuracy).
        """
        n = len(state)

        # 1. Attempt to get the sparsity pattern (optional optimization)
        sparsity_pattern = self._get_jacobian_sparsity_pattern()
        data_list = []

        perturbed_state = state.copy()
        f_unperturbed = self.evaluate_at(t, state) # Base F(u)
        for j in range(n):
            h = self._delta*max(abs(state[j]), 1.0)
            # Calculate the column vector difference (creates a dense N-vector)
            perturbed_state[j] += h
            f_right = self.evaluate_at(t, perturbed_state)

            # Forward difference approximation: [F(u+h) - F(u)] / h
            column_data = (f_right - f_unperturbed) / h

            # Restore for the next column
            perturbed_state[j] = state[j]
            
            # Extract non-zero entries based on the best available information
            data_list = self._extract_nonzeros_entry(j, data_list, column_data, sparsity_pattern)

        return self._build_sparse_matrix_from_triplets(data_list, dim = n)

    def _compute_jacobian_backward_sparse(self, t: float, state: np.ndarray) -> np.ndarray | csc_matrix:
        """
        Backward difference, sparse output (O(h) accuracy).
        """
        n = len(state)

        # 1. Attempt to get the sparsity pattern (optional optimization)
        sparsity_pattern = self._get_jacobian_sparsity_pattern()
        data_list = []

        perturbed_state = state.copy()
        f_unperturbed = self.evaluate_at(t, state) # Base F(u)
        for j in range(n):
            h = self._delta*max(abs(state[j]), 1.0)
            # Calculate the column vector difference (creates a dense N-vector)
            perturbed_state[j] -= h
            f_left = self.evaluate_at(t, perturbed_state)

            # Backward difference approximation: [F(u) - F(u-h)] / h
            column_data = (f_unperturbed - f_left) / h

            # Restore for the next column
            perturbed_state[j] = state[j]
            
            # Extract non-zero entries based on the best available information
            data_list = self._extract_nonzeros_entry(j, data_list, column_data, sparsity_pattern)

        return self._build_sparse_matrix_from_triplets(data_list, dim = n)

    def _compute_jacobian_central_sparse(self, t: float, state: np.ndarray) -> np.ndarray | csc_matrix:
        """
        Central difference, sparse output (O(h^2) accuracy).
        """
        n = len(state)

        # 1. Attempt to get the sparsity pattern (optional optimization)
        sparsity_pattern = self._get_jacobian_sparsity_pattern()
        data_list = []

        perturbed_state = state.copy()
        for j in range(n):
            h = self._delta*max(abs(state[j]), 1.0)
            # F(u + h)
            perturbed_state[j] += h
            f_right = self.evaluate_at(t, perturbed_state)

            # F(u - h)
            perturbed_state[j] -= 2 * h
            f_left = self.evaluate_at(t, perturbed_state)

            # Central difference approximation: [F(u+h) - F(u-h)] / (2h)
            column_data = (f_right - f_left) / (2 * h)

            # Restore for the next column 
            perturbed_state[j] = state[j]
            
            # Extract non-zero entries based on the best available information
            data_list = self._extract_nonzeros_entry(j, data_list, column_data, sparsity_pattern)

        return self._build_sparse_matrix_from_triplets(data_list, dim = n)
    
    def _extract_nonzeros_entry(self, j:int, data_list:list, column_data, sparsity_pattern):
        """
        Helper to append non-zero entries to the data list, applying tolerance.
        """
        if sparsity_pattern is not None:  # Strategy 1: Use the provided pattern (best filtering/control)
            rows, cols = sparsity_pattern
            nz_rows_for_j = rows[cols == j]
            
            for i in nz_rows_for_j:
                value = column_data[i]
                if np.abs(value) > _DEFAULT_ZERO_TOL:
                    data_list.append((i, j, value))
        else: # Strategy 2: Compute and filter non-zeros dynamically (user-friendly default)
            nz_indices = np.where(np.abs(column_data) > _DEFAULT_ZERO_TOL)[0]
            
            for i in nz_indices:
                data_list.append((i, j, column_data[i]))
        return data_list
    
    def _build_sparse_matrix_from_triplets(self, data_list:list, dim:int) -> csc_matrix:
        """
        Helper to build a CSC matrix from (row, col, data) triplets.
        """
        if not data_list:
             return csc_matrix((dim, dim), dtype=np.float64)
        final_rows = np.array([d[0] for d in data_list])
        final_cols = np.array([d[1] for d in data_list])
        final_data = np.array([d[2] for d in data_list])
        return csc_matrix((final_data, (final_rows, final_cols)), shape=(dim, dim))

    def _compute_mass_matrix(self, t: float, state: np.ndarray):
        """
        Helper method to compute the Mass Matrix M(t, u). 
        Subclasses MUST override this if M is NOT the Identity matrix. 
        
        If this method is NOT overridden, the base class assumes M = Identity.
        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector.

         Returns:
            np.ndarray: The mass matrix M(t, y)evaluated at (t, y=state)
        """
        return None

    @final_method
    def mass_matrix_at(self, t: float, state: np.ndarray):
        """
        [FINAL METHOD] Returns the Mass Matrix M(t, u), handling caching and identity checks.
        MUST NOT BE OVERRIDDEN BY SUBCLASSES.

        Args:
            t (float): Current simulation time.
            state (np.ndarray): Current state vector.

         Returns:
            np.ndarray: The mass matrix M(t, y)evaluated at (t, y=state)

        """
        if self.mass_matrix_is_identity:
            # Case 1: Identity Matrix (M=I)
            if self._cached_mass_matrix is None:
                n = self.number_of_equations
                # Use sparse identity for large systems
                self._cached_mass_matrix = identity(n, dtype=float, format='csc') if n > 100 else np.identity(n, dtype=float)
            return self._cached_mass_matrix

        if self.mass_matrix_is_constant:
            # Case 2: Constant Matrix (Calculated and cached once)
            if self._cached_mass_matrix is None:
                self._cached_mass_matrix = self._compute_mass_matrix(t, state)
            return self._cached_mass_matrix

        # Case 3: Variable Matrix (Calculated at every call by the subclass)
        return self._compute_mass_matrix(t, state)