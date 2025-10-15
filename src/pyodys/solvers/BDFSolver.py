from ..ode.ODEProblem import ODEProblem
from ..schemes.bdf.BDFScheme import BDFScheme
from .SolverBase import SolverBase
from ..utils import pyodys_utils as utils
import numpy as np
from typing import Union, Callable
from scipy.linalg import lu_factor, lu_solve, LinAlgError
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix, isspmatrix
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

class BDFSolver(SolverBase):
    """
    A placeholder for a BDF (Backward Differentiation Formula) solver for
    ordinary differential equations (ODEs).

    Currently, the BDF solver is **not implemented**. Attempting to instantiate
    this class will raise an Exception. Users should use `PyodysSolver` with
    a Runge-Kutta scheme (RKScheme) for now.

    Intended Features (for future implementation)
    ---------------------------------------------
    - Implicit multi-step solver for stiff ODEs
    - Adaptive and fixed step size support
    - Efficient handling of dense and sparse Jacobians
    - Newton iteration with automatic Jacobian refresh
    - Optional CSV export and progress reporting

    Parameters
    ----------
    method : (BDFScheme | str, optional): The BDF scheme to use, or its name as a string.
    fixed_step : (float, optional): Step size for fixed-step integration.
    adaptive : (bool, default=False) If True, enable adaptive step size control.
    first_step : float, optional
        Initial step size for adaptive integration.
    min_step : float, optional
        Minimum step size allowed for adaptive integration.
    max_step : float, optional
        Maximum step size allowed for adaptive integration.
    nsteps_max : int, default=1_000_000
        Maximum number of steps allowed during integration.
    newton_nmax : int, default=10
        Maximum number of Newton iterations per step.
    rtol : float, default=1e-8
        Relative tolerance for adaptive step size control.
    atol : float, default=1e-8
        Absolute tolerance for adaptive step size control.
    max_jacobian_refresh : int, default=1
        Maximum number of Jacobian recomputations if Newton fails.
    verbose : bool, default=False
        Enable progress and debug printing.
    progress_interval_in_time : float, optional
        Time interval to print progress updates.
    export_interval : int, optional
        Number of steps between CSV exports.
    export_prefix : str, optional
        File prefix for CSV export.
    auto_check_sparsity : bool, default=True
        Automatically detect sparse Jacobians.
    sparse_threshold : int, default=20
        Minimum system size to trigger sparse detection.
    sparsity_ratio_limit : float, default=0.2
        Threshold to classify a Jacobian as sparse.
    initial_step_safety : float, default=1e-4
        Safety factor for the first step size.
        
    Raises
    ------
    PyodysError
        Always raises a PyodysError exception because BDF is not implemented.
    TypeError
        If `method` is not a BDFScheme instance or a valid scheme name.
    ValueError
        If `method` is a string but does not correspond to a known BDF scheme.
    Notes
    -----
    This class currently only serves as a placeholder. Users should use
    Runge-Kutta solvers (RKSolver) via `PyodysSolver` until BDF is implemented.

    Jacobian Handling Policy
    ------------------------
    - If `ode_problem.jacobian_is_constant = True`:
      Compute the Jacobian once at initialization and reuse it throughout the integration.
    
    - If `ode_problem.jacobian_is_constant = False`:
      Recompute the Jacobian at the beginning of each time step.
      If Newton iterations fail to converge, refresh the Jacobian and retry
      (up to `max_jacobian_refresh` times).
    """
    def __init__(self,
                 method: Union[BDFScheme, str] = None,
                 fixed_step: float = None,
                 adaptive: bool = False,
                 first_step: float = None,
                 min_step: float = None, 
                 max_step: float = None,
                 nsteps_max: int = 1000000,
                 newton_nmax: int = 10,
                 rtol: float = 1e-8,
                 atol: float = 1e-8,
                 linear_solver: Union[str, Callable] = "lu",
                 linear_solver_opts:dict = None,
                 max_jacobian_refresh: int = 1,
                 verbose: bool = False,
                 progress_interval_in_time: int = None,
                 export_interval: int = None,
                 export_prefix: str = None,
                 auto_check_sparsity: bool = True,
                 sparse_threshold: int = 20,
                 sparsity_ratio_limit: float = 0.2,
                 initial_step_safety = 1e-4):
        """Initialize a BDF solver. """
    
        raise utils.PyodysError("The BDF solver is not yet implemented. Consider using a Runge-Kutta scheme.")

        super().__init__(
            fixed_step = fixed_step,
            adaptive = adaptive,
            first_step = first_step,
            min_step = min_step, 
            max_step = max_step,
            nsteps_max = nsteps_max,
            newton_nmax = newton_nmax,
            rtol = rtol,
            atol = atol,
            linear_solver=linear_solver,
            linear_solver_opts = linear_solver_opts,
            max_jacobian_refresh = max_jacobian_refresh,
            verbose = verbose,
            progress_interval_in_time = progress_interval_in_time,
            export_interval = export_interval,
            export_prefix = export_prefix,
            auto_check_sparsity = auto_check_sparsity,
            sparse_threshold = sparse_threshold,
            sparsity_ratio_limit = sparsity_ratio_limit,
            initial_step_safety=initial_step_safety)
        # Resolve RK scheme
        if isinstance(method, str):
            available = "\n".join(BDFScheme.available_schemes())
            if method not in BDFScheme.available_schemes():
                raise ValueError(
                    f"There is no available scheme with name {method}. "
                    f"Here is the list of available schemes:\n{available}"
                )
            scheme =  BDFScheme.from_name(method)
            self.alpha = scheme.alpha
            self.beta = scheme.beta
            self._error_estimator_order = scheme.order
        elif isinstance(method, BDFScheme):
            self.alpha = method.alpha
            self.beta = method.beta
            self._error_estimator_order = method.order
        else:
            raise TypeError("method must be an BDFScheme instance or a scheme name string.")

        self._linear_sparse_solver = None
        self._linear_dense_solver = None

        

        self._work_U_chap = None
        self._work_deltat_x_value_f = None
        self._work_U_pred = None
        self._work_U_n = None
        self._work_U_newton = None
        self._nb_equations = None

    def solve(self, ode_problem : ODEProblem):
        raise utils.PyodysError("The BDF solver is not yet implemented. Consider using a Runge-Kutta scheme.")
