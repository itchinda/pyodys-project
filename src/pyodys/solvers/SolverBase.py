from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csc_matrix, identity, isspmatrix
from scipy.sparse.linalg import gmres, cg
import csv
import os
import warnings
from typing import Union, Callable

from ..ode.ODEProblem import ODEProblem
from ..utils import pyodys_utils as utils


warnings.filterwarnings("ignore", category=RuntimeWarning)

class SolverBase(ABC):
    """
    Base class for ODE solvers in the PyOdys library.

    This class provides a common interface and shared functionalities for all solvers,
    handling initialization, parameter validation, and utility functions such as
    data export and matrix classification. It is an abstract base class and should
    not be instantiated directly.

    Parameters
    ----------
    fixed_step : float, optional
        The fixed step size to use for non-adaptive solvers.
        Required if `adaptive` is False.
    adaptive : bool, default True
        Whether to use an adaptive time-stepping algorithm.
    first_step : float, optional
        The initial step size to use for adaptive solvers. If not
        provided, a safe initial step is estimated.
    min_step : float, optional, default None
        The minimum allowed step size for adaptive solvers.
        If None, Pyodys automatically estimate its value based on the problem.
    max_step : float, optional, default None
        The maximum allowed step size for adaptive solvers.
        If None, Pyodys automatically set its value equal to the size time span.
    nsteps_max : int, default None
        Maximum number of steps allowed. The solver will terminate if this
        limit is reached.
        If None, Pyodys automatically set its value based on the time span and, the available memory.
    newton_nmax : int, default 10
        Maximum number of Newton iterations for implicit solvers.
    rtol : float, default 1e-8
        The relative tolerance for adaptive error control. Required for
        adaptive solvers.
    atol : float, default 1e-8
        The absolute tolerance for adaptive error control. Required for
        adaptive solvers.
    linear_solver : Union[str, Callable], default 'lu'
        Linear solver used for implicit schemes.
    linear_solver_opts : dict, optional
        Additional options for the linear solver.
    max_jacobian_refresh : int, default 1
        Maximum number of times to re-evaluate the Jacobian for implicit
        solvers.
    verbose : bool, default False
        If True, prints detailed information about the solver's progress.
    progress_interval_in_time : int, optional
        If provided, the solver will print progress at regular time intervals.
    export_interval : int, optional
        If provided, the solver will export results at regular step intervals.
    export_prefix : str, optional
        The prefix for exported CSV file names. If provided, results are
        automatically exported.
    auto_check_sparsity : bool, default True
        If True, the solver automatically checks matrix density and switches
        to sparse algebra if the matrix is sufficiently sparse.
    sparse_threshold : int, default 20
        The minimum size (number of equations) of a system for which a sparsity
        check is performed.
    sparsity_ratio_limit : float, default 0.2
        The maximum ratio of non-zero elements (density) for a matrix to be
        considered sparse and use sparse algebra.
    initial_step_safety : float, default 1e-4
        A safety factor used during the initial step size estimation for adaptive solvers.
    """
    def __init__(self,
                 fixed_step: float = None,
                 adaptive: bool = True,
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
                    
        if not adaptive and fixed_step is None:
            raise ValueError("Since you choose not to use adaptive stepping, you must provide a value for the fixed step size.")
        
        self.fixed_step = fixed_step
        self.first_step = first_step
        self.adaptive = adaptive
        self.min_step = min_step
        self.max_step = max_step
        self.nsteps_max = nsteps_max
        self.rtol = rtol
        self.atol = atol
        self.linear_solver = linear_solver
        self.linear_solver_opts = linear_solver_opts or {}
        self.max_jacobian_refresh = max_jacobian_refresh
        self.verbose = verbose
        self.progress_interval_in_time = progress_interval_in_time
        self.export_interval = export_interval
        self.export_prefix = export_prefix
        self.newton_nmax = newton_nmax
        self.auto_check_sparsity  = auto_check_sparsity
        self.sparse_threshold = sparse_threshold
        self.sparsity_ratio_limit = sparsity_ratio_limit
        self.initial_step_safety = initial_step_safety

        self.newton_failed = False

        self._export_counter = 0
        self._sparsity_checked = False
        self._jacobian_is_constant = False
        self._mass_matrix_is_sparse = False
        self._mass_matrix_is_constant = False
        self._mass_matrix_is_identity = False
        self._using_sparse_algebra = False

        self._Id = None
        self._jacobianF = None
        self._mass_matrix = None
        self._use_built_in_python_list = False

    def _print_verbose(self, message):
        """Print a message if verbose mode is enabled.
        Args:
            message (str): Message to display.
        """
        if self.verbose:
            print(message)

    def _print_pyodys_error_message(self, message):
        """Print a PyOdys error message regardless of verbosity.
        Args:
            message (str): Error message to display.
        """
        print(message)

    def _export(self, times, solutions: np.ndarray):
        """Export simulation results to a CSV file.

        Args:
            times (np.ndarray): Array of time points.
            solutions (np.ndarray): Array of states corresponding to `times`.

        Notes:
            Files are named using the format ``<prefix>_<counter>.csv``.
        """
        if self.export_prefix is None:
            return
        self._export_counter += 1
        filename = f"{self.export_prefix}_{self._export_counter:05d}.csv"
        dirpath = os.path.dirname(filename)
        if dirpath:  # only create if there is a directory component
            os.makedirs(dirpath, exist_ok=True)
        n_vars = solutions.shape[1] if solutions.ndim > 1 else 1
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["t"] + [f"u{i}" for i in range(n_vars)]
            writer.writerow(header)
            for t, u in zip(times, solutions):
                row = [t] + (u.tolist() if n_vars > 1 else [u])
                writer.writerow(row)
        self._print_verbose(f"Exported {len(times)} steps to {filename}")

    def _classify_matrix(self, A, name: str, n_eq: int, auto_check: bool, sparsity_ratio_limit: float, force_sparse: bool = False, use_sparse_if_sparse_provided = True):
        """
        Convert a matrix to sparse/dense and decide algebra mode.

        Parameters
        ----------
        A : array_like or sparse
            Input matrix (Jacobian, Mass, or Global).
        name : str
            Identifier for verbose logging.
        n_eq : int
            Matrix dimension.
        auto_check : bool
            Whether to check sparsity for dense inputs.
        sparsity_ratio_limit : float
            Threshold for sparse/dense decision.
        force_sparse : bool, default False
            If True, always return sparse (CSC).

        Returns
        -------
        A_proc : ndarray or csc_matrix
            Processed matrix in appropriate format.
        is_sparse : bool
            Whether the returned matrix is sparse.
        density : float
            Fraction of nonzeros.
        """
        if isspmatrix(A):
            A_csc = A.tocsc()
            density = A_csc.nnz / (n_eq * n_eq)
            if use_sparse_if_sparse_provided:
                return (A_csc, True, density)
            else:
                return (A_csc, density<self.sparsity_ratio_limit, density)

        # Dense input
        A = np.asarray(A, dtype=float)
        density = np.count_nonzero(A) / (n_eq * n_eq)

        if force_sparse or (auto_check and density < sparsity_ratio_limit):
            A_csc = csc_matrix(A)
            if self.verbose:
                self._print_verbose(f"{name}: dense --> sparse conversion, density={density:.3e}")
            return (A_csc, True, density)
        else:
            if self.verbose:
                self._print_verbose(f"{name}: dense kept as dense, density={density:.3e}")
            return (A, False, density)

    def _detect_sparsity_and_store_jacobian_if_constant(self, F: ODEProblem, tn: float, U_np: np.ndarray):
        """
        Detect whether the Jacobian provided by F is sparse or dense.
        Initialize identity matrices accordingly.
        If the Jacobian is constant, store it once for reuse.

        Parameters
        ----------
        F : ODEProblem
            The ODEProblem object, which must provide a `jacobian_at(t, u)` method.
        tn : float
            The current time.
        U_np : np.ndarray
            The current state vector.
        """

        J = F.jacobian_at(tn, U_np)
        n_eq = F.number_of_equations

        J_proc, self._using_sparse_algebra, density = self._classify_matrix(
            J, "Jacobian", n_eq, auto_check=self.auto_check_sparsity,
            sparsity_ratio_limit=self.sparsity_ratio_limit
        )

        if F.jacobian_is_constant:
            self._jacobianF = J_proc

        if self._using_sparse_algebra:
            self._Id = identity(n_eq, format="csc")
        else:
            self._Id = np.eye(n_eq)

        if self.verbose:
            self._print_verbose(f"Jacobian detected: sparse={self._using_sparse_algebra}, density={density:.3e}")

        if self._jacobian_is_constant and self.verbose:
            self._print_verbose("Jacobian marked as constant → will be reused across all steps.")

    def _detect_sparsity_and_store_mass_matrix_if_constant(self, F: ODEProblem, tn: float, U_np: np.ndarray):
        """
        Detect whether the Mass Matrix provided by F is sparse or dense.
        Initialize identity matrices accordingly.
        If the Mass Matrix is constant, store it once for reuse.

        Parameters
        ----------
        F : ODEProblem
            The ODEProblem object, which must provide a `jacobian_at(t, u)` method.
        tn : float
            The current time.
        U_np : np.ndarray
            The current state vector.
        """
        M = F.mass_matrix_at(tn, U_np)
        n_eq = F.number_of_equations

        M_proc, self._using_sparse_algebra, density = self._classify_matrix(
            M, "Mass Matrix", n_eq, auto_check=self.auto_check_sparsity,
            sparsity_ratio_limit=self.sparsity_ratio_limit
        )

        if F.mass_matrix_is_constant:
            self._mass_matrix = M_proc

        if self.verbose:
            self._print_verbose(f"Mass Matrix detected: sparse={self._using_sparse_algebra}, density={density:.3e}")

        if F.mass_matrix_is_constant and self.verbose:
            self._print_verbose("Mass Matrix marked as constant → will be reused across all steps.")

    def _detect_global_sparsity(self, F: ODEProblem, tn: float, U_np: np.ndarray, h: float, a_ii: float):
        """
        Detect whether the global matrix M - h*a_ii*J is sparse.
        If so, use sparse algebra for the solver. Only cache constant matrices.

        Parameters
        ----------
        F : ODEProblem
            The ODE system.
        tn : float
            Current time.
        U_np : np.ndarray
            Current solution.
        h : float
            Step size.
        a_ii : float
            A coefficient from Butcher tableau / BDF.
        """
        n_eq = F.number_of_equations
        J = F.jacobian_at(tn, U_np)
        M = F.mass_matrix_at(tn, U_np)

        # Always convert both to sparse before forming global A
        J_sparse = J if isspmatrix(J) else csc_matrix(J)
        M_sparse = M if isspmatrix(M) else csc_matrix(M)

        A = M_sparse - h * a_ii * J_sparse
        _, self._using_sparse_algebra, density = self._classify_matrix(
            A, "Global system matrix", n_eq, auto_check=self.auto_check_sparsity,
            sparsity_ratio_limit=self.sparsity_ratio_limit, use_sparse_if_sparse_provided=False
        )

        if F.mass_matrix_is_identity:
            self._mass_matrix_is_identity = True
            if self._using_sparse_algebra:
                self._Id = identity(n_eq, format="csc")
            else:
                self._Id = np.eye(n_eq)

        if F.jacobian_is_constant:
            self._jacobianF = J_sparse if self._using_sparse_algebra else J_sparse.toarray()
            self._jacobian_is_constant = True
        else:
            self._jacobian_is_constant = False

        if F.mass_matrix_is_constant:
            self._mass_matrix = M_sparse if self._using_sparse_algebra else M_sparse.toarray()
            self._mass_matrix_is_constant = True 
        else:
            self._mass_matrix_is_constant = False

        if self.verbose:
            mode = "SPARSE" if self._using_sparse_algebra else "DENSE"
            self._print_verbose(f"Global sparsity check: USING {mode} ALGEBRA, density={density:.3e}")

    def _compute_matrix(self, type:str, F:ODEProblem, t:float, U:np.ndarray):
        if type == "jacobian":
            Jf = F.jacobian_at(t, U)
            if self._using_sparse_algebra:
                if isspmatrix(Jf):
                    self._jacobianF = Jf.tocsc()
                else:
                    self._jacobianF = csc_matrix(Jf)
            else:
                self._jacobianF = np.asarray(Jf, dtype=float)
        elif type == "mass_matrix":
            M = F.mass_matrix_at(t, U)
            if self._using_sparse_algebra:
                if isspmatrix(M):
                    self._mass_matrix = M.tocsc()
                else:
                    self._mass_matrix = csc_matrix(M)
            else:
                self._mass_matrix = np.asarray(M, dtype=float)
        else:
            raise ValueError("`type` should be either `jacobian` or `mass_matrix`.")

    def _mass_matrix_jacobian_contract_d(self, ode_problem: ODEProblem, t: float, y: np.ndarray, d: np.ndarray, eps=None, central=True):
        """
        Compute (∂M/∂y)(t,y) ⋅ d, i.e. the directional derivative of the mass 
        matrix with respect to the state vector y, applied to direction d.

        Let M(t,y) ∈ R^{NxN} be the mass matrix. Its Jacobian with respect to y is a
        rank-3 tensor:
            (∂M/∂y)_{i,j,k} = ∂M_{i,j} / ∂y_k.

        For a direction vector d ∈ R^N, the contraction
            (∂M/∂y)(t,y) ⋅ d = Σ_k d_k (∂M/∂y_k)(t,y)
        is an NxN matrix, representing the sensitivity of M in the direction d.

        This function approximates (∂M/∂y)(t,y) ⋅ d using finite differences:

            central difference (default):
                ( M(t, y+ε d) - M(t, y-ε d) ) / (2ε)

            forward difference:
                ( M(t, y+ε d) - M(t, y) ) / ε

        Parameters
        ----------
        ode_problem : ODEProblem
            Problem definition providing mass_matrix_at(t, y).
        t : float
            Time parameter.
        y : ndarray, shape (N,)
            State vector.
        d : ndarray, shape (N,)
            Direction vector.
        eps : float, optional
            Step size for finite differences (chosen adaptively if None).
        central : bool, default True
            Use central difference (2 evaluations, more accurate) or forward 
            difference (1 evaluation).

        Returns
        -------
        Jd : ndarray, shape (N, N)
            The matrix (∂M/∂y)(t,y) ⋅ d.

        Example
        -------
        Suppose M(t,y) = diag(y), with y ∈ R^3.
          M(y) = [[y1,  0,  0],
                  [ 0, y2, 0],
                  [ 0,  0, y3]].

        Then ∂M/∂y1 = [[1,0,0],[0,0,0],[0,0,0]],
             ∂M/∂y2 = [[0,0,0],[0,1,0],[0,0,0]],
             ∂M/∂y3 = [[0,0,0],[0,0,0],[0,0,1]].

        For a direction d = (d1,d2,d3), the contraction is
            (∂M/∂y) ⋅ d = d1 ∂M/∂y1 + d2 ∂M/∂y2 + d3 ∂M/∂y3
                         = diag(d1, d2, d3).
        """
        y = np.asarray(y, dtype=float)
        d = np.asarray(d, dtype=float)

        if eps is None:
            eps = np.sqrt(np.finfo(float).eps) * (1 + np.linalg.norm(y)) / (np.linalg.norm(d) + 1e-20)

        if central:
            M_plus  = ode_problem.mass_matrix_at(t, y + eps * d)
            M_minus = ode_problem.mass_matrix_at(t, y - eps * d)
            return (M_plus - M_minus) / (2 * eps)
        else:
            M0 = ode_problem.mass_matrix_at(t, y)
            M1 = ode_problem.mass_matrix_at(t, y + eps * d)
            return (M1 - M0) / eps
        

    def _estimate_initial_step(self, f:ODEProblem, t0: float, y0: np.ndarray, error_estimator_order: float ):
        """
        Estimates an initial step size for an ODE solver using the Normalized Derivative Approach.

        Parameters
        ----------
        f : ODEProblem
            The function defining the ODE, f(t, y) = dy/dt.
        t0 : float
            The initial time.
        y0 : ndarray
            The initial state vector.
        error_estimator_order : float
            The order of the ODE solver's local error method (e.g., 4 for RK45).

        Returns
        -------
        float
            The estimated initial step size.
        """
        dy_dt = f.evaluate_at(t0, y0)
        denominator = np.maximum(self.atol, np.abs(y0) * self.rtol)
        dy_dt_norm = np.linalg.norm(dy_dt / (denominator + utils._DEFAULT_ZERO_TOL))

        if dy_dt_norm < 1e-10:
            return 1e-6 
        
        h0 = min(self.max_step, max(self.min_step, self.initial_step_safety * (1.0 / dy_dt_norm) ** (1.0 / (error_estimator_order + 1))))
        return h0

    @abstractmethod
    def solve(self, ode_problem : ODEProblem):
        """
        Solve the ODE problem.

        This is an abstract method that must be implemented by all subclasses.

        Args:
            ode_problem (ODEProblem): The ODE problem to solve.
        """
        raise NotImplementedError("Any subclass must implement this method.")
    
    def _build_linear_solver(self, A: np.ndarray, solver_name: str = "lu", **kwargs):
        """
        Build a linear solver callable for the given matrix A.
        Automatically chooses sparse or dense solver based on self._using_sparse_algebra.

        Parameters
        ----------
        A : ndarray or sparse matrix
            The system matrix to solve.
        solver_name : str
            Solver type: "lu", "cg", "gmres", "svd".
        kwargs : dict
            Extra solver-specific arguments (e.g., tol, restart).

        Returns
        -------
        solver : callable
            Callable `solver(rhs)` returning a 1D NumPy array.
        """

        solver_name = solver_name.lower()

        n = A.shape[0]
        # user-level defaults stored on the solver (set at init)
        defaults = {
            "lu": {},
            "cg": {"atol": 1e-10, "rtol": 1e-8, "maxiter": None},
            "gmres": {"atol": 1e-10, "rtol": 1e-8, "restart": min(n, 50), "maxiter": None},
            "svd": {"full_matrices": False},
        }
        opts = {**defaults.get(solver_name, {}), **kwargs}

        # --- LU solver ---
        if solver_name == "lu":
            if self._using_sparse_algebra:
                from scipy.sparse.linalg import splu
                LU = splu(A)
                return lambda rhs: LU.solve(rhs, **opts) #.ravel()
            else:
                from scipy.linalg import lu_factor, lu_solve
                LU_piv = lu_factor(A)
                return lambda rhs: lu_solve(LU_piv, rhs, **opts)

        # --- Conjugate Gradient ---
        elif solver_name == "cg":
            from scipy.sparse.linalg import cg
            return lambda rhs: cg(A, rhs, **opts)[0]

        # --- GMRES ---
        elif solver_name == "gmres":
            from scipy.sparse.linalg import gmres
            def gmres_solver(rhs):
                x, info = gmres(A, rhs, **opts)
                if info != 0:
                    raise RuntimeError(f"GMRES did not converge. Info={info}")
                return x
            return gmres_solver

        # --- SVD (dense only) ---
        elif solver_name == "svd":
            if self._using_sparse_algebra:
                raise NotImplementedError("SVD solver not implemented for sparse matrices")
            import numpy.linalg as la
            U, s, Vh = np.linalg.svd(A, full_matrices=False)
            s_inv = np.diag(1.0 / s)
            def svd_solver(rhs):
                return (Vh.T @ (s_inv @ (U.T @ rhs))) #.ravel()
            return svd_solver

        else:
            raise ValueError(f"Unknown linear solver '{solver_name}'. Choose from 'lu', 'cg', 'gmres', 'svd'.")


    # ------------------------------------------------------------------
    # Internal helper: automatic scaling of max number of adaptive steps
    # ------------------------------------------------------------------
    def _auto_nsteps_max(self, ode_problem: ODEProblem) -> int:
        """
        Automatically estimate a safe upper bound for the maximum number
        of adaptive steps based on system size, time span, and memory.
        """

        try:
            N = len(ode_problem.initial_state)
        except Exception:
            N = 1

        try:
            t_span = abs(ode_problem.t_final - ode_problem.t_init)
        except Exception:
            t_span = 1.0
        from psutil import virtual_memory
        mem_available = virtual_memory().available

        bytes_per_state = 8 * N  # 64-bit floats
        safety_factor = 2.0  # accounts for work arrays, overhead

        # Memory-limited maximum
        nsteps_mem_limited = int(mem_available / (safety_factor * bytes_per_state))
        nsteps_mem_limited = max(1000, nsteps_mem_limited)  # enforce minimal cap

        # Time-span scaling (logarithmic to avoid overgrowth)
        nsteps_time_scaled = int(1e4 * np.log10(1.0 + 1e3 * t_span) + 1e5)

        # Final combined limit
        nsteps_auto = min(nsteps_mem_limited, nsteps_time_scaled)
        nsteps_auto = int(np.clip(nsteps_auto, 10_000, 2_000_000))

        return nsteps_auto

    def _set_none_params(self, ode_problem: ODEProblem, time_span_length: float):
        # Auto-tune nsteps_max if not provided
        if self.adaptive and self.nsteps_max is None:
            auto_nmax = self._auto_nsteps_max(ode_problem)
            self.nsteps_max = auto_nmax
            if self.verbose:
                self._print_verbose(f"Auto-selected nsteps_max = {auto_nmax}")
        
        # set auto_check_sparsity based on the number of equations, and the sparse threshold
        if ode_problem.number_of_equations < self.sparse_threshold:
            self.auto_check_sparsity = False

        if self.adaptive:
            if self.min_step is None:
                from ..utils.pyodys_utils import _DEFAULT_ZERO_TOL
                scale_factor = 1e-12 
                self.min_step = min(time_span_length, max(_DEFAULT_ZERO_TOL, scale_factor * time_span_length) ) 
                if self.verbose:
                    self._print_verbose(f"Setting adaptive minimum step size {self.min_step}.")

            if self.max_step is None:
                self.max_step = time_span_length 
                if self.verbose:
                    self._print_verbose(f"Setting adaptive maximum step size to {self.max_step}.")

    def _wrms_norm(self, delta, u, atol=1e-12, rtol=1e-6, type: str="mean"):
        """Weighted Root Mean Square norm.
            Args:
                - delta (float) : Newton update vector
                - u (numpy.ndarray) : current Newton iterate
                - type (str): Either mean or max

            Returns:
                - The Weighted Root Mean Square norm
        """
        scale = atol + rtol * np.abs(u)
        if type=="max":
            return np.max(np.abs(delta / scale) )
        elif type=="mean":
            return np.sqrt(np.mean((delta / scale) ** 2))
        else:
            raise ValueError("The type must be either 'mean' or 'max'")

    def _check_step_size(
            self,
            U_approx : np.ndarray, 
            U_pred : np.ndarray, 
            step_size : float,
            min_step : float, 
            max_step : float, 
            current_time : float, 
            t_final : float, 
            atol : float,
            rtol : float,
            error_estimator_order: int,
            previous_step_failed: bool,
            print_verbose: callable = print
        ):
        """Validate and adapt the time step size based on error estimates.
        Args:
            U_approx (np.ndarray): Computed solution.
            U_pred (np.ndarray): Predictor solution.
            step_size (float): Current time step.
            rtol (float): Target relative error.
            p (int): Order of the RK method.
            min_step (float): Minimum allowed time step.
            max_step (float): Maximum allowed time step.
            current_time (float): Current simulation time.
            t_final (float): Final simulation time.
        Returns:
            tuple:
                - float: New time step size.
                - bool: True if current step is accepted, False otherwise.
        """
        err = self._wrms_norm(U_approx - U_pred, U_approx, atol, rtol)
        step_accepted = err <= 1
        safety = 0.9
        min_factor = 0.2
        max_factor = 5.0
        # avoid division by zero:
        err = max(err, 1e-16)
        factor = safety * (1.0 / err) ** (1.0 / (error_estimator_order+1))
        factor = max(min_factor, min(max_factor, factor))
        new_step_size = step_size * factor

        if new_step_size < min_step:
            print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} < min step size {min_step:.4e}. Using min step size."
            )
            new_step_size = min_step
        elif new_step_size > max_step:
            print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} > max step size {max_step:.4e}. Using max step size."
            )
            new_step_size = max_step

        if step_accepted:
            new_time = current_time + step_size
            if new_time + new_step_size > t_final:
                new_step_size = max(t_final - new_time, 0.0)

        if previous_step_failed and abs(new_step_size - step_size) < utils._DEFAULT_ZERO_TOL:
            raise utils.PyodysError(
                                f"Adaptive solver cannot advance the solution: current step {step_size:.2e} "
                                f"could not satisfy error tolerance (err = {err:.2e}). "
                                f"Consider adjusting min_step, atol, or rtol."
                            )
        return new_step_size, step_accepted
