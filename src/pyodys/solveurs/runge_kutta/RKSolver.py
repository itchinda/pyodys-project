from ...systemes.ODEProblem import ODEProblem
from .ButcherTableau import ButcherTableau
import numpy as np
from typing import Union
from scipy.linalg import lu_factor, lu_solve, LinAlgError
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix, identity, isspmatrix
import csv
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class PyOdysError(RuntimeError):
    """Exception raised when PyOdys fails to solve a problem."""
    def __init__(self, message):
        super().__init__(message)

# class UnsupportedSchemeError(RuntimeError):
#     """Exception raised when a General Implicit RK scheme is passed by a user"""
#     def __init__(self, message):
#         super().__init__(message)

def wrms_norm(delta, u, atol=1e-12, rtol=1e-6):
    """
    Weighted Root Mean Square norm.
    delta : Newton update vector
    u : current Newton iterate
    """
    scale = atol + rtol * np.abs(u)
    return np.sqrt(np.mean((delta / scale) ** 2))


class RKSolver(object):
    """
    A Runge-Kutta solver for ordinary differential equations (ODEs) that
    can handle both explicit and implicit schemes. It supports adaptive
    time-stepping and efficient handling of sparse or dense Jacobians.

    Jacobian handling policy
    ------------------------
    - If ``ode_problem.jacobian_is_constant = True``:
      The Jacobian is computed once at initialization and reused for all
      Newton iterations, stages, and steps.
    
    - If ``ode_problem.jacobian_is_constant = False``:
      The Jacobian is recomputed once at the beginning of every time step.
      If a Newton iteration fails to converge, the Jacobian is refreshed
      and the Newton process is retried (up to ``max_jacobian_refresh`` times).

    This policy ensures robustness by keeping the Jacobian current, while
    avoiding unnecessary recomputations unless Newton convergence problems are detected.

    Other features
    --------------
    - Explicit and implicit RK schemes
    - Adaptive and fixed time stepping
    - Automatic detection of sparse Jacobians
    - CSV export of intermediate results
    - Verbosity and progress reporting

    """
    def __init__(self,
                 method: Union[ButcherTableau, str] = None,
                 first_step: float = None,
                 adaptive: bool = False,
                 min_step: float = None, 
                 max_step: float = None,
                 nsteps_max: int = 1000000,
                 adaptive_rtol: float = None,
                 max_jacobian_refresh: int = 1,
                 newton_atol: float = 1e-10,
                 newton_rtol: float = 1e-8,
                 newton_nmax: int = 10,
                 verbose: bool = True,
                 progress_interval_in_time: int = None,
                 export_interval: int = None,
                 export_prefix: str = None,
                 auto_check_sparsity: bool = True,
                 sparse_threshold: int = 20,
                 sparsity_ration_limit: float = 0.2):
        """Initialize a Runge-Kutta solver with a Butcher tableau.

        Args:
            method (ButcherTableau | str): The name of the scheme (e.g. erk1, erk2, erk4, sdirk1, sdirk2, sdirk4, esdirk6, dopri5, etc.), or the Butcher tableau that defines the specific RK method to use. For adaptive methods, the tableau must provide an embedded solution.
            first_step (float): The initial time step (Deltat). This is the fixed step size for non-adaptive mode.
            adaptive (bool, optional): If True, the solver will adjust the step size to meet the specified adaptive_rtol. Defaults to False.
            min_step (float, optional): The minimum allowed step size when using adaptive time-stepping. Required if adaptive is True.
            max_step (float, optional): The maximum allowed step size when using adaptive time-stepping. Required if adaptive is True.
            adaptive_rtol (float, optional): The target relative error for adaptive step-size control. Required if adaptive is True.
            max_jacobian_refresh (int, optional): The maximum number of times to recompute and re-factorize the Jacobian within a single time step's implicit stages before giving up.
            newton_atol (float, optional): The absolute tolerance for Newton's method convergence.
            newton_rtol (float, optional): The relative tolerance for Newton's method convergence.
            newton_nmax (int, optional): The maximum number of iterations for Newton's method per Jacobian refresh.
            verbose (bool, optional): If True, the solver prints progress and debug information.
            progress_interval_in_time (float, optional): The time interval at which to print progress updates.
            export_interval (int, optional): The number of steps between data exports to CSV files. If None, no export is performed.
            export_prefix (str, optional): The file path prefix for exported CSV files.
            auto_check_sparsity  (bool, optional): A flag to enable or disable automatic sparsity detection.
            sparse_threshold (int, optional): The number of equations a system must have to trigger the automatic sparsity detection.

        Raises:
            TypeError: If configuration is inconsistent.
            ValueError: If required arguments are missing.
        """
        if isinstance(method, str):
            if method not in ButcherTableau.available_schemes():
                raise ValueError(f'There is not available schemes with name {method}. Here is the list of available schemes: {ButcherTableau.available_schemes()}')
            self.butcher_tableau=ButcherTableau.from_name(method)
        elif isinstance(method, ButcherTableau):
            self.butcher_tableau = method
        else:
            raise TypeError("method must be an object of class ButcherTableau,or the method name(str). Please provide either ButchherTableau, or the name of the method.")
                    
        if first_step==None:
            raise ValueError("You must specify the time step or the initial time step (if you choose adaptive time stepping).")
        if adaptive and (min_step==None or max_step==None):
            raise TypeError("Since you choose adaptive time stepping, you must specify the minimal and maximal time steps.")
        if adaptive and adaptive_rtol == None:
            raise TypeError("Since you choose adaptive time stepping, you must specify the the target relative error.")
        if self.butcher_tableau.is_implicit and not self.butcher_tableau.is_diagonally_implicit:
            raise PyOdysError("General Implicit Runge-Kutta schemes are not currently supported. Consider using a Diagonally Implicit scheme instead.")
        
        self.first_step = first_step
        self.adaptive = adaptive
        self.min_step = min_step
        self.max_step = max_step
        self.nsteps_max = nsteps_max
        self.adaptive_rtol = adaptive_rtol
        self.max_jacobian_refresh = max_jacobian_refresh
        self.verbose = verbose
        self.progress_interval_in_time = progress_interval_in_time
        self.export_interval = export_interval
        self.export_prefix = export_prefix
        self.newton_atol = newton_atol
        self.newton_rtol = newton_rtol
        self.newton_nmax = newton_nmax
        self.auto_check_sparsity  = auto_check_sparsity 
        self.sparse_threshold = sparse_threshold
        self.sparsity_ration_limit = sparsity_ration_limit
        self.newton_failed = False

        self._export_counter = 0
        self._with_prediction = self.butcher_tableau.with_prediction
        self._sparsity_checked = False
        self._jacobian_is_sparse = None
        self._mass_matrix_is_sparse = None
        self._using_sparse_algebra = False

        self._I_dense = None
        self._I_sparse = None
        self._jacobianF_csr = None
        self._jacobianF_dense = None
        self._mass_matrix = None
        self._use_built_in_python_list = False

        self._is_dirk = self.butcher_tableau.is_diagonally_implicit
        self._is_erk = self.butcher_tableau.is_explicit
        self._is_irk = self.butcher_tableau.is_implicit
        self._is_sdirk = self.butcher_tableau.is_sdirk
        self._is_esdirk = self.butcher_tableau.is_esdirk

        self._jacobian_constant_and_sdirk_and_fixed_step_size = False
        self._static_linear_sparse_solver = None
        self._static_linear_dense_solver = None


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

    def _detect_sparsity(self, F: ODEProblem, tn: float, U_np: np.ndarray):
        """
        Detect whether the Jacobian provided by F is sparse or dense.
        Initialize identity matrices accordingly.
        If the Jacobian is constant, store it once for reuse.

        Args:
            self: The instance of the RKSolver class.
            F (ODEProblem): The ODEProblem object. This object is expected to have a method jacobian_at(t, u) that returns the Jacobian matrix of the ODE system.
            tn (float): The current time, used to evaluate the Jacobian at a specific point in the simulation.
            U_np (numpy.ndarray): The current state vector, used to evaluate the Jacobian at a specific state.
        """
        J = F.jacobian_at(tn, U_np)
        n_eq = J.shape[0]

        if isspmatrix(J):
            # Always sparse if user returned sparse
            self._jacobian_is_sparse = True
            self._jacobianF_csr = J.tocsr()
            self._I_sparse = identity(n_eq, format="csr")
            if self.verbose:
                density = self._jacobianF_csr.nnz / (n_eq * n_eq)
                print(f"Sparse Jacobian returned by user: size={n_eq}x{n_eq}, density={density:.3e}")

        else:
            # Dense returned
            if self.auto_check_sparsity :
                # check sparsity fraction
                nz_frac = np.count_nonzero(J) / (n_eq * n_eq)
                if nz_frac < self.sparsity_ration_limit:  # threshold: <20% nonzeros
                    self._jacobian_is_sparse = True
                    self._jacobianF_csr = csr_matrix(J)
                    self._I_sparse = identity(n_eq, format="csr")
                    if self.verbose:
                        print(f"Dense Jacobian treated as sparse: size={n_eq}x{n_eq}, density={nz_frac:.3e}")
                else:
                    self._jacobian_is_sparse = False
                    self._jacobianF_dense = np.asarray(J, dtype=float)
                    self._I_dense = np.eye(n_eq)
                    if self.verbose:
                        print(f"Dense Jacobian treated as dense: size={n_eq}x{n_eq}")
            else:
                # Always use dense
                self._jacobian_is_sparse = False
                self._jacobianF_dense = np.asarray(J, dtype=float)
                self._I_dense = np.eye(n_eq)
                if self.verbose:
                    print(f"Dense Jacobian returned by user, using dense: size={n_eq}x{n_eq}")

        if F.jacobian_is_constant and self.verbose:
            print("Jacobian marked as constant → will be reused across all steps.")
        elif self.verbose:
            print("Jacobian marked as variable → will be recomputed at each stage refresh.")

    def _detect_global_sparsity(self, F: ODEProblem, tn: float, U_np: np.ndarray, h: float):
        non_zero_diagonals = np.diag(self.butcher_tableau.A).nonzero()[0]
        if non_zero_diagonals.size > 0:
            first_implicit_stage_index = non_zero_diagonals[0]
            a_ii = self.butcher_tableau.A[first_implicit_stage_index, first_implicit_stage_index]
        else:
            a_ii = 0.0

        n_eq = F.number_of_equations

        
        Jf = F.jacobian_at(tn, U_np)
        J_sparse = Jf.tocsr() if isspmatrix(Jf) else csr_matrix(Jf)
    
        M = F.mass_matrix_at(tn, U_np)
        M_sparse = M.tocsr() if isspmatrix(M) else csr_matrix(M)

        # Build a test matrix A for the sparsity check
        A_sparse = M_sparse - h * a_ii * J_sparse

        # Decide based on the final system matrix's sparsity
        nz_frac = A_sparse.nnz / (n_eq * n_eq)

        self._using_sparse_algebra = False # Default to dense
        if nz_frac < self.sparsity_ration_limit:
            self._using_sparse_algebra = True

        if self._using_sparse_algebra:
            # Cache matrices in sparse format
            self._J_cached = J_sparse
            self._M_cached = M_sparse
            self._I_sparse = identity(n_eq, format="csr")
            if self.verbose:
                print(f"Global sparsity check: USING SPARSE ALGEBRA. Density = {nz_frac:.3e}")
        else:
            # Cache matrices in dense format
            self._J_cached = J_sparse.toarray()
            self._M_cached = M_sparse.toarray()
            self._I_dense = np.eye(n_eq)
            if self.verbose:
                print(f"Global sparsity check: USING DENSE ALGEBRA. Density = {nz_frac:.3e}")

        # Handle constant matrices as a final step
        if F.jacobian_is_constant:
            self.J = self._J_cached
            self.J_is_constant = True
        if F.mass_matrix_is_constant:
            self.M = self._M_cached
            self.M_is_constant = True

    def _get_mass_matrix_jacobian_times_d(ode_problem: ODEProblem, t: float, y: np.ndarray, d: np.ndarray, eps=None, central=True):
        """
        Compute J(t,y) @ d = directional derivative of the mass matrix of a given ODE problem, wrt y in direction d.

        Parameters
        ----------
        ode_problem (ODEProblem): The ODEProblem object. This object is expected to have a method mass_matrix_at(t, u) that returns 
                                  the NxN Mass matrix of the ODE/ADE system.
        t : float
            Time parameter
        y : array_like, shape (N,)
            State vector
        d : array_like, shape (N,)
            Direction vector
        eps : float, optional
            Step size for finite difference (chosen automatically if None)
        central : bool, default True
            Use central difference (more accurate, 2 evaluations) or forward difference (1 evaluation)

        Returns
        -------
        Jd : ndarray, shape (N, N)
            Matrix equal to J(t,y) @ d
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
    
    def _perform_single_rk_step(
            self, F: ODEProblem, tn: float, delta_t: float, U_np: np.ndarray
        ):
        """Perform one Runge-Kutta step based on the Butcher tableau.

        Jacobian handling
        -----------------
            - If ``F.jacobian_is_constant = True``:
              The Jacobian is computed once at initialization and reused for all
              implicit stages and Newton iterations.
        
            - If ``F.jacobian_is_constant = False``:
              The Jacobian is computed once at the start of this time step.
              If Newton iterations fail to converge for an implicit stage, the
              Jacobian is refreshed (up to ``max_jacobian_refresh`` times) and
              the Newton iterations are retried.
        
        Args:
            self: The instance of the RKSolver class.
            F: The ODEProblem object representing the ODE system.
            tn (float): The current time point, tn.
            delta_t (float): The time step size, Δt.
            U_np (numpy.ndarray): The state vector at the current time, u(tn).

        Returns:
            tuple: A tuple containing:
                - U_n_plus_1 (numpy.ndarray): The computed solution at the next time step, u(tn+1).
                - U_pred (numpy.ndarray): An approximation of the solution used for error estimation in adaptive stepping. If the method doesn't support an embedded solution, this will be a zero array.
                - newton_not_happy (bool): A flag that is True if Newton's method failed to converge for any of the implicit stages, and False otherwise.
    
        """

        n_stages = self.butcher_tableau.n_stages
        n_eq = len(U_np)

        a = self.butcher_tableau.A
        c = self.butcher_tableau.C

        if self._with_prediction:
            b = self.butcher_tableau.B[0, :]
            d = self.butcher_tableau.B[1, :]
        else:
            b = self.butcher_tableau.B
            d = np.zeros_like(b)

        newton_not_happy = False
        U_chap = np.zeros((n_eq, n_stages))
        deltat_x_value_f = np.zeros((n_eq, n_stages))

        newton_nmax = self.newton_nmax
        newton_atol = self.newton_atol
        newton_rtol = self.newton_rtol

        U_n = np.copy(U_np)
        U_pred = np.copy(U_np) if self._with_prediction else np.zeros_like(U_np)

        # --- Explicit scheme → no Newton iterations
        if self._is_erk:
            for k in range(n_stages):
                tn_k = tn + c[k] * delta_t
                U_chap[:, k] = U_np + np.sum(a[k, :k] * deltat_x_value_f[:, :k], axis=1)
                deltat_x_value_f[:, k] = delta_t * F.evaluate_at(tn_k, U_chap[:, k])
                U_n += b[k] * deltat_x_value_f[:, k]
                if self._with_prediction:
                    U_pred += d[k] * deltat_x_value_f[:, k]
            return U_n, U_pred, newton_not_happy
        

        # --- Implicit scheme → Newton iterations
        # compute the jacobian only once! and update only if Newton fails

        tn_k = tn

        first_implicit_stage_idx = np.diag(a).nonzero()[0][0]
        gamma = a[first_implicit_stage_idx, first_implicit_stage_idx]
        for refresh_attempt in range(self.max_jacobian_refresh + 1):
            
            if not self._jacobian_constant_and_sdirk_and_fixed_step_size:
                Jf = F.jacobian_at(tn_k, U_n)
                if self._jacobian_is_sparse:
                    if isspmatrix(Jf):
                        self._jacobianF_csr = Jf.tocsr()
                    else:
                        self._jacobianF_csr = csr_matrix(Jf)
                    assert self._I_sparse is not None
                else:
                    self._jacobianF_dense = np.asarray(Jf, dtype=float)
                    assert self._I_dense is not None
            
            solver = None
            if self._jacobian_constant_and_sdirk_and_fixed_step_size:
                if self._jacobian_is_sparse:
                    solver_sdirk = self._static_linear_sparse_solver
                else:
                    solver_sdirk = self._static_linear_dense_solver
            elif (self._is_sdirk or self._is_esdirk):
                delta_t_x_gamma = delta_t*gamma 
                if self._jacobian_is_sparse:
                    A_sparse = self._I_sparse - delta_t_x_gamma * self._jacobianF_csr
                    LU = splu(A_sparse.tocsc())
                    solver_sdirk = LU.solve
                else:
                    A_dense = self._I_dense - delta_t_x_gamma * self._jacobianF_dense
                    LU_piv = lu_factor(A_dense)
                    solver_sdirk = lambda rhs: lu_solve(LU_piv, rhs)

        
            newton_failed = False
            for k in range(n_stages):
                U_chap_k = U_np + np.sum(a[k, :k] * deltat_x_value_f[:, :k], axis=1)
                if a[k, k] == 0.0:  # no implicit coupling # USEFULL FOR ESDIRK SCHEMES!
                    tn_k = tn + c[k] * delta_t
                    U_chap[:, k] = U_chap_k
                    deltat_x_value_f[:, k] = delta_t * F.evaluate_at(tn_k, U_chap[:, k])
                    U_n += b[k] * deltat_x_value_f[:, k]
                    if self._with_prediction:
                        U_pred += d[k] * deltat_x_value_f[:, k]
                    continue

                # --- Implicit stage: Newton solve
                tn_k = tn + c[k] * delta_t
                delta_t_x_akk = delta_t * a[k, k]
                U_newton = np.copy(U_chap_k)

                if self._is_esdirk or self._is_sdirk:  # use the pre-factored solver
                    linear_solver = solver_sdirk
                else:
                    #assemble system matrix and factorize
                    if self._jacobian_is_sparse:
                        A_sparse = self._I_sparse - delta_t_x_akk * self._jacobianF_csr
                        LU = splu(A_sparse.tocsc())
                        linear_solver = LU.solve
                    else:
                        A_dense = self._I_dense - delta_t_x_akk * self._jacobianF_dense
                        LU_piv = lu_factor(A_dense)
                        linear_solver = lambda rhs: lu_solve(LU_piv, rhs)

                # Newton iterations
                newton_succeeded = False
                for iteration_newton in range(newton_nmax):
                    # The original problem to folve is : Find K_k s.t. K_k - h* f(t_nk, u_chap_k + akk*K_k) = 0. We can set X = u_chap_k + akk*K_k to end with
                    #                                               X - u_chap_k - h*akk*f(t_nk, X) = 0, X being the new unknown.
                    residu = None
                    if F.mass_matrix_is_identity:
                        residu = U_newton - (U_chap_k + delta_t_x_akk * F.evaluate_at(tn_k, U_newton)) 
                    else:
                        if F.mass_matrix_is_constant:
                            residu = self._mass_matrix*(U_newton - U_chap_k) - delta_t_x_akk * F.evaluate_at(tn_k, U_newton) 
                        else:
                            residu = F.mass_matrix_at(tn_k, U_newton)*(U_newton - U_chap_k) - delta_t_x_akk * F.evaluate_at(tn_k, U_newton) 
                    try:
                        delta = linear_solver(residu)
                    except (LinAlgError, RuntimeError, ValueError) as e:
                        self._print_verbose(f"Linear solve failed at stage {k}: {e}")
                        newton_not_happy = True
                        return U_n, U_pred, newton_not_happy
                    U_newton -= delta
                    if wrms_norm(delta, U_newton, newton_atol, newton_rtol) < 1.0:
                        newton_succeeded = True
                        break

                if not newton_succeeded:
                    newton_failed = True
                    break   # Refresh the jacobian

                # store stage result
                U_chap[:, k] = U_newton
                deltat_x_value_f[:, k] = (U_newton - U_chap_k) / a[k,k] # = delta_t * F.evaluate_at(tn_k, U_newton). Be smart, avoid calling f again!!
                U_n += b[k] * deltat_x_value_f[:, k]
                if self._with_prediction:
                    U_pred += d[k] * deltat_x_value_f[:, k]

            if not newton_failed:
                newton_not_happy = False
                return U_n, U_pred, newton_not_happy
            
        self._print_verbose(f"Newton failed {k} after Jacobian refreshes")
        newton_not_happy = True
        return U_n, U_pred, newton_not_happy
            
    def solve(self, ode_problem: ODEProblem):
        """
        Solves an ODE system using either fixed or adaptive time stepping.

        Args:
            ode_problem (ODEProblem): ODE system to integrate.

        Returns:
            tuple: A tuple containing arrays of time points and corresponding states.
                   Returns `None` if `export_interval` is set.

        Raises:
            PyOdysError: If the solver encounters a fatal error, such as repeated Newton failures.
        """
        if not ode_problem.mass_matrix_is_identity:
            raise ValueError("Ptoblem with non identity mass matrix not currently supported.")
    
        if self._jacobian_is_sparse is None:
            self._detect_sparsity(ode_problem, ode_problem.t_init, ode_problem.initial_state)

        # compute the jacobian only once if constant!
        if ode_problem.jacobian_is_constant:
            if self._jacobian_is_sparse == True and self._jacobianF_csr is None:
                J = ode_problem.jacobian_at(ode_problem.t_init, ode_problem.initial_state)
                if isspmatrix(J):
                    self._jacobianF_csr = J.tocsr()
                else:
                    self._jacobianF_csr = csr_matrix(J)
            elif not self._jacobian_is_sparse and self._jacobianF_dense is None:
                self._jacobianF_dense = ode_problem.jacobian_at(ode_problem.t_init, ode_problem.initial_state)

        if not self.adaptive:
            return self._solve_with_fixed_step_size(ode_problem, self.first_step)

        U_courant = np.copy(ode_problem.initial_state)
        current_time = ode_problem.t_init
        nsteps_max = self.nsteps_max
        if self.progress_interval_in_time == None:
            self.progress_interval_in_time = (ode_problem.t_final - ode_problem.t_init) / 100.0

        next_progress_in_time = ode_problem.t_init + self.progress_interval_in_time

        try:
            if self.export_interval:
                times = np.empty(self.export_interval+1, dtype=float)
                solutions = np.empty((self.export_interval+1, len(ode_problem.initial_state)), dtype=float)
            else :
                times = np.empty(nsteps_max+1, dtype=float)
                solutions = np.empty((nsteps_max+1, len(ode_problem.initial_state)), dtype=float)
            times[0] = ode_problem.t_init
            solutions[0,:] = np.copy(U_courant)
            self._print_verbose("Successfully pre-allocated memory for the solution array.")
        except MemoryError:
            message = (
                "Memory allocation failed. Using the built-in python list. May slow down the solver performance..."
                "Consider enabling the export mode by setting 'export_interval' and 'export_prefix in the solver's"
                "constructor for a better performance."
            )
            self._print_verbose(message)
            self._use_built_in_python_list = True
            times = [ode_problem.t_init]
            solutions = [ode_problem.initial_state]

        t_final = ode_problem.t_final
        step_size = self.first_step
        embedded_order = self.butcher_tableau.embedded_order if self._with_prediction else self.butcher_tableau.order

        number_of_time_steps = 0
        newton_failure_count = 0
        max_newton_failures = 10

        k = 0
        while current_time < t_final and number_of_time_steps < nsteps_max:
            # tronquer pour ne pas dépasser t_final
            step_size = min(step_size, t_final - current_time)

            U_n_plus_1, U_pred, newton_not_happy = \
                self._perform_single_rk_step(
                    ode_problem, current_time, step_size, U_courant
                )

            if newton_not_happy:
                newton_failure_count += 1
                self._print_verbose(
                    f"Newton failed at t = {current_time:.4f}. "
                    f"Reducing step size and retrying. Failure count: {newton_failure_count}"
                )
                step_size = max(step_size / 2.0, self.min_step)
                if newton_failure_count >= max_newton_failures:
                    message = (
                        f"Maximum consecutive Newton failures ({max_newton_failures}) reached. "
                        "Stopping the simulation."
                    )
                    self._print_verbose(message)
                    self.newton_failed = True
                    raise PyOdysError(message)
                continue  # retry immediately at same time

            # succès Newton !
            # computing a predicted  solution if the selected solver does not provide an embedded solution
            if not self._with_prediction:
                try:
                    U_pred = self._perform_richardson_step(
                        ode_problem, current_time, step_size, U_courant
                    )
                except ValueError as e:
                    # Handle the specific error from Richardson extrapolation
                    self._print_verbose(f"Richardson extrapolation failed: {e}. Retrying with smaller step.")
                    step_size = max(step_size / 2.0, self.min_step)
                    newton_failure_count += 1
                    continue


            newton_failure_count = 0

            new_step_size, step_accepted = self._check_step_size(
                U_n_plus_1, U_pred, step_size, self.adaptive_rtol, embedded_order,
                self.min_step, self.max_step, current_time, t_final
            )

            if step_accepted:
                U_courant = U_n_plus_1
                current_time += step_size
                if self._use_built_in_python_list:
                    times.append(current_time)
                    solutions.append(np.copy(U_courant))
                else:
                    times[k+1] = current_time
                    solutions[k+1, :] = np.copy(U_courant)
                step_size = new_step_size
                number_of_time_steps += 1
                k += 1

                if self.export_interval and k == self.export_interval:
                    self._export(times[:k], solutions[:k, :])
                    times[0] = times[k]
                    solutions[0, :] = np.copy(solutions[k, :])
                    k = 0

                if current_time >= next_progress_in_time:
                    self._print_verbose(
                        f"Time step #{number_of_time_steps} completed. Current time: {current_time:.4f}"
                    )
                    next_progress_in_time += self.progress_interval_in_time

            else:
                self._print_verbose(
                    f"Time step {step_size:.4e} rejected at t = {current_time:.4f}. "
                    f"Retrying with step size: {new_step_size:.4e}"
                )
                step_size = new_step_size

        if self.export_interval:
            if k > 0:
                self._export(times[:k+1], solutions[:k+1, :])
            print(f"Simulation completed. The results have been saved to {self.export_prefix}*.csv")
            return None
        if current_time < t_final - 1e-12:
            warnings.warn(f"\nWarnings: I stopped the simulation at t = {current_time}, due to the limited number of adaptive steps allowed.\n"
                          f"Number of steps completed: {self.nsteps_max}.\n"
                          f"Time after {self.nsteps_max} steps:  {current_time}\n"
                          f"Expected Final Time: {t_final}.")
        else:
            self._print_verbose(
                f"The total number of time steps required to reach t_final = {t_final} is {number_of_time_steps}."
            )
        return np.array(times[:number_of_time_steps+1], dtype=float), np.array(solutions[:number_of_time_steps+1], dtype=float)

    def _perform_richardson_step(self, F: ODEProblem, tn: float, delta_t: float, U_np: np.ndarray):
        """Performs Richardson extrapolation for schemes without embedded estimators."""

        # First half-step
        U_half_step, _, newton_not_happy = \
            self._perform_single_rk_step(
                F, tn, delta_t / 2.0, U_np
            )
        if newton_not_happy:
            raise ValueError("Newton failed during the first Richardson half-step.")

        # Second half-step
        U_pred, _, newton_not_happy = \
            self._perform_single_rk_step(
                F, tn + delta_t / 2.0, delta_t / 2.0, U_half_step
            )
        if newton_not_happy:
            raise ValueError("Newton failed during the second Richardson half-step.")

        return U_pred

    def _check_step_size(self, U_approx : np.ndarray, U_pred : np.ndarray, step_size : float, adaptive_rtol : float,
                          embedded_order : int, min_step : float, max_step : float, current_time : float, t_final : float):
        """Validate and adapt the time step size based on error estimates.

        Args:
            U_approx (np.ndarray): Computed solution.
            U_pred (np.ndarray): Predictor solution.
            step_size (float): Current time step.
            adaptive_rtol (float): Target relative error.
            embedded_order (int): Order of the embedded RK method.
            min_step (float): Minimum allowed time step.
            max_step (float): Maximum allowed time step.
            current_time (float): Current simulation time.
            t_final (float): Final simulation time.

        Returns:
            tuple:
                - float: New time step size.
                - bool: True if current step is accepted, False otherwise.
        """
        alpha = 0.1
        beta = 0.9
        eps = 1e-12
        err = np.linalg.norm((U_approx - U_pred) / (np.abs(U_pred)+1e-12), ord=2) 
        step_accepted = err < (1 + alpha) * adaptive_rtol

        if step_accepted and err > (1-alpha)*adaptive_rtol:
            new_step_size = step_size
        else:
            new_step_size = beta * step_size * (adaptive_rtol / max(err, eps)) ** (1.0 / (embedded_order + 1))

        if new_step_size < min_step:
            self._print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} < min step size {min_step:.4e}. Using min step size."
            )
            new_step_size = min_step
        elif new_step_size > max_step:
            self._print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} > max step size {max_step:.4e}. Using max step size."
            )
            new_step_size = max_step

        if step_accepted:
            new_time = current_time + step_size
            if new_time + new_step_size > t_final:
                new_step_size = max(t_final - new_time, 0.0)
        return new_step_size, step_accepted

    def _solve_with_fixed_step_size(self, ode_problem: ODEProblem, step_size):
        """Solve an ODE system with a fixed time step.

        Args:
            ode_problem (ODEProblem): ODE system to integrate.
            step_size (float): Fixed time step size.

        Returns:
            tuple:
                - np.ndarray: Array of time points.
                - np.ndarray: Array of corresponding states.

        Raises:
            PyOdysError: If Newton iterations fail.
        """
        U_courant = np.copy(ode_problem.initial_state)
        current_time = ode_problem.t_init
        max_number_of_time_steps = int((ode_problem.t_final - ode_problem.t_init) / step_size)


        if self.progress_interval_in_time == None:
            self.progress_interval_in_time = np.max([(float(max_number_of_time_steps) / 100.0)*self.first_step, 1.0])

        next_progress_in_time = ode_problem.t_init + self.progress_interval_in_time
        if self.export_interval:
            times = np.empty(self.export_interval+1, dtype=float)
            solutions = np.empty((self.export_interval+1, len(ode_problem.initial_state)), dtype=float)
        else :
            times = np.empty(max_number_of_time_steps+1, dtype=float)
            solutions = np.empty((max_number_of_time_steps+1, len(ode_problem.initial_state)), dtype=float)
        times[0] = ode_problem.t_init
        solutions[0,:] = np.copy(U_courant)

        k = 0

        if (self._is_sdirk or self._is_esdirk) and ode_problem.jacobian_is_constant:
            self._jacobian_constant_and_sdirk_and_fixed_step_size = True
            first_implicit_stage_idx = np.diag(self.butcher_tableau.A).nonzero()[0][0]
            gamma = self.butcher_tableau.A[first_implicit_stage_idx, first_implicit_stage_idx]
            delta_t_x_gamma = step_size*gamma 

            if self._jacobian_is_sparse:
                A_sparse = self._I_sparse - delta_t_x_gamma * self._jacobianF_csr
                LU = splu(A_sparse.tocsc())
                self._static_linear_sparse_solver = LU.solve
            else:
                A_dense = self._I_dense - delta_t_x_gamma * self._jacobianF_dense
                LU_piv = lu_factor(A_dense)
                self._static_linear_dense_solver = lambda rhs: lu_solve(LU_piv, rhs)

        for n in range(max_number_of_time_steps):
            U_n_plus_1, _, newton_not_happy = self._perform_single_rk_step(
                ode_problem, current_time, step_size, U_courant
            )
            if newton_not_happy:
                self.newton_failed = True
                message = f"Newton failed at time step {n+1} even after Jacobian refresh."
                self._print_verbose(message)
                raise PyOdysError(message)

            U_courant = U_n_plus_1
            current_time += step_size
            times[k+1] = current_time
            solutions[k+1, :] = U_courant
            k += 1

            if self.export_interval and k == self.export_interval:
                self._export(times[:k], solutions[:k, :])
                times[0] = times[k]
                solutions[0, :] = solutions[k, :]
                k = 0

            if current_time >= next_progress_in_time:
                self._print_verbose(
                    f"Time step #{n+1} completed. Current time: {current_time:.4f}"
                )
                next_progress_in_time += self.progress_interval_in_time
        
        if self.export_interval:
            if k > 0:
                self._export(times[:k+1], solutions[:k+1, :])
            print(f"Simulation completed. The results have been saved to {self.export_prefix}*.csv")
            return None

        return np.array(times), np.array(solutions)

    def resoud(self, ode_problem: ODEProblem):
        """Alias for :meth:`solve`.

        Args:
            ode_problem (ODEProblem): ODE system to solve.

        Returns:
            tuple:
                - np.ndarray: Array of time points.
                - np.ndarray: Array of states.
        """
        return self.solve(ode_problem)

    