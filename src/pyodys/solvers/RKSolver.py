from ..ode.ODEProblem import ODEProblem
from ..schemes.rk.RKScheme import RKScheme
from .SolverBase import SolverBase
from ..utils.pyodys_utils import PyodysError
import numpy as np
from typing import Union, Callable
from scipy.linalg import lu_factor, lu_solve, LinAlgError
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix, isspmatrix
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

class RKSolver(SolverBase):
    """
    Runge-Kutta solver for ordinary differential equations (ODEs).

    This solver supports explicit, diagonally implicit (DIRK/SDIRK),
    and embedded Runge-Kutta schemes. It provides both fixed and adaptive
    time-stepping, with efficient handling of dense and sparse Jacobians.

    Key Features
    ------------
    - Explicit and implicit RK schemes (ERK, DIRK, SDIRK, ESDIRK).
    - Adaptive and fixed step sizes.
    - Automatic detection of sparse vs. dense Jacobians.
    - Efficient linear solvers with LU factorization or sparse splu.
    - Optional CSV export of intermediate solutions.
    - Verbose output and progress reporting.

    Parameters
    ----------
    method : RKScheme or str
        Runge-Kutta scheme to use. Either an RKScheme instance or the name of
        a built-in scheme (e.g. "erk4", "sdirk2", "esdirk6", "dopri5").
        For adaptive methods, the scheme must provide an embedded solution.
    fixed_step : float, optional
        The fixed step size to use for non-adaptive solvers.
        Required if `adaptive` is False.
    adaptive : bool, default True
        If True, enable adaptive step size control.
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
    Raises
    ------
    TypeError
        If `method` is not str or RKScheme.
    ValueError
        If solver configuration is inconsistent.

    Jacobian Handling
    -----------------
    - If ``ode_problem.jacobian_is_constant`` is True:
        The Jacobian is computed once at initialization and reused for all
        Newton iterations, stages, and steps.
    - If ``ode_problem.jacobian_is_constant`` is False:
        The Jacobian is recomputed once per time step. If Newton fails to
        converge, the Jacobian is refreshed and Newton is retried, up to
        ``max_jacobian_refresh`` times.

    Notes
    -----
    - General fully implicit RK schemes are not supported (only diagonally-implicit).
    - Problems with non-identity mass matrices are not yet supported.
    """
    def __init__(self,
                 method: Union[RKScheme, str] = None,
                 fixed_step: float = None,
                 adaptive: bool = True,
                 first_step: float = None,
                 min_step: float = None, 
                 max_step: float = None,
                 nsteps_max: int = None,
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
            initial_step_safety = initial_step_safety)
        
        if not adaptive and fixed_step is None:
            raise ValueError("Since you choose not to use adaptive stepping, you must provide a value for the fixed step size.")

        # Resolve RK scheme
        if isinstance(method, str):
            available = "\n".join(RKScheme.available_schemes())
            if method not in RKScheme.available_schemes():
                raise ValueError(
                    f"There is no available scheme with name {method}. "
                    f"Here is the list of available schemes:\n{available}"
                )
            self.butcher_tableau = RKScheme.from_name(method)
        elif isinstance(method, RKScheme):
            self.butcher_tableau = method
        else:
            raise TypeError("method must be an RKScheme instance or a scheme name string.")

        if self.butcher_tableau.is_implicit and not self.butcher_tableau.is_diagonally_implicit:
            raise PyodysError("General implicit RK schemes are not supported. Use a diagonally-implicit variant.")

        if not adaptive and fixed_step is None:
            raise ValueError("When adaptive=False you must provide fixed_step.")

        self._with_prediction = self.butcher_tableau.with_prediction
        self.rk_scheme_is_dirk = self.butcher_tableau.is_diagonally_implicit
        self.rk_scheme_is_erk = self.butcher_tableau.is_explicit
        self.rk_scheme_is_irk = self.butcher_tableau.is_implicit
        self.rk_scheme_is_sdirk = self.butcher_tableau.is_sdirk
        self.rk_scheme_is_esdirk = self.butcher_tableau.is_esdirk

        self._linear_sparse_solver = None
        self._linear_dense_solver = None

        self._error_estimator_order = self.butcher_tableau.embedded_order if self._with_prediction else self.butcher_tableau.order

        if self.rk_scheme_is_irk:
            first_implicit_stage_idx = np.diag(self.butcher_tableau.A).nonzero()[0][0]
            self.gamma_sdirk  = self.butcher_tableau.A[first_implicit_stage_idx, first_implicit_stage_idx] # The diagonal entry
        else:
            self.gamma_sdirk  = None

        self._work_U_chap = None
        self._work_deltat_x_value_f = None
        self._work_U_pred = None
        self._work_U_n = None
        self._work_U_newton = None
        self._nb_equations = None

    # -------------------------
    # Helper to create prefactored SDIRK solver for constant Jacobian
    # -------------------------
    def _build_prefactored_dirk_solver(self, delta_t, akk):
        """
        Build a pre-factored linear solver for DIRK/SDIRK/ESDIRK schemes.

        The solver is reused across implicit stages if the Jacobian and mass matrix are constant.

        Parameters
        ----------
        delta_t : float
            Current time step size.
        akk : float
            Diagonal coefficient of the Butcher tableau for the current stage.

        Returns
        -------
        solver : Callable
            Function that solves `A x = rhs` for given `rhs`, using the pre-factored LU or sparse solver.
        """

        # Ensure matrices are in the expected format
        if self._using_sparse_algebra:
            if not isspmatrix(self._jacobianF):
                self._jacobianF = csc_matrix(self._jacobianF)
            if not self._mass_matrix_is_identity and not isspmatrix(self._mass_matrix):
                self._mass_matrix = csc_matrix(self._mass_matrix)
        else:
            if isspmatrix(self._jacobianF):
                self._jacobianF = self._jacobianF.toarray()
            if not self._mass_matrix_is_identity and isspmatrix(self._mass_matrix):
                self._mass_matrix = self._mass_matrix.toarray()

        delta_t_x_gamma = delta_t * akk

        # Build matrix for the linear solve
        if self._mass_matrix_is_identity:
            A = self._Id - delta_t_x_gamma * self._jacobianF
        else:
            A = self._mass_matrix - delta_t_x_gamma * self._jacobianF

        if isinstance(self.linear_solver, str):
            solver = self._build_linear_solver(A, self.linear_solver, **self.linear_solver_opts)
        else:
            solver = lambda rhs: np.asarray(self.linear_solver(A, rhs, **self.linear_solver_opts))

        # Store for reuse
        if self._using_sparse_algebra:
            self._linear_sparse_solver = solver
        else:
            self._linear_dense_solver = solver

        return solver

    def _build_prefactored_erk_solver(self):
        """
        Build a linear solver for explicit RK stages with a non-identity mass matrix.

        Only required for ERK schemes when mass matrix is not the identity.

        Returns
        -------
        solver : Callable or None
            Solver function for mass matrix solves, or None if mass matrix is identity.
        """

        if self._mass_matrix_is_identity:
            return None  # explicit → no linear solve needed

        # Ensure mass matrix format
        if self._using_sparse_algebra and not isspmatrix(self._mass_matrix):
            self._mass_matrix = csc_matrix(self._mass_matrix)
        elif not self._using_sparse_algebra and isspmatrix(self._mass_matrix):
            self._mass_matrix = self._mass_matrix.toarray()

        A = self._mass_matrix

        if isinstance(self.linear_solver, str):
            solver = self._build_linear_solver(A, self.linear_solver, **self.linear_solver_opts)
        else:
            solver = lambda rhs: np.asarray(self.linear_solver(A, rhs **self.linear_solver_opts))

        # Store for reuse
        if self._using_sparse_algebra:
            self._linear_sparse_solver = solver
        else:
            self._linear_dense_solver = solver

        return solver
                
    def _perform_single_explicit_stage(self, k:int, F:ODEProblem, tn:float, delta_t:float, U_np:np.ndarray, U_n:np.ndarray,
                                       U_pred:np.ndarray, U_chap:np.ndarray, deltat_x_value_f:np.ndarray, 
                                       a:np.ndarray, b:np.ndarray, c:np.ndarray, d:np.ndarray):
        """
        Perform a single stage of an explicit RK scheme.

        Parameters
        ----------
        k : int
            Stage index.
        F : ODEProblem
            The ODE system.
        tn : float
            Current time.
        delta_t : float
            Time step size.
        U_np : np.ndarray
            Current solution u_n.
        U_n : np.ndarray
            Solution accumulator for u_{n+1}.
        U_pred : np.ndarray
            Prediction array for embedded solution (error estimator).
        U_chap : np.ndarray
            Stage solution storage.
        deltat_x_value_f : np.ndarray
            Stage derivatives storage (Δt * f evaluations).
        a, b, c, d : np.ndarray
            Butcher tableau coefficients.

        Returns
        -------
        U_n : np.ndarray
            Updated solution at the end of the stage.
        U_pred : np.ndarray
            Updated prediction for embedded solution.
        """

        tn_k = tn + c[k] * delta_t
        if k == 0:
            U_chap[:, k] = U_np
        else:
            U_chap[:, k] = U_np + deltat_x_value_f[:, :k] @ a[k, :k]
        if self._mass_matrix_is_identity:
            deltat_x_value_f[:, k] = delta_t * F.evaluate_at(tn_k, U_chap[:, k])
        else:
            if self._mass_matrix_is_constant:
                if self._using_sparse_algebra and self._linear_sparse_solver is None:
                    self._linear_sparse_solver = self._build_prefactored_erk_solver()
                elif not self._using_sparse_algebra and self._linear_dense_solver is None:
                    self._linear_dense_solver = self._build_prefactored_erk_solver()
            else:
                # We must recompute the mass matrix!
                self._compute_matrix("mass_matrix", F, tn_k, U_chap[:, k])
                if self._using_sparse_algebra:
                    self._linear_sparse_solver = self._build_prefactored_erk_solver()
                else:
                    self._linear_dense_solver = self._build_prefactored_erk_solver()
            if self._using_sparse_algebra:
                deltat_x_value_f[:, k] = delta_t * self._linear_sparse_solver(F.evaluate_at(tn_k, U_chap[:, k]))
            else:
                deltat_x_value_f[:, k] = delta_t * self._linear_dense_solver(F.evaluate_at(tn_k, U_chap[:, k]))
        U_n += b[k] * deltat_x_value_f[:, k]
        if self._with_prediction:
            U_pred += d[k] * deltat_x_value_f[:, k]
        return U_n, U_pred

    def _perform_single_rk_step(
            self, F: ODEProblem, tn: float, delta_t: float, U_np: np.ndarray
        ):
        """
        Perform one full Runge-Kutta step (all stages) for given time and step size.

        Handles explicit or DIRK/SDIRK/ESDIRK schemes, including Newton iterations
        for implicit stages and optional error estimation.

        Parameters
        ----------
        F : ODEProblem
            ODE system to integrate.
        tn : float
            Current time.
        delta_t : float
            Time step size.
        U_np : np.ndarray
            Current solution vector u_n.

        Returns
        -------
        U_n_plus_1 : np.ndarray
            Solution at the next time step.
        U_pred : np.ndarray
            Predicted solution for error estimation (embedded or Richardson extrapolation).
        newton_not_happy : bool
            True if Newton iteration failed for any implicit stage.
        """

        n_stages = self.butcher_tableau.n_stages
        a = self.butcher_tableau.A
        c = self.butcher_tableau.C

        if self._with_prediction:
            b = self.butcher_tableau.B[0, :]
            d = self.butcher_tableau.B[1, :]
        else:
            b = self.butcher_tableau.B
            d = np.zeros_like(b)

        newton_not_happy = False

        U_chap = self._work_U_chap
        deltat_x_value_f = self._work_deltat_x_value_f
        U_pred = self._work_U_pred
        U_n = np.copy(U_np)

        if self._with_prediction:
            U_pred[:] = U_np 

        # --- Explicit scheme → no Newton iterations
        if self.rk_scheme_is_erk:
            for k in range(n_stages):
                U_n, U_pred = self._perform_single_explicit_stage(k, F, tn, delta_t, U_np, U_n, U_pred, U_chap, deltat_x_value_f, a, b, c, d)
            return U_n, U_pred, newton_not_happy

        # --- Implicit(DIRK-like) scheme → Newton iterations
        tn_k = tn
        newton_nmax = self.newton_nmax
        U_newton = self._work_U_newton

        for refresh_attempt in range(self.max_jacobian_refresh + 1):
            # compute or reuse jacobian for this attempt
            ###########################################################################333
            if not F.jacobian_is_constant:
                self._compute_matrix("jacobian", F, tn_k, U_n)
            
            if not (self._mass_matrix_is_identity and self._mass_matrix_is_constant):
                self._compute_matrix("mass_matrix", F, tn_k, U_n)

            # if jacobian is constant AND scheme is sdirk/esdirk, we can use prefactored solver
            if F.jacobian_is_constant and (F.mass_matrix_is_constant or F.mass_matrix_is_identity) and (self.rk_scheme_is_esdirk or self.rk_scheme_is_sdirk) :
                if self._using_sparse_algebra:
                    solver_sdirk = self._linear_sparse_solver
                else:
                    solver_sdirk = self._linear_dense_solver
            elif (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk):
                solver_sdirk = self._build_prefactored_dirk_solver(delta_t, self.gamma_sdirk)
            
            ################################################################################

            newton_failed = False
            for k in range(n_stages):
                if k == 0:
                    U_chap_k = U_np
                else:
                    U_chap_k = U_np + deltat_x_value_f[:, :k] @ a[k, :k]

                if a[k, k] == 0.0:  # No implicit coupling # USEFULL FOR ESDIRK SCHEMES. 
                    U_n, U_pred = self._perform_single_explicit_stage(k, F, tn, delta_t, U_np, U_n, U_pred, U_chap, deltat_x_value_f, a, b, c, d)
                    continue

                # --- Implicit stage: Newton solve
                tn_k = tn + c[k] * delta_t
                delta_t_x_akk = delta_t * a[k, k]
                U_newton[:] = U_chap_k
                
                ################################################# Uncomment this block if if you want to recompute the Jacobian every stage 
                # if not F.jacobian_is_constant:
                #     self._compute_matrix("jacobian", F, tn_k, U_n)

                # if not (self._mass_matrix_is_identity and self._mass_matrix_is_constant):
                #     self._compute_matrix("mass_matrix", F, tn_k, U_n)

                # # if jacobian is constant AND scheme is sdirk/esdirk, we can use prefactored solver
                # if F.jacobian_is_constant and (F.mass_matrix_is_constant or F.mass_matrix_is_identity) and (self.rk_scheme_is_esdirk or self.rk_scheme_is_sdirk) :
                #     if self._using_sparse_algebra:
                #         solver_sdirk = self._linear_sparse_solver
                #     else:
                #         solver_sdirk = self._linear_dense_solver
                # elif (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk):
                #     solver_sdirk = self._build_prefactored_dirk_solver(delta_t, self.gamma_sdirk)
                #######################################################################

                if self.rk_scheme_is_esdirk or self.rk_scheme_is_sdirk:  # use the pre-factored solver
                    linear_solver = solver_sdirk
                else:
                    linear_solver = self._build_prefactored_dirk_solver(delta_t, a[k, k])

                # Newton loop
                newton_succeeded = False
                for iteration_newton in range(newton_nmax):
                    # The original problem to solve is : Find K_k s.t. M K_k - f(t_nk, u_chap_k + h*akk*K_k) = 0. We can set X = u_chap_k + h*akk*K_k to end with
                    #                                               M(X - u_chap_k) - h*akk*f(t_nk, X) = 0, X being the new unknown.
                    if self._mass_matrix_is_identity:
                        Mu_uchap = U_newton - U_chap_k
                    else:
                        if not self._mass_matrix_is_constant:
                            self._compute_matrix("mass_matrix", F, tn_k, U_newton )
                        Mu_uchap = self._mass_matrix @ (U_newton - U_chap_k)

                    residu = Mu_uchap - delta_t_x_akk * F.evaluate_at(tn_k, U_newton)
                    try:
                        delta = linear_solver(residu)
                    except (LinAlgError, RuntimeError, ValueError) as e:
                        super()._print_verbose(f"Linear solve failed at stage {k}: {e}")
                        newton_not_happy = True
                        return U_n, U_pred, newton_not_happy
                    U_newton -= delta

                    eta=1
                    if self._wrms_norm(delta, U_newton, eta*self.atol, eta*self.rtol) < 1.0:
                        newton_succeeded = True
                        break

                if not newton_succeeded:
                    newton_failed = True
                    break   # refresh jacobian and retry outer loop

                # store stage result
                delta_U = U_newton - U_chap_k
                U_chap[:, k] = U_newton
                deltat_x_value_f[:, k] = delta_U / a[k, k]   # equals h * K_k
                U_n += b[k] * deltat_x_value_f[:, k]
                if self._with_prediction:
                    U_pred += d[k] * deltat_x_value_f[:, k]

            if not newton_failed:
                newton_not_happy = False
                return U_n, U_pred, newton_not_happy
            
        super()._print_verbose(f"Newton failed {k} after Jacobian refreshes")
        return U_n, U_pred, True
            
    def solve(self, ode_problem: ODEProblem):
        """
        Solves an ODE system using either fixed or adaptive time stepping.

        Args:
            ode_problem (ODEProblem): ODE system to integrate.

        Returns:
            tuple: A tuple containing arrays of time points and corresponding states.
                   Returns `None` if `export_interval` is set.

        Raises:
            PyodysError: If the solver encounters a fatal error, such as repeated Newton failures.
        """
        
        # Reinitialize optimization infos
        self._jacobianF = None
        self._mass_matrix = None
        self._Id = None
        self._using_sparse_algebra = False
        self._jacobian_is_constant = ode_problem.jacobian_is_constant
        self._mass_matrix_is_constant = ode_problem.mass_matrix_is_constant
        self._mass_matrix_is_identity = ode_problem.mass_matrix_is_identity
        self._linear_sparse_solver = None
        self._linear_dense_solver = None

        self._nb_equations = ode_problem.number_of_equations
        n_stages = self.butcher_tableau.n_stages
        n_eq = self._nb_equations
        
        dT = abs(ode_problem.t_final - ode_problem.t_init)
        if self.adaptive and self.min_step is not None and self.min_step > dT:
            raise ValueError (f"Min step must be less than the time interval size. You provided min_step = "
                              f"{self.min_step}, but the length of the time span is {abs(ode_problem.t_final - ode_problem.t_init)}")

        self._set_none_params(ode_problem = ode_problem, time_span_length = dT)

        if self.auto_check_sparsity:
            if self._mass_matrix_is_identity and not self.rk_scheme_is_erk:
                self._detect_sparsity_and_store_jacobian_if_constant(ode_problem, ode_problem.t_init, ode_problem.initial_state)
            elif not self._mass_matrix_is_identity and self.rk_scheme_is_erk:
                self._detect_sparsity_and_store_mass_matrix_if_constant(ode_problem, ode_problem.t_init, ode_problem.initial_state)
            elif not self._mass_matrix_is_identity and self.rk_scheme_is_dirk:
                self._detect_global_sparsity(ode_problem, ode_problem.t_init, ode_problem.initial_state, h = dT/100.0, a_ii = self.gamma_sdirk)

        # Precompute identity for future use. Will probably be removed for optinmization. I don't actually need to store this.
        if self._Id is None and self._mass_matrix_is_identity and self.rk_scheme_is_dirk:
            if self._using_sparse_algebra:
                from scipy.sparse import identity
                self._Id = identity(n_eq, format="csc")
            else:
                self._Id = np.identity(n_eq, dtype=float)
        
        # Precompute constant jacobian for future use
        if ode_problem.jacobian_is_constant and self.rk_scheme_is_irk and self._jacobianF is None:
            self._compute_matrix("jacobian", ode_problem, ode_problem.t_init, ode_problem.initial_state)

        # Precompute constant mass matrix for future use
        if not self._mass_matrix_is_identity:
            if self._mass_matrix_is_constant and self._mass_matrix is None:
                self._compute_matrix("mass_matrix", ode_problem, ode_problem.t_init, ode_problem.initial_state)

        U_courant = np.copy(ode_problem.initial_state)
        self._work_U_chap = np.zeros((n_eq, n_stages))
        self._work_deltat_x_value_f = np.zeros((n_eq, n_stages))
        self._work_U_pred = np.empty_like(U_courant)
        self._work_U_n = np.empty_like(U_courant)
        self._work_U_newton = np.empty_like(U_courant)

        #################### FIXED STEP SOLVE ##########################
        if not self.adaptive:
            return self._solve_with_fixed_step_size(ode_problem)
        ################################################################

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
            solutions[0,:] = U_courant
            super()._print_verbose("Successfully pre-allocated memory for the solution array.")
        except MemoryError:
            message = (
                "Memory allocation failed. Falling back to Python lists (slower). "
                "Enable export mode for better performance."
            )
            super()._print_verbose(message)
            self._use_built_in_python_list = True
            times = [ode_problem.t_init]
            solutions = [ode_problem.initial_state]

        t_final = ode_problem.t_final
        t_init =  ode_problem.t_init

        if self.first_step is None:
            self.first_step = super()._estimate_initial_step(
                f = ode_problem, 
                t0 = t_init, 
                y0 = ode_problem.initial_state, 
                error_estimator_order=self._error_estimator_order
            )

        step_size = self.first_step
        number_of_time_steps = 0
        newton_failure_count = 0
        max_newton_failures = 10
        k = 0

        previous_step_failed = False

        while current_time < t_final and number_of_time_steps < nsteps_max:
            step_size = min(step_size, t_final - current_time)

            if (self._mass_matrix_is_constant or self._mass_matrix_is_identity) and (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk) and ode_problem.jacobian_is_constant: # and self.adaptive:
                self._build_prefactored_dirk_solver(step_size, self.gamma_sdirk)

            U_n_plus_1, U_pred, newton_not_happy = self._perform_single_rk_step(
                    ode_problem, current_time, step_size, U_courant
                )

            if newton_not_happy:
                newton_failure_count += 1
                super()._print_verbose(
                    f"Newton failed at t = {current_time:.6e}. Reducing step size and retrying. "
                    f"Failure count: {newton_failure_count}"
                )
                step_size = max(step_size / 2.0, self.min_step)
                if newton_failure_count >= max_newton_failures:
                    message = (
                        f"Maximum consecutive Newton failures ({max_newton_failures}) reached. "
                        "Stopping."
                    )
                    super()._print_verbose(message)
                    self.newton_failed = True
                    raise PyodysError(message)
                continue  # retry immediately at same time

            # succes case:
            # if no embedded estimator, build prediction using Richardson estrapolation
            if not self._with_prediction:
                try:
                    U_pred = U_n_plus_1
                    U_n_plus_1 = self._perform_richardson_step(
                        ode_problem, current_time, step_size, U_courant, U_pred
                    )
                except ValueError as e:
                    super()._print_verbose(f"Richardson extrapolation failed: {e}. Retrying with smaller step.")
                    step_size = max(step_size / 2.0, self.min_step)
                    newton_failure_count += 1
                    continue

            newton_failure_count = 0

            new_step_size, step_accepted = self._check_step_size(
                U_approx = U_n_plus_1, 
                U_pred = U_pred, 
                step_size = step_size,
                min_step = self.min_step, 
                max_step = self.max_step, 
                current_time = current_time, 
                t_final = t_final, 
                atol = self.atol, 
                rtol = self.rtol, 
                error_estimator_order=self._error_estimator_order,
                previous_step_failed=previous_step_failed,
                print_verbose=super()._print_verbose
            )

            if step_accepted:
                U_courant = U_n_plus_1
                current_time += step_size
                if self._use_built_in_python_list:
                    times.append(current_time)
                    solutions.append(U_courant)
                else:
                    times[k+1] = current_time
                    solutions[k+1, :] = U_courant
                step_size = new_step_size
                number_of_time_steps += 1
                k += 1
                previous_step_failed = False

                if self.export_interval and k == self.export_interval:
                    super()._export(times[:k], solutions[:k, :])
                    times[0] = times[k]
                    solutions[0, :] = solutions[k, :]
                    k = 0

                if current_time >= next_progress_in_time:
                    super()._print_verbose(
                        f"Time step #{number_of_time_steps} completed. Current time: {current_time:.4f}"
                    )
                    next_progress_in_time += self.progress_interval_in_time

            else:
                super()._print_verbose(
                    f"Step {step_size:.4e} rejected at t = {current_time:.4f}. "
                    f"Retrying with step size: {new_step_size:.4e}"
                )
                step_size = new_step_size
                previous_step_failed = True

        if self.export_interval:
            if k > 0:
                super()._export(times[:k + 1], solutions[:k + 1, :])
            print(f"Simulation completed. The results have been saved to {self.export_prefix}*.csv")
            return None
        
        if current_time < t_final - 1e-12:
            warnings.warn(
                f"Stopped at t = {current_time} after {number_of_time_steps} steps (limit reached). "
                f"Requested final time: {t_final}."
            )
        else:
            super()._print_verbose(f"Reached t_final = {t_final} in {number_of_time_steps} steps.")
        return np.array(times[:number_of_time_steps + 1], dtype=float), np.array(solutions[:number_of_time_steps + 1], dtype=float)


    def _perform_richardson_step(self, F: ODEProblem, tn: float, delta_t: float, U_np: np.ndarray, U_pred: np.ndarray):
        """
        Perform Richardson extrapolation for schemes without embedded error estimators.

        The method performs two half-steps and combines them for higher-order estimate.

        Parameters
        ----------
        F : ODEProblem
            ODE system to integrate.
        tn : float
            Current time.
        delta_t : float
            Step size.
        U_np : np.ndarray
            Current solution vector u_n.
        U_pred : np.ndarray
            Initial predicted solution (usually u_n).

        Returns
        -------
        U_n_plus_1 : np.ndarray
            Extrapolated solution at tn + delta_t.

        Raises
        ------
        ValueError
            If Newton iteration fails during any half-step.
        """
        # First half-step
        U_half_step, _, newton_not_happy = \
            self._perform_single_rk_step(
                F, tn, delta_t / 2.0, U_np
            )
        if newton_not_happy:
            raise ValueError("Newton failed during the first Richardson half-step.")

        # Second half-step
        U_n_plus_1, _, newton_not_happy = \
            self._perform_single_rk_step(
                F, tn + delta_t / 2.0, delta_t / 2.0, U_half_step
            )
        if newton_not_happy:
            raise ValueError("Newton failed during the second Richardson half-step.")

        return U_n_plus_1


    def _solve_with_fixed_step_size(self, ode_problem: ODEProblem):
        """
        Solve ODE system with a fixed time step.

        Parameters
        ----------
        ode_problem : ODEProblem
            ODE system to integrate.

        Returns
        -------
        times : np.ndarray
            Array of time points.
        solutions : np.ndarray
            Array of solution vectors corresponding to each time point.

        Raises
        ------
        PyodysError
            If Newton iterations fail during implicit stages.
        """
        U_courant = np.copy(ode_problem.initial_state)
        current_time = ode_problem.t_init
        max_number_of_time_steps = int((ode_problem.t_final - ode_problem.t_init) / self.fixed_step)

        if self.progress_interval_in_time == None:
            self.progress_interval_in_time = (ode_problem.t_final - ode_problem.t_init) / 100.0

        next_progress_in_time = ode_problem.t_init + self.progress_interval_in_time
        if self.export_interval:
            times = np.empty(self.export_interval+1, dtype=float)
            solutions = np.empty((self.export_interval+1, len(ode_problem.initial_state)), dtype=float)
        else :
            times = np.empty(max_number_of_time_steps+1, dtype=float)
            solutions = np.empty((max_number_of_time_steps+1, len(ode_problem.initial_state)), dtype=float)
        times[0] = ode_problem.t_init
        solutions[0,:] = U_courant

        k = 0

        if (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk) and ode_problem.jacobian_is_constant:
            self._build_prefactored_dirk_solver(self.fixed_step, self.gamma_sdirk)

        for n in range(max_number_of_time_steps):
            U_n_plus_1, _, newton_not_happy = self._perform_single_rk_step(
                ode_problem, current_time, self.fixed_step, U_courant
            )
            if newton_not_happy:
                self.newton_failed = True
                message = f"Newton failed at time step {n+1} even after Jacobian refresh."
                super()._print_verbose(message)
                raise PyodysError(message)

            U_courant = U_n_plus_1
            current_time += self.fixed_step
            times[k+1] = current_time
            solutions[k+1, :] = U_courant
            k += 1

            if self.export_interval and k == self.export_interval:
                super()._export(times[:k], solutions[:k, :])
                times[0] = times[k]
                solutions[0, :] = solutions[k, :]
                k = 0

            if current_time >= next_progress_in_time:
                super()._print_verbose(
                    f"Time step #{n+1} completed. Current time: {current_time:.4f}"
                )
                next_progress_in_time += self.progress_interval_in_time
        
        if self.export_interval:
            if k > 0:
                super()._export(times[:k+1], solutions[:k+1, :])
            print(f"Simulation completed. The results have been saved to {self.export_prefix}*.csv")
            return None

        return np.array(times), np.array(solutions)
