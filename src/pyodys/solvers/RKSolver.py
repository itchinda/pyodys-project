from ..ode.ODEProblem import ODEProblem
from ..schemes.rk.RKScheme import RKScheme
from .SolverBase import SolverBase
from ..utils import pyodys_utils as utils
import numpy as np
from typing import Union
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
        Fixed step size. Required if adaptive=False.
    adaptive : bool, default=False
        If True, the solver adjusts the step size to satisfy error tolerances.
    first_step : float, optional
        Initial step size for adaptive mode. If None, estimated automatically.
    min_step, max_step : float, optional
        Minimum and maximum allowed step sizes for adaptive mode.
    nsteps_max : int, default=1_000_000
        Maximum number of allowed time steps.
    newton_nmax : int, default=10
        Maximum Newton iterations per implicit stage.
    rtol, atol : float, default=1e-8
        Relative and absolute tolerances for adaptive step control.
    max_jacobian_refresh : int, default=1
        Max number of times to recompute/refactorize Jacobian within a step.
    verbose : bool, default=False
        If True, print progress/debug output.
    progress_interval_in_time : float, optional
        Time interval between progress messages.
    export_interval : int, optional
        Number of steps between CSV exports. If None, export is disabled.
    export_prefix : str, optional
        File prefix for exported CSV files.
    auto_check_sparsity : bool, default=True
        Automatically detect sparse vs dense Jacobian structure.
    sparse_threshold : int, default=20
        Minimum system size before checking sparsity.
    sparsity_ratio_limit : float, default=0.2
        If the Jacobian has fewer than this fraction of nonzeros,
        it is treated as sparse.
    initial_step_safety : float, default=1e-4
        Safety factor used when estimating the initial adaptive step size.
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
                 adaptive: bool = False,
                 first_step: float = None,
                 min_step: float = None, 
                 max_step: float = None,
                 nsteps_max: int = 1000000,
                 newton_nmax: int = 10,
                 rtol: float = 1e-8,
                 atol: float = 1e-8,
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
            raise utils.PyodysError("General implicit RK schemes are not supported. Use a diagonally-implicit variant.")

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

        if self.butcher_tableau.is_implicit:
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
    def _build_prefactored_sdirk_solver(self, delta_t):
        """Build and store linear solver (sparse or dense) for SDIRK/ESDIRK when Jacobian is not constant."""
        delta_t_x_gamma = delta_t * self.gamma_sdirk
        if self._jacobian_is_sparse:
            A_sparse = self._Id - delta_t_x_gamma * self._jacobianF
            LU = splu(A_sparse)
            self._linear_sparse_solver = LU.solve
            return self._linear_sparse_solver
        else:
            assert(self._jacobianF is not None)
            assert(self._Id is not None)
            A_dense = self._Id - delta_t_x_gamma * self._jacobianF
            LU_piv = lu_factor(A_dense)
            self._linear_dense_solver = lambda rhs: lu_solve(LU_piv, rhs)
            return self._linear_dense_solver

    def _build_linear_solver_for_akk(self, delta_t_x_akk):
        """Return a callable linear solver for given akk (delta_t * a[k,k])."""
        if self._jacobian_is_sparse:
            A_sparse = self._Id - delta_t_x_akk * self._jacobianF
            LU = splu(A_sparse)
            return LU.solve
        else:
            A_dense = self._Id - delta_t_x_akk * self._jacobianF
            LU_piv = lu_factor(A_dense)
            return lambda rhs: lu_solve(LU_piv, rhs)

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
                tn_k = tn + c[k] * delta_t
                if k == 0:
                    U_chap[:, k] = U_np
                else:
                    U_chap[:, k] = U_np + deltat_x_value_f[:, :k] @ a[k, :k]

                deltat_x_value_f[:, k] = delta_t * F.evaluate_at(tn_k, U_chap[:, k])
                U_n += b[k] * deltat_x_value_f[:, k]
                if self._with_prediction:
                    U_pred += d[k] * deltat_x_value_f[:, k]
            return U_n, U_pred, newton_not_happy

        # --- Implicit(DIRK-like) scheme → Newton iterations
        tn_k = tn
        newton_nmax = self.newton_nmax
        U_newton = self._work_U_newton

        for refresh_attempt in range(self.max_jacobian_refresh + 1):
            # compute or reuse jacobian for this attempt
            if not F.jacobian_is_constant:
                Jf = F.jacobian_at(tn_k, U_n)
                if self._jacobian_is_sparse:
                    if isspmatrix(Jf):
                        self._jacobianF = Jf.tocsc()
                    else:
                        self._jacobianF = csc_matrix(Jf)
                else:
                    self._jacobianF = np.asarray(Jf, dtype=float)

            # if jacobian is constant AND scheme is sdirk/esdirk, we can use prefactored solver
            if F.jacobian_is_constant and (self.rk_scheme_is_esdirk or self.rk_scheme_is_sdirk):
                if self._jacobian_is_sparse:
                    solver_sdirk = self._linear_sparse_solver
                else:
                    solver_sdirk = self._linear_dense_solver
            elif (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk):
                solver_sdirk = self._build_prefactored_sdirk_solver(delta_t)

            newton_failed = False
            for k in range(n_stages):
                if k == 0:
                    U_chap_k = U_np
                else:
                    U_chap_k = U_np + deltat_x_value_f[:, :k] @ a[k, :k]

                if a[k, k] == 0.0:  # No implicit coupling # USEFULL FOR ESDIRK SCHEMES!
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
                U_newton[:] = U_chap_k

                if self.rk_scheme_is_esdirk or self.rk_scheme_is_sdirk:  # use the pre-factored solver
                    linear_solver = solver_sdirk
                else:
                    linear_solver = self._build_linear_solver_for_akk(delta_t_x_akk)

                # Newton loop
                newton_succeeded = False
                for iteration_newton in range(newton_nmax):
                    # The original problem to solve is : Find K_k s.t. K_k - h* f(t_nk, u_chap_k + akk*K_k) = 0. We can set X = u_chap_k + akk*K_k to end with
                    #                                               X - u_chap_k - h*akk*f(t_nk, X) = 0, X being the new unknown.
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
                        super()._print_verbose(f"Linear solve failed at stage {k}: {e}")
                        newton_not_happy = True
                        return U_n, U_pred, newton_not_happy
                    
                    U_newton -= delta

                    eta=1
                    if utils.wrms_norm(delta, U_newton, eta*self.atol, eta*self.rtol) < 1.0:
                        newton_succeeded = True
                        break

                if not newton_succeeded:
                    newton_failed = True
                    break   # refresh jacobian and retry outer loop

                # store stage result
                U_chap[:, k] = U_newton
                deltat_x_value_f[:, k] = (U_newton - U_chap_k) / a[k,k] # = delta_t * F.evaluate_at(tn_k, U_newton). Be smart, avoid calling f again!!
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
        if not ode_problem.mass_matrix_is_identity:
            raise ValueError("Ptoblem with non identity mass matrix not currently supported.")

        self._jacobian_is_sparse = None
        self._jacobianF = None
        self._Id = None

        if self.auto_check_sparsity and not self.rk_scheme_is_erk:
            self._detect_sparsity_and_store_jacobian_if_constant(ode_problem, ode_problem.t_init, ode_problem.initial_state)

        self._nb_equations = ode_problem.number_of_equations
        n_stages = self.butcher_tableau.n_stages
        n_eq = self._nb_equations

        U_courant = np.copy(ode_problem.initial_state)
        self._work_U_chap = np.zeros((n_eq, n_stages))
        self._work_deltat_x_value_f = np.zeros((n_eq, n_stages))
        self._work_U_pred = np.empty_like(U_courant)
        self._work_U_n = np.empty_like(U_courant)
        self._work_U_newton = np.empty_like(U_courant)
        
        if not self._using_sparse_algebra and self._Id is None:
            self._Id = np.identity(n_eq, dtype = float)

        # Precompute constant jacobian for future use
        if ode_problem.jacobian_is_constant and not self.rk_scheme_is_erk and self._jacobianF is None:
            J = ode_problem.jacobian_at(ode_problem.t_init, ode_problem.initial_state)
            if self._jacobian_is_sparse:
                if isspmatrix(J):
                    self._jacobianF = J.tocsc()
                else:
                    self._jacobianF = csc_matrix(J)
            else:
                self._jacobianF = np.asarray(J, dtype=float)

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

        while current_time < t_final and number_of_time_steps < nsteps_max:
            step_size = min(step_size, t_final - current_time)

            if (self.rk_scheme_is_sdirk or self.rk_scheme_is_esdirk) and ode_problem.jacobian_is_constant: # and self.adaptive:
                self._build_prefactored_sdirk_solver(step_size)

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
                    raise utils.PyodysError(message)
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

            new_step_size, step_accepted = utils.check_step_size(
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
        """Performs Richardson extrapolation for schemes without embedded estimators."""

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
        """Solve an ODE system with a fixed time step.
        Args:
            ode_problem (ODEProblem): ODE system to integrate.
            step_size (float): Fixed time step size.
        Returns:
            tuple:
                - np.ndarray: Array of time points.
                - np.ndarray: Array of corresponding states.
        Raises:
            PyodysError: If Newton iterations fail.
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
            self._build_prefactored_sdirk_solver(self.fixed_step)

        for n in range(max_number_of_time_steps):
            U_n_plus_1, _, newton_not_happy = self._perform_single_rk_step(
                ode_problem, current_time, self.fixed_step, U_courant
            )
            if newton_not_happy:
                self.newton_failed = True
                message = f"Newton failed at time step {n+1} even after Jacobian refresh."
                super()._print_verbose(message)
                raise utils.PyodysError(message)

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

    