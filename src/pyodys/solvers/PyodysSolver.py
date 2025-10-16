from .BDFSolver import BDFSolver
from .RKSolver import RKSolver
from ..schemes.bdf.BDFScheme import BDFScheme
from ..schemes.rk.RKScheme import RKScheme
from ..ode.ODEProblem import ODEProblem
from ..utils.pyodys_utils import PyodysError
from numpy import inf
from typing import Union, Callable


class PyodysSolver(object):
    """
    High-level interface for solving Ordinary Differential Equations (ODEs)
    using various numerical schemes.

    This class serves as a factory for specific ODE solvers (Runge-Kutta, BDF)
    based on the `method` provided. It handles method selection and
    initialization of the appropriate solver with all user-defined parameters.

    Parameters
    ----------
    method : Union[str, RKScheme, BDFScheme]
        The numerical scheme to use. It can be a string name of a predefined
        scheme (e.g., 'dopri54', 'erk4', 'sdirk2', 'sdirk21', 'dirk64', 'esdirk64', 'bdf1', 'bdf2'), or a custom
        `RKScheme` or `BDFScheme` object.
    fixed_step : float, optional
        The fixed step size for non-adaptive solvers. Required if `adaptive` is False.
    adaptive : bool, default True
        If True, enable adaptive step size control.
    first_step : float, optional
        Initial step size for adaptive solvers. If not provided, it's estimated automatically.
    min_step : float, optional
        Minimum allowed step size for adaptive solvers.
    max_step : float, optional
        Maximum allowed step size for adaptive solvers.
    nsteps_max : int, default 100000
        Maximum number of time steps before the solver terminates.
    newton_nmax : int, default 10
        Maximum number of iterations for Newton's method in implicit solvers.
    rtol : float, default 1e-8
        Relative tolerance for adaptive step-size control.
    atol : float, default 1e-8
        Absolute tolerance for adaptive step-size control.
    linear_solver : Union[str, Callable], default 'lu'
        Linear solver used for implicit schemes.
    linear_solver_opts : dict, optional
        Additional options for the linear solver.
    max_jacobian_refresh : int, default 1
        Maximum number of times to re-evaluate the Jacobian per step in implicit solvers.
    verbose : bool, default False
        Enables verbose output for debugging and progress tracking.
    progress_interval_in_time : int, optional
        Time interval at which to print progress updates.
    export_interval : int, optional
        Number of steps between data exports to a CSV file.
    export_prefix : str, optional
        File path prefix for exported CSV files.
    auto_check_sparsity : bool, default True
        Enables automatic sparsity detection for a system's matrices.
    sparse_threshold : int, default 20
        The minimum number of equations for a sparsity check to be performed.
    sparsity_ratio_limit : float, default 0.2
        The density threshold for a matrix to be considered sparse.
    initial_step_safety : float, default 1e-4
        A safety factor used in the initial step size estimation for adaptive solvers.

    Attributes
    ----------
    _solver_cls : Union[RKSolver, BDFSolver]
        The class of the solver to be instantiated.
    _solver_kwargs : dict
        A dictionary containing all keyword arguments passed to the solver.
    _call_count : int
       The number of time the solver is called.

    Raises
    ------
    TypeError
        If `method` is not a valid type or if required arguments for
        adaptive stepping are missing.
    ValueError
        If the specified `method` name is not a known predefined scheme.
    """
    def __init__(self, method: Union[str, RKScheme, BDFScheme] = None,
                 fixed_step: float = None,
                 adaptive: bool = True,
                 first_step: float = None,
                 min_step: float = None, 
                 max_step: float = None,
                 nsteps_max: int = None,
                 newton_nmax: int = 10,
                 rtol: float = 1e-8,
                 atol: float = 1e-8,
                 linear_solver : Union[str, Callable] = "lu",
                 linear_solver_opts : dict = None,
                 max_jacobian_refresh: int = 1,
                 verbose: bool = False,
                 progress_interval_in_time: int = None,
                 export_interval: int = None,
                 export_prefix: str = None,
                 auto_check_sparsity: bool = True,
                 sparse_threshold: int = 20,
                 sparsity_ratio_limit: float = 0.2,
                 initial_step_safety = 1e-4):
        
        self._solver_cls=None
        if isinstance(method, str):
            rk_schemes = RKScheme.available_schemes()
            bdf_schemes= BDFScheme.available_schemes()
            available = "\n".join(rk_schemes) + "\n" + "\n".join(bdf_schemes)
            if method in rk_schemes:
                self._solver_cls = RKSolver
            elif method in bdf_schemes:
                self._solver_cls = BDFSolver
            else:
                raise ValueError(
                    f"There is no available scheme with name {method}. "
                    f"Here is the list of available schemes:\n{available}"
                )
        elif isinstance(method, RKScheme):
            self._solver_cls = RKSolver
        elif isinstance(method, BDFScheme):
            self._solver_cls = BDFSolver
        else:
            raise TypeError("method must be an instance of `RKScheme`, `BDFScheme`,or a str object from the list of predefined schemes names.")

        self._solver_kwargs = dict(
            method=method,
            fixed_step=fixed_step,
            adaptive=adaptive,
            first_step=first_step,
            min_step=min_step,
            max_step=max_step,
            nsteps_max=nsteps_max,
            newton_nmax=newton_nmax,
            rtol=rtol,
            atol=atol,
            linear_solver=linear_solver,
            linear_solver_opts = linear_solver_opts,
            max_jacobian_refresh=max_jacobian_refresh,
            verbose=verbose,
            progress_interval_in_time=progress_interval_in_time,
            export_interval=export_interval,
            export_prefix=export_prefix,
            auto_check_sparsity=auto_check_sparsity,
            sparse_threshold=sparse_threshold,
            sparsity_ratio_limit=sparsity_ratio_limit,
            initial_step_safety=initial_step_safety
        )

        self._call_count = 0
        

    def solve(self, ode_problem: ODEProblem):
        """
        Solves the Ordinary Differential Equation (ODE) problem.

        This method initializes and executes the appropriate solver (RKSolver or BDFSolver)
        based on the chosen method.

        Parameters
        ----------
        ode_problem : ODEProblem
            The ODE problem instance containing the system of equations,
            initial conditions, and time span.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays:
            - The first array represents the time points of the solution.
            - The second array contains the solution at each corresponding time point.
        None if the export mode is activated
        """
        if self._solver_kwargs['verbose'] and self._call_count==0:
            self._print_params()

        self._call_count+=1
        solver = self._solver_cls(**self._solver_kwargs)
        return solver.solve(ode_problem)
        

    def _print_params(self):
        """
        Prints the solver's parameters in a clear, formatted table.
        """
        print("----------------------------------------")
        print("       Pyodys Solver Parameters         ")
        print("----------------------------------------")

        # Sort the dictionary keys for a consistent print order
        sorted_params = sorted(self._solver_kwargs.items())

        for key, value in sorted_params:
            # Use an f-string to format key and value with alignment
            print(f"{key:<25}: {value}")

        print("----------------------------------------")

    
