import argparse
from numpy import inf
def extract_args(
        description:str = None,
        method = 'esdirk64',
        fixed_step = 1e-4,
        first_step:float = None,
        final_time:float = None,
        rtol = 1e-10,
        atol = 1e-10,
        min_step = 1e-8,
        max_step = inf
    ):
    """
    Parse and extract command-line arguments for configuring an ODE solver run.

    This helper sets up a flexible command-line interface for running simulations 
    with the PyOdys ODE solver. It allows the user to specify the integration method, 
    time-stepping parameters, and output options.

    Parameters
    ----------
    description : str, optional
        The description of the ODE system to be solved (default: None)
    method : str, optional
        The Runge-Kutta scheme identifier (default: "esdirk64").
    fixed_step : float, optional
        Fixed step size used when adaptive stepping is disabled (default: 1e-4).
    first_step : float or None, optional
        Initial step size for adaptive stepping (default: None, let solver decide).
    final_time : float or None, optional
        Final simulation time (default: None).
    rtol : float, optional
        Target relative tolerance for adaptive time-stepping (default: 1e-10).
    atol : float, optional
        Target absolute tolerance for adaptive time-stepping (default: 1e-10).
    min_step : float, optional
        Minimum time step size for adaptive schemes (default: 1e-8).
    max_step : float, optional
        Maximum time step size for adaptive schemes (default: 100).

    Command-line Arguments
    ----------------------
    --method, -m : str
        Name of the time integration method.
    --fixed-step, -f : float
        Fixed time-step size (used if adaptive stepping is disabled).
    --first-step, -s : float
        Initial time step size.
    --final-time, -t : float
        Final time for the simulation.
    --rtol, -rt : float
        Relative tolerance for adaptive stepping.
    --atol, -at : float
        Absolute tolerance for adaptive stepping.
    --no-adaptive :
        Disable adaptive time-stepping (use fixed step instead).
    --min-step, -n : float
        Minimum allowable time step size.
    --max-step, -x : float
        Maximum allowable time step size.
    --save-csv :
        Save the results to a CSV file.
    --save-png :
        Save the results as a PNG plot.
    --verbose, -v :
        Enable verbose mode for progress reporting.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing all solver configuration parameters.

    Examples
    --------
    Run a simulation with a specific method and adaptive tolerance:
        $ python run_solver.py --method dopri54 --rtol 1e-8 --atol 1e-8

    Run a simulation with fixed time stepping:
        $ python run_solver.py --no-adaptive --fixed-step 1e-3

    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--method', '-m', 
                        type=str, 
                        default=method,
                        help='The Runge-Kutta method to use.')
    parser.add_argument('--fixed-step', '-f', 
                        type=float, 
                        default=fixed_step,
                        help='The fixed step size used if adaptive stepping is disabled.')
    parser.add_argument('--first-step', '-s', 
                        type=float, 
                        default=first_step,
                        help='The initial time step size.')
    parser.add_argument('--final-time', '-t', 
                        type=float, 
                        default=final_time,
                        help='The final time for the simulation.')
    parser.add_argument('--rtol', '-rt', 
                        type=float,
                        default=rtol,
                        help='The target relative error for adaptive time stepping.')
    parser.add_argument('--atol', '-at', 
                        type=float,
                        default=atol,
                        help='The target absolute error for adaptive time stepping.')
    parser.add_argument('--no-adaptive', 
                        action='store_false', 
                        dest='adaptive',
                        help='Disable adaptive time stepping.')
    parser.add_argument('--min-step','-n', 
                        type=float,
                        default=min_step,
                        help='The minimum time step size for adaptive stepping.')
    parser.add_argument('--max-step', '-x',
                        type=float,
                        default=max_step,
                        help='The maximum time step size for adaptive stepping.')
    parser.add_argument('--save-csv', 
                        action='store_true', 
                        help='Save the results to a CSV file.')
    parser.add_argument('--save-png', 
                        action='store_true', 
                        help='Save the results to a png file.')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print progress info.')

    return parser.parse_args()