import numpy as np

_DEFAULT_ZERO_TOL =  10*np.finfo(float).eps if np.finfo(float).eps != 0.0 else 1e-15

class PyodysError(RuntimeError):
    """Exception raised when PyOdys fails to solve a problem."""
    def __init__(self, message):
        super().__init__(message)

def wrms_norm(delta, u, atol=1e-12, rtol=1e-6, type: str="mean"):
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

def check_step_size(
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
    err = wrms_norm(U_approx - U_pred, U_approx, atol, rtol)
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

    if previous_step_failed and abs(new_step_size - step_size) < _DEFAULT_ZERO_TOL:
        raise PyodysError(
                            f"Adaptive solver cannot advance the solution: current step {step_size:.2e} "
                            f"could not satisfy error tolerance (err = {err:.2e}). "
                            f"Consider adjusting min_step, atol, or rtol."
                        )
    return new_step_size, step_accepted
