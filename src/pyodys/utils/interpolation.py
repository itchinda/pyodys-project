import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly
from typing import Union, Callable

# Assuming this import path is correct based on your project structure
from ..systemes.EDOs import EDOs


def pchipd(x: ArrayLike, y: ArrayLike, d: ArrayLike, xx: ArrayLike = None):
    """
    Piecewise Cubic Hermite Interpolating Polynomial with Derivatives.

    Constructs a piecewise cubic polynomial that interpolates given function
    values and derivatives at a set of breakpoints.

    Parameters
    ----------
    x : array_like, shape (n,)
        Breakpoints (must be sorted).
    y : array_like, shape (n,) or (m, n)
        Function values at x.
    d : array_like, shape (n,) or (m, n)
        Derivatives at x.
    xx : array_like, optional
        Points at which to evaluate the interpolant. If not provided,
        a `PPoly` object is returned.

    Returns
    -------
    pp : scipy.interpolate.PPoly or ndarray
        If `xx` is None, returns a `PPoly` object representing the piecewise
        polynomial. If `xx` is provided, returns the interpolated values
        at `xx` as an `ndarray`.

    Raises
    ------
    ValueError
        If input array dimensions do not match requirements.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1D array of length >= 2")
    n = len(x)

    # Handle y and d dimensions for both single and multiple dimensions
    if y.ndim == 1 and d.ndim == 1:
        if len(y) != n or len(d) != n:
            raise ValueError("y and d must match length of x")
        y = y.reshape(-1, 1)
        d = d.reshape(-1, 1)
    elif y.shape == d.shape and y.shape[0] == n:
        pass  # Dimensions are already correct
    else:
        raise ValueError("y and d must be vectors of length n or matrices of shape (n, m)")

    # Sort x if not already sorted, and re-order y and d accordingly
    sort_idx = np.argsort(x)
    if not np.all(sort_idx == np.arange(n)):
        x = x[sort_idx]
        y = y[sort_idx, :]
        d = d[sort_idx, :]

    # Calculate differences
    dx = np.diff(x)
    dy = np.diff(y, axis=0)

    # Calculate the coefficients for the cubic polynomial segments
    # The polynomial is defined as:
    # a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + d
    # Coefficients are stored in rows of `coef` for each segment
    coef = np.zeros((4, n - 1, y.shape[1]), dtype=float)

    # d: constant term
    coef[3, :, :] = y[:-1, :]

    # c: linear term
    coef[2, :, :] = d[:-1, :]

    # b: quadratic term
    coef[1, :, :] = (3 * dy / (dx[:, None] ** 2)) - (2 * d[:-1, :] + d[1:, :]) / dx[:, None]

    # a: cubic term
    coef[0, :, :] = (-2 * dy / (dx[:, None] ** 3)) + (d[:-1, :] + d[1:, :]) / (dx[:, None] ** 2)

    # Create the PPoly object
    pp = PPoly(coef, x, extrapolate=False)

    if xx is not None:
        # Evaluate the polynomial at new points if requested
        return pp(xx)
    return pp


def hermite_interpolate(xi: ArrayLike, yi: ArrayLike, f: Union[EDOs, Callable[[ArrayLike, ArrayLike], ArrayLike]], xnouveau: ArrayLike):
    """
    Hermite solution interpolation for systems of Ordinary Differential Equations (ODEs).

    This function uses the `pchipd` function to perform Hermite interpolation on a
    solution to an ODE system, given the solution's values and derivatives at
    a set of known points.

    Parameters
    ----------
    xi : array_like, shape (nbx,)
        Abscissas (e.g., time points) where the solution is known.
    yi : array_like, shape (nbx, nbeq)
        Values of the solution at `xi`. Each row corresponds to a point in `xi`,
        and each column to a different variable in the ODE system.
    f : Union[EDOs, Callable[[ArrayLike, ArrayLike], ArrayLike]]
        The function `f(t, y)` that returns the derivative `dy/dt`. This can be
        an instance of the `EDOs` class or any callable that accepts a time `t`
        and a solution array `y`.
    xnouveau : array_like
        Points where the interpolated solution is desired.

    Returns
    -------
    yinouveau : ndarray, shape (len(xnouveau), nbeq)
        The interpolated solution values at the `xnouveau` points.

    Raises
    ------
    ValueError
        If input dimensions do not match or the derivative function `f`
        returns a value with an incorrect shape.
    """
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)
    xnouveau = np.asarray(xnouveau, dtype=float)

    if xi.ndim != 1:
        raise ValueError("xi must be a 1D array (row vector).")
    nbx = xi.shape[0]

    if yi.shape[0] != nbx:
        raise ValueError("yi must have nbx rows (same length as xi).")
    nbeq = yi.shape[1]

    # Compute derivatives at each xi using the provided function f
    fprimei = np.zeros_like(yi, dtype=float)
    is_edos_instance = isinstance(f, EDOs)
    for i in range(nbx):
        # The function `f` is expected to return the derivative for a single point
        if is_edos_instance:
            derivative_at_point = f.evalue(xi[i], yi[i, :])
        else:
            derivative_at_point = f(xi[i], yi[i, :])

        # Ensure the derivative has the expected shape (nbeq,)
        derivative_at_point = np.asarray(derivative_at_point)
        if derivative_at_point.shape not in [(nbeq,), (nbeq, 1)]:
            raise ValueError(f"f must return a vector of shape ({nbeq},) for given (t,y).")
        fprimei[i, :] = derivative_at_point.flatten()

    # Call pchipd to perform the interpolation
    yinouveau = pchipd(xi, yi, fprimei, xnouveau)
    
    return yinouveau
