import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly
from typing import Union, Callable
from pyodys import ODEProblem


def hermite_pchipd(x: ArrayLike, y: ArrayLike, d: ArrayLike, xx: ArrayLike = None, extrapolate: bool = False):
    """
    Piecewise cubic Hermite interpolation using specified derivatives.

    Constructs a cubic Hermite interpolant such that the polynomial
    passes through given points `y` with derivatives `d`.

    Parameters
    ----------
    x : array_like, shape (n,)
        Breakpoints (must be sorted).
    y : array_like, shape (n,) or (n, m)
        Function values at breakpoints.
    d : array_like, shape (n,) or (n, m)
        Derivatives at breakpoints.
    xx : array_like, optional
        Points at which to evaluate the interpolant. If None, returns PPoly object.
    extrapolate : bool, default False
        Whether to allow evaluation outside the original interval.

    Returns
    -------
    pp : PPoly or ndarray
        If `xx` is None, returns a PPoly object. Otherwise, returns values at `xx`.
    """
    x = np.asarray(x, dtype=float)
    y = np.atleast_2d(y).astype(float)
    d = np.atleast_2d(d).astype(float)

    n = len(x)
    if x.ndim != 1 or n < 2:
        raise ValueError("x must be a 1D array of length >= 2")
    if y.shape[0] != n or d.shape[0] != n:
        raise ValueError("y and d must have same number of rows as x")

    # Ensure sorted
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx, :]
    d = d[sort_idx, :]

    dx = np.diff(x)
    dy = np.diff(y, axis=0)

    coef = np.zeros((4, n-1, y.shape[1]), dtype=float)
    coef[3, :, :] = y[:-1, :]                     # constant term
    coef[2, :, :] = d[:-1, :]                     # linear term
    coef[1, :, :] = (3*dy/dx[:, None]**2) - (2*d[:-1, :] + d[1:, :])/dx[:, None]  # quadratic
    coef[0, :, :] = (-2*dy/dx[:, None]**3) + (d[:-1, :] + d[1:, :])/dx[:, None]**2  # cubic

    pp = PPoly(coef, x, extrapolate=extrapolate)

    if xx is not None:
        return pp(xx)
    return pp


def hermite_interpolate(xi: ArrayLike, yi: ArrayLike, f: Union[ODEProblem, Callable[[ArrayLike, ArrayLike], ArrayLike]], xnew: ArrayLike):
    """
    Hermite interpolation of ODE solutions using values and derivatives.

    Parameters
    ----------
    xi : array_like, shape (nbx,)
        Abscissas where solution is known.
    yi : array_like, shape (nbx, nbeq)
        Solution values at `xi`.
    f : ODEProblem or callable
        Function returning derivatives: f(t, y) -> dy/dt.
    xnew : array_like
        Points where interpolation is desired.

    Returns
    -------
    ynew : ndarray, shape (len(xnew), nbeq)
        Interpolated solution at `xnew`.
    """
    xi = np.asarray(xi, dtype=float)
    yi = np.atleast_2d(yi).astype(float)
    xnew = np.asarray(xnew, dtype=float)

    nbx = xi.shape[0]
    if yi.shape[0] != nbx:
        raise ValueError("yi must have same number of rows as xi")
    nbeq = yi.shape[1]

    fprimei = np.zeros_like(yi)
    is_ODEProblem = isinstance(f, ODEProblem)

    for i in range(nbx):
        deriv = f.evalue(xi[i], yi[i, :]) if is_ODEProblem else f(xi[i], yi[i, :])
        deriv = np.atleast_1d(deriv)
        if deriv.shape[0] != nbeq:
            raise ValueError(f"Derivative at xi[{i}] has incorrect shape")
        fprimei[i, :] = deriv

    return hermite_pchipd(xi, yi, fprimei, xnew)