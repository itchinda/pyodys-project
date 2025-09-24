import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Dict, Any
from ..scheme import Scheme

class BDFScheme(Scheme):
    """Representation of a BDF (Backward Differentiation Formula) scheme."""

    # Predefined BDF schemes
    _schemes: Dict[str, Dict[str, Any]] = {
        "bdf1": {"alpha": [1, -1], "beta": 1},
        "bdf2": {"alpha": [3/2, -2, 1/2], "beta": 1},
        "bdf3": {"alpha": [11/6, -3, 3/2, -1/3], "beta": 1},
        "bdf4": {"alpha": [25/12, -4, 3, -4/3, 1/4], "beta": 1},
        "bdf5": {"alpha": [137/60, -5, 5, -10/3, 5/4, -1/5], "beta": 1},
        "bdf6": {"alpha": [147/60, -360/60, 450/60, -400/60, 225/60, -72/60, 10/60], "beta": 1}
    }

    def __init__(self, alpha: ArrayLike, beta: float, order: int):
        """
        Initialize a BDF scheme.

        Parameters
        ----------
        alpha : array_like
            Coefficients of the scheme, length = order + 1.
        beta : float
            Coefficient multiplying f(t_n, y_n).
        order : int
            Order of the BDF scheme.
        """
        super().__init__(order)
        self.alpha = np.asarray(alpha, dtype=float)
        if self.alpha.ndim != 1:
            raise ValueError("alpha must be a 1D arrayLike.")
        if not isinstance(beta, (int, float)):
            raise TypeError("beta must be numeric.")
        
        expected_order = len(self.alpha) - 1
        if order != expected_order:
            raise ValueError(f"Order mismatch: provided order={order}, "
                             f"but len(alpha)-1={expected_order}.")
        self.beta = beta

    @property
    def n_stages(self) -> int:
        """Number of previous steps used by the BDF scheme (order)."""
        return len(self.alpha) - 1

    @classmethod
    def from_name(cls, name: str) -> "BDFScheme":
        """Create a BDF scheme from a predefined name."""
        name_upper = name.upper()
        if name_upper not in cls._schemes:
            available = ", ".join(cls._schemes.keys())
            raise ValueError(f"Unknown BDF scheme '{name}'. Available: {available}")
        data = cls._schemes[name_upper]
        return cls(alpha=data["alpha"], beta=data["beta"], order=len(data["alpha"]) - 1)

    @classmethod
    def available_schemes(cls):
        """Return a list of predefined BDF scheme names."""
        return list(cls._schemes.keys())

    def __str__(self) -> str:
        return f"BDF Scheme (order {self.order}, n_stages={self.n_stages}, alpha={self.alpha}, beta={self.beta})"
