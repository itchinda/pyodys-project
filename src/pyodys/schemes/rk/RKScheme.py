import numpy as np
import json
import os
from typing import Union, ClassVar, List
from functools import lru_cache
from typing import Dict, Any
from ..scheme import Scheme
from ...utils.pyodys_utils import _DEFAULT_ZERO_TOL
from .built_in_rk_schemes.built_in_rk_schemes import get_scheme, get_available_rk_schemes_names

class RKScheme(Scheme):
    """
    Representation of a Runge-Kutta scheme using a Butcher tableau.

    A Butcher tableau is defined by three arrays (A, B, C) and an order.
    It encodes the coefficients of a Runge-Kutta method for ODE integration.

    Parameters
    ----------
    A : np.ndarray (s x s)
        Stage coefficients (matrix).
    B : np.ndarray (length s or shape (2, s))
        Weights of the method.
        - shape (s,) for classical RK methods
        - shape (2, s) for embedded methods (predictor/corrector pair)
    C : np.ndarray (length s or shape (s, 1))
        Nodes of the method (usually sum of rows of A).
    order : int
        The order of the scheme.

    Raises
    ------
    TypeError
        If A, B, C are not NumPy arrays of numeric type, or if order is not int/float.
    ValueError
        If dimensions of A, B, or C are inconsistent.

    Attributes
    ----------
    A : np.ndarray
        Stage coefficient matrix.
    B : np.ndarray
        Weight vector(s).
    C : np.ndarray
        Node vector.
    order : int
        The order of accuracy of the scheme.
    embedded_order : int, optional
            The order of accuracy of the embedded scheme. Must be provided if the method has an embedded secheme for error estimate.
    n_stages : int
        Number of stages (size of A).
    with_prediction : bool
        True if method has embedded predictor/corrector (B has two rows).
    is_explicit : bool
        True if method is explicit (strictly lower-triangular A).
    is_implicit : bool
        True if method is implicit (not explicit).
    is_diagonally_implicit : bool
        True if method is a DIRK (diagonal entries all equal and nonzero).

    Methods
    -------
    __str__():
        Return a formatted string of the Butcher tableau.
    par_nom(nom: str) -> RKScheme:
        Load a predefined scheme by French name.
    from_name(name: str) -> RKScheme:
        Load a predefined scheme by English name (alias).

    Examples
    --------
    >>> tableau = RKScheme.from_name("erk4")
    >>> print(tableau)
    Runge-Kutta method of order 4
    
                     0 |                  0                  0                  0                  0
                   0.5 |                0.5                  0                  0                  0
                   0.5 |                  0                0.5                  0                  0
                     1 |                  0                  0                  1                  0         
    -------------------+-----------------------------------------------------------------------------
                         0.1666666666666667 0.3333333333333333 0.3333333333333333 0.1666666666666667
    
    >>> tableau.is_explicit
    True
    >>> tableau.n_stages
    4
    """

    def __init__(self, A: np.ndarray, 
                 B: np.ndarray, 
                 C: np.ndarray, 
                 order: Union[int, float], 
                 embedded_order: Union[int, float] = None, 
                 check_consistency: bool = False,
                 a_stable: bool = None,
                 l_stable: bool = None,
                 reference: str = None
                 ):
        """
        Initialize a Runge-Kutta Butcher tableau.

        Parameters
        ----------
        A : np.ndarray, shape (s, s)
            Stage coefficient matrix. Must be square (s x s).
        B : np.ndarray, shape (s,) or (2, s)
            Weights of the method.
            - Shape (s,) for classical RK methods.
            - Shape (2, s) for embedded methods (predictor/corrector pair).
        C : np.ndarray, shape (s,) or (s, 1)
            Node vector, usually sum of rows of A.
        order : int or float
            The order of accuracy of the scheme.
        embedded_order : int or float, optional
            The order of accuracy of the embedded scheme. Must be provided if the method has an embedded secheme for error estimate.
        check_consistency : bool, optional (default=False)
            If True, raise a ValueError if the sum of each row of A does not equal the corresponding entry of C, or if the sum of each row of B does not equal 1.0.
            If False, no check is performed (user-defined tableaus with custom C and B allowed).

        Raises
        ------
        TypeError
            If A, B, C are not NumPy arrays of numeric type, or if `order` is not int/float.
        ValueError
            If dimensions of A, B, or C are inconsistent, or if check_consistency=True and sum(A, axis=1) != C.

        Examples
        --------
        >>> import numpy as np
        >>> from pyodys import RKScheme
        >>> A = np.array([[0.0, 0.0], [0.5, 0.0]])
        >>> B = np.array([0.5, 0.5])
        >>> C = np.sum(A, axis=1)
        >>> tableau = RKScheme(A, B, C, order=2, check_consistency=True)
        """
        # Type checks
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise TypeError("A doit être une matrice numpy de dimension 2.")
        if not isinstance(B, np.ndarray) or B.ndim not in (1, 2):
            raise TypeError("B doit être un vecteur numpy de dimension 1 ou 2.")
        if not isinstance(C, np.ndarray) or C.ndim not in (1, 2):
            raise TypeError("C doit être un vecteur numpy de dimension 1 ou 2.")
        if not isinstance(order, (int, float)):
            raise TypeError("Le paramètre 'order' doit être de type entier ou réel.")
        if embedded_order is not None and not isinstance(embedded_order, (int, float)):
            raise TypeError("Le paramètre 'embedded_order' doit être de type entier ou réel.")
        if B.ndim == 2 and embedded_order is None:
            raise ValueError("For embedded schemes, you must provide the volue of the parameter 'embedded_order' for the embedded scheme order.")
        if not all(np.issubdtype(arr.dtype, np.number) for arr in [A, B, C]):
            raise TypeError("Les tableaux A, B, et C doivent contenir des nombres (entiers ou réels).")
        if check_consistency:
            if not np.allclose(C, np.sum(A, axis=1), rtol=1e-14, atol=1e-14):  # NOT REALLY NECESSARY FOR CONSISTENCY, BUT THIS IS GENERALLY THE CONDITION IMPOSED TO SATISFY A SPECIFIC ORDER
                raise ValueError("Sum of A per row does not match C.")
            if B.ndim == 1 and not np.isclose(1.0, np.sum(B), rtol=1e-14, atol = 1e-14):
                raise ValueError("Sum of B coefficients does not match 1.") # MOST BE SATISFIED FOR CONSISTENCY
            elif B.ndim == 2 and not np.allclose(np.ones(2, dtype=float), np.sum(B, axis=1), rtol=1e-14, atol = 1e-14):
                raise ValueError("Sum of B coefficients per rows does not match 1.") # MOST BE SATISFIED FOR CONSISTENCY

        # Shape checks
        rows_A, cols_A = A.shape
        if rows_A != cols_A:
            raise ValueError("La matrice A doit être carrée.")
        s = rows_A

        if B.ndim == 1:
            if B.size != s:
                raise ValueError(f"Le vecteur B doit avoir une taille de {s}.")
        else:  # B.ndim == 2
            rows_B, cols_B = B.shape
            if cols_B != s or rows_B not in (1, 2):
                raise ValueError(f"Le vecteur B doit avoir {s} colonnes et 1 ou 2 lignes.")

        if C.ndim == 1:
            if C.size != s:
                raise ValueError(f"Le vecteur C doit avoir une taille de {s}.")
        else:  # C.ndim == 2
            rows_C, cols_C = C.shape
            if rows_C != s or cols_C != 1:
                raise ValueError(f"Le vecteur C doit être une matrice {s}x1.")
            
        super().__init__(order)
        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.C: np.ndarray = C
        self.embedded_order: Union[int, float] = embedded_order
        self.a_stable = a_stable
        self.l_stable = l_stable
        self.reference = reference

    # ------------------ Properties ------------------
    @property
    def n_stages(self) -> int:
        """Number of stages in the RK scheme."""
        return self.A.shape[0]

    @property
    def with_prediction(self) -> bool:
        """Whether the method is embedded (has predictor/corrector)."""
        return self.B.ndim == 2 and self.B.shape[0] == 2

    @property
    def is_explicit(self) -> bool:
        """
        Explicit RK if A is strictly lower-triangular, i.e. all diagonal and
        above-diagonal entries are (close to) zero.
        """
        return np.allclose(np.triu(self.A, 0), 0.0, atol=_DEFAULT_ZERO_TOL, rtol=_DEFAULT_ZERO_TOL)

    @property
    def is_implicit(self) -> bool:
        """True if not explicit."""
        return not self.is_explicit

    @property
    def is_diagonally_implicit(self) -> bool:
        """
        DIRK test: upper triangle above diagonal should be zero,At least one diagonal entrie
        should be non-zero.
        Returns True for general DIRK (diagonal may vary).
        """
        # upper triangle above diagonal must be (close to) zero
        if not np.allclose(np.triu(self.A, 1), 0.0, atol=_DEFAULT_ZERO_TOL, rtol=_DEFAULT_ZERO_TOL):
            return False

        diag = np.diag(self.A)
        # require at least one diagonal entrie to be non-zero (larger than tolerance)
        return np.any(np.abs(diag) > _DEFAULT_ZERO_TOL)

    @property
    def is_sdirk(self) -> bool:
        """
        Stiffly-diagonally-implicit RK (SDIRK): DIRK with all diagonal entries equal.
        """
        if not self.is_diagonally_implicit:
            return False
        diag = np.diag(self.A)
        return np.allclose(diag, diag[0], atol=_DEFAULT_ZERO_TOL, rtol=_DEFAULT_ZERO_TOL)

    @property
    def is_esdirk(self) -> bool:
        """
        Check if the scheme is ESDIRK (Explicit Singly Diagonally Implicit Runge-Kutta): 
        - First stage explicit (a_11 ~= 0)
        - Remaining diagonal entries equal and non-zero
        - Lower-triangular A
        """
        if not np.allclose(np.triu(self.A, 1), 0.0, atol=_DEFAULT_ZERO_TOL):
            return False

        diag = np.diag(self.A)
        if np.abs(diag[0]) > _DEFAULT_ZERO_TOL:
            return False  # first stage must be explicit

        remaining_diag = diag[1:]
        nonzero_diag = remaining_diag[np.abs(remaining_diag) > _DEFAULT_ZERO_TOL]
        return len(nonzero_diag) > 0 and np.allclose(nonzero_diag, nonzero_diag[0], atol=_DEFAULT_ZERO_TOL)

    # ------------------ Display ------------------
    def __str__(self) -> str:
        s = self.A.shape[0]
        output = f"Runge-Kutta method of order {self.order}\n\n"
        output += f"Embedded scheme order: {self.embedded_order}" if self.embedded_order is not None else ""

        # Determine max width for alignment
        max_width = max(
            len(f"{x:.16g}")
            for x in np.concatenate([self.A.flatten(), self.B.flatten(), self.C.flatten()])
        )

        # Print C and A
        for i in range(s):
            c_val = f"{self.C[i]:>{max_width}.16g}"
            a_row = " ".join(f"{val:>{max_width}.16g}" for val in self.A[i])
            output += f"{c_val} | {a_row}\n"

        # Separator
        separator = "-" * (max_width + 1) + "+" + "-" * (s * (max_width + 1) + 1)
        output += f"{separator}\n"

        # Print B
        if self.B.ndim == 1:
            b_row = " ".join(f"{val:>{max_width}.16g}" for val in self.B)
            output += f"{'':>{max_width}}   {b_row}\n"
        else:
            for row in self.B:
                b_row = " ".join(f"{val:>{max_width}.16g}" for val in row)
                output += f"{'':>{max_width}}   {b_row}\n"

        if self.reference is not None:
            output += f"Reference: {self.reference}\n"

        return output
    
    def info(self) -> str:
        """
        Return a concise textual summary of the Runge-Kutta type.
        
        Example output:
            Type: ESDIRK
            Stages: 6
            Order: 4
            Embedded: Yes
            Embedded order: 3
        """
        if self.is_esdirk:
            rk_type = "ESDIRK"
        elif self.is_sdirk:
            rk_type = "SDIRK"
        elif self.is_diagonally_implicit:
            rk_type = "DIRK"
        elif self.is_implicit:
            rk_type = "Implicit RK"
        elif self.is_explicit:
            rk_type = "Explicit RK"
        else:
            rk_type = "Unknown"

        if self.a_stable is not None:
            a_stable = "Yes" if self.a_stable else "No"
        if self.l_stable is not None:
            l_stable = "Yes" if self.l_stable else "No"


        solver_info = (
            f"Type: {rk_type}\n"
            f"Stages: {self.n_stages}\n"
            f"Order: {self.order}\n"
            f"Embedded: {'Yes' if self.with_prediction else 'No'}"
        )
        if self.with_prediction:
            solver_info += f"\nEmbedded order: {self.embedded_order}"

        if self.a_stable is not None:
            solver_info += f"\nA-stable: {self.a_stable}"

        if self.l_stable is not None:
            solver_info += f"\nL-stable: {self.l_stable}"
        
        if self.reference is not None:
            solver_info += f"\nReference: {self.reference}"
        return solver_info

    @classmethod
    def from_name(cls, name: str) -> "RKScheme":
        """
        Load a predefined scheme by name.
        Args:
            name : The name of the method.

        Examples:
        >>> from pyodys import RKScheme
        >>> print(RKScheme.available_schemes())
        ['erk1', 'erk2_midpoint', 'erk4', 'sdirk1', 'sdirk2_midpoint', 'sdirk43_crouzeix', 'cooper_verner', 'euler_heun', 'bogacki_shampine', 'fehlberg45', 'dopri5', 'sdirk21_crouzeix_raviart', 'sdirk_norsett_thomson_23', 'sdirk_norsett_thomson_34', 'sdirk_hairer_norsett_wanner_45', 'esdirk6']
        >>> tableau = RKScheme.from_name('erk4')
        >>> tableau.info()
        Type: Explicit RK
        Stages: 4
        Order: 4
        Embedded: No
        """

        scheme_data = get_scheme(name)
        try:
            embedded_order = scheme_data['embedded_order']
        except KeyError:
            embedded_order = None

        try:
            a_stable = scheme_data['a_stable']
        except KeyError:
            a_stable = None

        try:
            l_stable = scheme_data['l_stable']
        except KeyError:
            l_stable = None

        return cls(np.asarray(scheme_data["A"]), np.asarray(scheme_data["B"]), np.asarray(scheme_data["C"]), scheme_data["order"], embedded_order, True, a_stable, l_stable)

    @classmethod
    def par_nom(cls, nom: str) -> "RKScheme":
        """Alias in French for `from_name`."""
        return cls.from_name(nom)

    @classmethod
    def available_schemes(cls):
        """List names of available schemes."""
        return get_available_rk_schemes_names()

