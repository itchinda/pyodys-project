import numpy as np
import pytest
from pyodys import BDFScheme

# --- Test class for BDFScheme ---
class TestBDFScheme:

    # --- Valid inputs ---
    @pytest.mark.parametrize("name, alpha, beta, order", [
        ("bdf1", [1, -1], 1, 1),
        ("bdf2", [3/2, -2, 1/2], 1, 2),
        ("bdf3", [11/6, -3, 3/2, -1/3], 1, 3),
        ("bdf4", [25/12, -4, 3, -4/3, 1/4], 1, 4),
        ("bdf5", [137/60, -5, 5, -10/3, 5/4, -1/5], 1, 5),
        ("bdf6", [147/60, -360/60, 450/60, -400/60, 225/60, -72/60, 10/60], 1, 6)
    ])
    def test_valid_schemes(self, name, alpha, beta, order):
        scheme = BDFScheme(alpha, beta, order)
        assert isinstance(scheme, BDFScheme)
        assert scheme.n_stages == len(alpha) - 1
        # Order matches n_stages
        assert scheme.order == len(alpha) - 1

    # --- Type and shape checks ---
    def test_invalid_alpha_type(self):
        with pytest.raises(ValueError, match="alpha must be a 1D arrayLike."):
            BDFScheme([[1, 2], [3, 4]], 1, 2)

    def test_invalid_beta_type(self):
        # beta type is not strictly checked, but we can test for non-numeric
        with pytest.raises(TypeError):
            BDFScheme([1, -1], "one", 1)

    def test_alpha_not_1d(self):
        with pytest.raises(ValueError, match="alpha must be a 1D arrayLike."):
            BDFScheme(np.array([[1, 2], [3, 4]]), 1, 2)

    # --- n_stages property ---
    def test_n_stages_matches_length_alpha(self):
        alpha = [2, -1, 0.5]
        scheme = BDFScheme(alpha, 1, 2)
        assert scheme.n_stages == len(alpha) - 1

    # --- Predefined schemes ---
    def test_available_schemes(self):
        schemes = BDFScheme.available_schemes()
        expected = ["bdf1", "bdf2", "bdf3", "bdf4", "bdf5", "bdf6"]
        assert schemes == expected

    @pytest.mark.parametrize("name", ["bdf1", "bdf2", "bdf3", "bdf4", "bdf5", "bdf6"])
    def test_predefined_scheme_properties(self, name):
        data = BDFScheme._schemes[name]
        scheme = BDFScheme(data["alpha"], data["beta"], len(data["alpha"]) - 1)
        assert scheme.alpha.tolist() == data["alpha"]
        assert scheme.beta == data["beta"]
        assert scheme.n_stages == len(data["alpha"]) - 1
        assert scheme.order == len(data["alpha"]) - 1

    # --- Edge cases ---
    def test_empty_alpha(self):
        with pytest.raises(ValueError):
            BDFScheme([], 1, 0)

    def test_single_stage(self):
        alpha = [1, -1]
        scheme = BDFScheme(alpha, 1, 1)
        assert scheme.n_stages == 1
        assert scheme.order == 1
