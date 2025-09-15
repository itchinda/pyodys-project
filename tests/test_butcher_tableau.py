import numpy as np
import pytest
from pyodys import ButcherTableau


# --- Cas de test 1: Données valides ---
class TestButcherTableau:

    def test_valide_schema_explicite(self):
        A = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        B = np.array([1/6, 1/3, 1/3, 1/6])
        C = np.array([0.0, 0.5, 0.5, 1.0])
        schema = ButcherTableau(A, B, C, 4)
        assert isinstance(schema, ButcherTableau)

    def test_valid_sdirk_schema(self):
        alpha = 5/6
        A = np.array([
            [alpha, 0, 0, 0],
            [-15/26, alpha, 0, 0],
            [215/54, -130/27, alpha, 0],
            [4007/6075, -31031/24300, -133/2700, alpha]
        ])
        B = np.array([
            [32/75, 169/300, 1/100, 0],
            [61/150, 2197/2100, 19/100, -9/14]
        ])
        C = np.array([alpha, 10/39, 0, 1/6])
        schema = ButcherTableau(A, B, C, 4, 3)
        assert isinstance(schema, ButcherTableau)

# --- Cas de test 2: Types invalides ---
def test_invalid_type_A():
    with pytest.raises(TypeError, match="A doit être une matrice numpy de dimension 2."):
        ButcherTableau([1, 2], np.array([1]), np.array([1]), 1)

def test_invalid_type_B():
    with pytest.raises(TypeError, match="B doit être un vecteur numpy de dimension 1 ou 2."):
        ButcherTableau(np.array([[1]]), [1], np.array([1]), 1)

def test_invalid_type_C():
    with pytest.raises(TypeError, match="C doit être un vecteur numpy de dimension 1 ou 2."):
        ButcherTableau(np.array([[1]]), np.array([1]), [1], 1)

def test_invalid_type_order():
    with pytest.raises(TypeError, match="Le paramètre 'order' doit être de type entier ou réel."):
        ButcherTableau(np.array([[1]]), np.array([1]), np.array([1]), "un")

def test_entrees_non_reelles_ou_entieres_A():
    A = np.array([['a', 'b'], ['c', 'd']])
    with pytest.raises(TypeError, match="Les tableaux A, B, et C doivent contenir des nombres"):
        ButcherTableau(A, np.array([1, 2]), np.array([1, 2]), 2)

# --- Cas de test 3: Dimensions incompatibles ---
def test_matrice_A_non_carree():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="La matrice A doit être carrée."):
        ButcherTableau(A, np.array([1, 2, 3]), np.array([1, 2, 3]), 3)

def test_dimensions_incompatibles():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([1, 2, 3])
    C = np.array([1, 2])
    with pytest.raises(ValueError, match="Le vecteur B doit avoir une taille de 2."):
        ButcherTableau(A, B, C, 2)

def test_incorrect_B_shape():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    C = np.array([1, 2])
    with pytest.raises(ValueError, match="Le vecteur B doit avoir 2 colonnes et 1 ou 2 lignes."):
        ButcherTableau(A, B, C, 2, 1)

def test_check_consistency_pass():
    # A matches C
    A = np.array([[0.0, 0.0], [0.5, 0.0]])
    B = np.array([0.5, 0.5])
    C = np.sum(A, axis=1)
    
    # Should not raise
    tableau = ButcherTableau(A, B, C, 2, check_consistency=True)
    assert np.allclose(tableau.C, np.sum(tableau.A, axis=1))

def test_check_consistency_fail():
    # A does NOT match C
    A = np.array([[0.0, 0.0], [0.5, 0.0]])
    B = np.array([0.5, 0.75])
    C = np.array([0.0, 0.5])
    
    with pytest.raises(ValueError, match="Sum of B coefficients does not match 1."):
        ButcherTableau(A, B, C, 2, check_consistency=True)

def test_check_consistency_default():
    # By default, check_consistency=False
    A = np.array([[0.0, 0.0], [0.5, 0.0]])
    B = np.array([0.5, 0.5])
    C = np.array([0.0, 1.0])  # mismatch
    tableau = ButcherTableau(A, B, C, 2)  # Should not raise
    assert np.allclose(tableau.C, C)

@pytest.fixture
def tableau_examples():
    """Return a dict of artificial tableaus to test all properties."""
    # --- Positive cases ---
    # Explicit RK
    A_exp = np.array([[0.0, 0.0], [0.5, 0.0]])
    B_exp = np.array([0.5, 0.5])
    C_exp = np.sum(A_exp, axis=1)
    explicit = ButcherTableau(A_exp, B_exp, C_exp, 2)

    # Implicit RK
    A_imp = np.array([[0.5, 0.5], [0.5, 0.5]])
    B_imp = np.array([0.5, 0.5])
    C_imp = np.sum(A_imp, axis=1)
    implicit = ButcherTableau(A_imp, B_imp, C_imp, 2)

    # DIRK
    A_dirk = np.array([[0.5, 0.0], [0.3, 0.6]])
    B_dirk = np.array([0.5, 0.5])
    C_dirk = np.sum(A_dirk, axis=1)
    dirk = ButcherTableau(A_dirk, B_dirk, C_dirk, 2)

    # SDIRK
    A_sdirk = np.array([[0.6, 0.0], [0.4, 0.6]])
    B_sdirk = np.array([0.5, 0.5])
    C_sdirk = np.sum(A_sdirk, axis=1)
    sdirk = ButcherTableau(A_sdirk, B_sdirk, C_sdirk, 2)

    # ESDIRK
    A_esdirk = np.array([[0.0, 0.0], [0.3, 0.6]])
    B_esdirk = np.array([0.4, 0.6])
    C_esdirk = np.sum(A_esdirk, axis=1)
    esdirk = ButcherTableau(A_esdirk, B_esdirk, C_esdirk, 2)

    # Embedded
    A_emb = np.array([[0.5, 0.0], [0.3, 0.6]])
    B_emb = np.array([[0.4, 0.6], [0.25, 0.75]])
    C_emb = np.sum(A_emb, axis=1)
    embedded = ButcherTableau(A_emb, B_emb, C_emb, 2, 1)

    # --- Negative cases ---
    # Not strictly lower-triangular (fails explicit)
    A_non_explicit = np.array([[0.1, 0.0], [0.3, 0.6]])
    B_non_explicit = np.array([0.5, 0.5])
    C_non_explicit = np.sum(A_non_explicit, axis=1)
    non_explicit = ButcherTableau(A_non_explicit, B_non_explicit, C_non_explicit, 2)

    # Diagonal zeros (fails DIRK)
    A_zero_diag = np.array([[0.0, 0.0], [0.3, 0.0]])
    B_zero_diag = np.array([0.5, 0.5])
    C_zero_diag = np.sum(A_zero_diag, axis=1)
    zero_diag = ButcherTableau(A_zero_diag, B_zero_diag, C_zero_diag, 2)

    # --- SDIRK edge cases ---
    # Diagonal entries not all equal
    A_sdirk_wrong_diag = np.array([[0.6, 0.0], [0.5, 0.7]])
    B_sdirk_wrong_diag = np.array([0.5, 0.5])
    C_sdirk_wrong_diag = np.sum(A_sdirk_wrong_diag, axis=1)
    sdirk_wrong_diag = ButcherTableau(A_sdirk_wrong_diag, B_sdirk_wrong_diag, C_sdirk_wrong_diag, 2)

    # --- ESDIRK edge cases ---
    # First stage not zero
    A_esdirk_first_nonzero = np.array([[0.1, 0.0], [0.3, 0.6]])
    B_esdirk_first_nonzero = np.array([0.4, 0.6])
    C_esdirk_first_nonzero = np.sum(A_esdirk_first_nonzero, axis=1)
    esdirk_first_nonzero = ButcherTableau(A_esdirk_first_nonzero, B_esdirk_first_nonzero, C_esdirk_first_nonzero, 2)

    # Remaining diagonals not equal
    A_esdirk_unequal_diag = np.array([[0.0, 0.0, 0.0], [0.3, 0.6, 0.0], [0.1, 0.5, 0.9]])
    B_esdirk_unequal_diag = np.array([0.4, 0.6, 0.6])
    C_esdirk_unequal_diag = np.sum(A_esdirk_unequal_diag, axis=1)
    esdirk_unequal_diag = ButcherTableau(A_esdirk_unequal_diag, B_esdirk_unequal_diag, C_esdirk_unequal_diag, 2)


    return {
        "explicit": explicit,
        "implicit": implicit,
        "dirk": dirk,
        "sdirk": sdirk,
        "esdirk": esdirk,
        "embedded": embedded,
        "non_explicit": non_explicit,
        "zero_diag": zero_diag,
        "sdirk_wrong_diag": sdirk_wrong_diag,
        "esdirk_first_nonzero": esdirk_first_nonzero,
        "esdirk_unequal_diag": esdirk_unequal_diag
    }

# --- Test positive cases ---
def test_n_stages(tableau_examples):
    for name, t in tableau_examples.items():
        assert t.n_stages == t.A.shape[0]

def test_is_explicit(tableau_examples):
    assert tableau_examples["explicit"].is_explicit
    assert not tableau_examples["implicit"].is_explicit
    assert not tableau_examples["dirk"].is_explicit
    assert not tableau_examples["sdirk"].is_explicit
    assert not tableau_examples["esdirk"].is_explicit

def test_is_implicit(tableau_examples):
    assert not tableau_examples["explicit"].is_implicit
    assert tableau_examples["implicit"].is_implicit
    assert tableau_examples["dirk"].is_implicit
    assert tableau_examples["sdirk"].is_implicit
    assert tableau_examples["esdirk"].is_implicit

def test_is_diagonally_implicit(tableau_examples):
    assert not tableau_examples["explicit"].is_diagonally_implicit
    assert not tableau_examples["implicit"].is_diagonally_implicit
    assert tableau_examples["dirk"].is_diagonally_implicit
    assert tableau_examples["sdirk"].is_diagonally_implicit
    assert tableau_examples["esdirk"].is_diagonally_implicit

def test_is_sdirk(tableau_examples):
    assert not tableau_examples["explicit"].is_sdirk
    assert not tableau_examples["dirk"].is_sdirk
    assert tableau_examples["sdirk"].is_sdirk
    assert not tableau_examples["esdirk"].is_sdirk

def test_is_esdirk(tableau_examples):
    assert tableau_examples["esdirk"].is_esdirk
    assert not tableau_examples["explicit"].is_esdirk
    assert not tableau_examples["sdirk"].is_esdirk

# --- SDIRK failures ---
def test_sdirk_wrong_diag(tableau_examples):
    sdirk_wrong_diag = tableau_examples["sdirk_wrong_diag"]
    assert sdirk_wrong_diag.is_diagonally_implicit
    assert not sdirk_wrong_diag.is_sdirk

# --- ESDIRK failures ---
def test_esdirk_first_stage_nonzero(tableau_examples):
    esdirk_first_nonzero = tableau_examples["esdirk_first_nonzero"]
    assert esdirk_first_nonzero.is_diagonally_implicit
    assert not esdirk_first_nonzero.is_esdirk

def test_esdirk_remaining_diag_unequal(tableau_examples):
    esdirk_unequal_diag = tableau_examples["esdirk_unequal_diag"]
    assert np.any(np.diag(esdirk_unequal_diag.A)[1:] != np.diag(esdirk_unequal_diag.A)[1])
    assert not esdirk_unequal_diag.is_esdirk

def test_with_prediction(tableau_examples):
    assert tableau_examples["embedded"].with_prediction
    assert not tableau_examples["sdirk"].with_prediction

def test_sum_of_A_matches_C(tableau_examples):
    for name, t in tableau_examples.items():
        np.testing.assert_allclose(np.sum(t.A, axis=1), t.C, rtol=1e-14, atol=1e-14)

def test_info(tableau_examples):
    assert "ESDIRK" in tableau_examples["esdirk"].info()
    assert "SDIRK" in tableau_examples["sdirk"].info()
    assert "DIRK" in tableau_examples["dirk"].info()
    assert "Explicit RK" in tableau_examples["explicit"].info()
    assert "Implicit RK" in tableau_examples["implicit"].info()
    assert "Embedded: Yes" in tableau_examples["embedded"].info()

# --- Negative / failure cases ---
def test_non_explicit_is_not_explicit(tableau_examples):
    assert not tableau_examples["non_explicit"].is_explicit

def test_zero_diag_is_not_dirk(tableau_examples):
    assert not tableau_examples["zero_diag"].is_diagonally_implicit


# --- Tests par_nom ---
class TestParNom:

    def test_des_proprietes_des_schemas_predefinis(self):
        for nom in ButcherTableau.available_schemes():
            tableau = ButcherTableau.par_nom(nom)
            assert isinstance(tableau, ButcherTableau)

    def test_des_proprietes_erk1(self):
        tableau = ButcherTableau.par_nom('erk1')
        assert tableau.n_stages == 1
        assert tableau.is_explicit
        assert not tableau.is_implicit
        assert not tableau.is_diagonally_implicit

    def test_des_proprietes_sdirk1(self):
        tableau = ButcherTableau.par_nom('sdirk1')
        assert tableau.n_stages == 1
        assert tableau.is_implicit
        assert not tableau.is_explicit
        assert tableau.is_diagonally_implicit
        assert tableau.is_sdirk

    def test_des_proprietes_erk4(self):
        tableau = ButcherTableau.par_nom('erk4')
        assert tableau.n_stages == 4
        assert tableau.is_explicit
        assert not tableau.is_implicit
        assert not tableau.is_diagonally_implicit

    def test_des_proprietes_sdirk_order3_predefini(self):
        tableau = ButcherTableau.par_nom('sdirk_norsett_thomson_34')
        assert tableau.n_stages == 4
        assert tableau.is_implicit
        assert not tableau.is_explicit
        assert tableau.is_diagonally_implicit

    def test_nom_inconnu(self):
        with pytest.raises(ValueError, match=r"Nom de schema inconnu: 'non_existent'"):
            ButcherTableau.par_nom('non_existent')

    def test_insensibilite_a_la_casse(self):
        tableau = ButcherTableau.par_nom('erk1')
        assert isinstance(tableau, ButcherTableau)
        assert tableau.order == 1

@pytest.mark.parametrize("scheme", [m for m in ButcherTableau.available_schemes()])
def test_sum_of_matrix_a_per_rows_matches_c(scheme):
    """Test that sum of A coefficients per row matches C coefficients."""
    tableau = ButcherTableau.par_nom(scheme)
    sum_a = np.sum(tableau.A, axis=1)
    assert np.allclose(tableau.C, sum_a, rtol=1e-7, atol=1e-7)