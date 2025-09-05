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
        schema = ButcherTableau(A, B, C, 4)
        assert isinstance(schema, ButcherTableau)

# --- Cas de test 2: Types invalides ---
def test_type_invalide_A():
    with pytest.raises(TypeError, match="A doit être une matrice numpy de dimension 2."):
        ButcherTableau([1, 2], np.array([1]), np.array([1]), 1)

def test_type_invalide_B():
    with pytest.raises(TypeError, match="B doit être un vecteur numpy de dimension 1 ou 2."):
        ButcherTableau(np.array([[1]]), [1], np.array([1]), 1)

def test_type_invalide_C():
    with pytest.raises(TypeError, match="C doit être un vecteur numpy de dimension 1 ou 2."):
        ButcherTableau(np.array([[1]]), np.array([1]), [1], 1)

def test_type_invalide_ordre():
    with pytest.raises(TypeError, match="Le paramètre 'ordre' doit être de type entier ou réel."):
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
        ButcherTableau(A, B, C, 2)

# --- Cas de test 4: Propriétés ---
@pytest.fixture
def schemas():
    A_explicit = np.array([[0.0, 0.0], [0.5, 0.0]])
    B_explicit = np.array([0.5, 0.5])
    C_explicit = np.array([0.0, 0.5])
    explicit_schema = ButcherTableau(A_explicit, B_explicit, C_explicit, 2)

    A_implicit = np.array([[0.5, 0.5], [0.5, 0.5]])
    B_implicit = np.array([0.5, 0.5])
    C_implicit = np.array([0.5, 0.5])
    implicit_schema = ButcherTableau(A_implicit, B_implicit, C_implicit, 2)

    A_sdirk = np.array([[0.5, 0.0], [0.5, 0.5]])
    B_sdirk = np.array([0.5, 0.5])
    C_sdirk = np.array([0.5, 1.0])
    sdirk_schema = ButcherTableau(A_sdirk, B_sdirk, C_sdirk, 2)

    return explicit_schema, implicit_schema, sdirk_schema

def test_n_stages_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert explicit_schema.n_stages == 2
    assert implicit_schema.n_stages == 2

def test_is_explicit_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert explicit_schema.is_explicit
    assert not implicit_schema.is_explicit

def test_is_implicit_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert not explicit_schema.is_implicit
    assert implicit_schema.is_implicit

def test_is_diagonally_implicit_property(schemas):
    explicit_schema, _, sdirk_schema = schemas
    assert sdirk_schema.is_diagonally_implicit
    assert not explicit_schema.is_diagonally_implicit

# --- Tests par_nom ---
class TestParNom:

    def test_des_proprietes_des_schemas_predefinis(self):
        for nom in ButcherTableau.AVAILABLE_SCHEMES:
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

    def test_des_proprietes_erk4(self):
        tableau = ButcherTableau.par_nom('erk4')
        assert tableau.n_stages == 4
        assert tableau.is_explicit
        assert not tableau.is_implicit
        assert not tableau.is_diagonally_implicit

    def test_des_proprietes_sdirk_ordre3_predefini(self):
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
        assert tableau.ordre == 1

@pytest.mark.parametrize("scheme", [m for m in ButcherTableau.AVAILABLE_SCHEMES])
def test_sum_of_matrix_a_per_rows_matches_c(scheme):
    """Test that sum of A coefficients per row matches C coefficients."""
    tableau = ButcherTableau.par_nom(scheme)
    sum_a = np.sum(tableau.A, axis=1)
    assert np.allclose(tableau.C, sum_a, rtol=1e-7, atol=1e-7)