import numpy as np
import pytest
from pyodys import TableauDeButcher

# --- Cas de test 1: Données valides ---
class TestTableauDeButcher:

    def test_valide_schema_explicite(self):
        A = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        B = np.array([1/6, 1/3, 1/3, 1/6])
        C = np.array([0.0, 0.5, 0.5, 1.0])
        schema = TableauDeButcher(A, B, C, 4)
        assert isinstance(schema, TableauDeButcher)

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
        schema = TableauDeButcher(A, B, C, 4)
        assert isinstance(schema, TableauDeButcher)

# --- Cas de test 2: Types invalides ---
def test_type_invalide_A():
    with pytest.raises(TypeError, match="A doit être une matrice numpy de dimension 2."):
        TableauDeButcher([1, 2], np.array([1]), np.array([1]), 1)

def test_type_invalide_B():
    with pytest.raises(TypeError, match="B doit être un vecteur numpy de dimension 1 ou 2."):
        TableauDeButcher(np.array([[1]]), [1], np.array([1]), 1)

def test_type_invalide_C():
    with pytest.raises(TypeError, match="C doit être un vecteur numpy de dimension 1 ou 2."):
        TableauDeButcher(np.array([[1]]), np.array([1]), [1], 1)

def test_type_invalide_ordre():
    with pytest.raises(TypeError, match="Le paramètre 'ordre' doit être de type entier ou réel."):
        TableauDeButcher(np.array([[1]]), np.array([1]), np.array([1]), "un")

def test_entrees_non_reelles_ou_entieres_A():
    A = np.array([['a', 'b'], ['c', 'd']])
    with pytest.raises(TypeError, match="Les tableaux A, B, et C doivent contenir des nombres"):
        TableauDeButcher(A, np.array([1, 2]), np.array([1, 2]), 2)

# --- Cas de test 3: Dimensions incompatibles ---
def test_matrice_A_non_carree():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="La matrice A doit être carrée."):
        TableauDeButcher(A, np.array([1, 2, 3]), np.array([1, 2, 3]), 3)

def test_dimensions_incompatibles():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([1, 2, 3])
    C = np.array([1, 2])
    with pytest.raises(ValueError, match="Le vecteur B doit avoir une taille de 2."):
        TableauDeButcher(A, B, C, 2)

def test_incorrect_B_shape():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    C = np.array([1, 2])
    with pytest.raises(ValueError, match="Le vecteur B doit avoir 2 colonnes et 1 ou 2 lignes."):
        TableauDeButcher(A, B, C, 2)

# --- Cas de test 4: Propriétés ---
@pytest.fixture
def schemas():
    A_explicit = np.array([[0.0, 0.0], [0.5, 0.0]])
    B_explicit = np.array([0.5, 0.5])
    C_explicit = np.array([0.0, 0.5])
    explicit_schema = TableauDeButcher(A_explicit, B_explicit, C_explicit, 2)

    A_implicit = np.array([[0.5, 0.5], [0.5, 0.5]])
    B_implicit = np.array([0.5, 0.5])
    C_implicit = np.array([0.5, 0.5])
    implicit_schema = TableauDeButcher(A_implicit, B_implicit, C_implicit, 2)

    A_sdirk = np.array([[0.5, 0.0], [0.5, 0.5]])
    B_sdirk = np.array([0.5, 0.5])
    C_sdirk = np.array([0.5, 1.0])
    sdirk_schema = TableauDeButcher(A_sdirk, B_sdirk, C_sdirk, 2)

    return explicit_schema, implicit_schema, sdirk_schema

def test_nombre_de_niveaux_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert explicit_schema.nombre_de_niveaux == 2
    assert implicit_schema.nombre_de_niveaux == 2

def test_est_explicite_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert explicit_schema.est_explicite
    assert not implicit_schema.est_explicite

def test_est_implicite_property(schemas):
    explicit_schema, implicit_schema, _ = schemas
    assert not explicit_schema.est_implicite
    assert implicit_schema.est_implicite

def test_est_diagonalement_implicite_property(schemas):
    explicit_schema, _, sdirk_schema = schemas
    assert sdirk_schema.est_diagonalement_implicite
    assert not explicit_schema.est_diagonalement_implicite

# --- Tests par_nom ---
class TestParNom:

    def test_des_proprietes_des_schemas_predefinis(self):
        for nom in TableauDeButcher.SCHEMAS_DISPONIBLES:
            tableau = TableauDeButcher.par_nom(nom)
            assert isinstance(tableau, TableauDeButcher)

    def test_des_proprietes_euler_explicite(self):
        tableau = TableauDeButcher.par_nom('euler_explicite')
        assert tableau.nombre_de_niveaux == 1
        assert tableau.est_explicite
        assert not tableau.est_implicite
        assert not tableau.est_diagonalement_implicite

    def test_des_proprietes_euler_implicite(self):
        tableau = TableauDeButcher.par_nom('euler_implicite')
        assert tableau.nombre_de_niveaux == 1
        assert tableau.est_implicite
        assert not tableau.est_explicite
        assert tableau.est_diagonalement_implicite

    def test_des_proprietes_rk4(self):
        tableau = TableauDeButcher.par_nom('rk4')
        assert tableau.nombre_de_niveaux == 4
        assert tableau.est_explicite
        assert not tableau.est_implicite
        assert not tableau.est_diagonalement_implicite

    def test_des_proprietes_sdirk_ordre3_predefini(self):
        tableau = TableauDeButcher.par_nom('sdirk_norsett_thomson_34')
        assert tableau.nombre_de_niveaux == 4
        assert tableau.est_implicite
        assert not tableau.est_explicite
        assert tableau.est_diagonalement_implicite

    def test_nom_inconnu(self):
        with pytest.raises(ValueError, match=r"Nom de schema inconnu: 'non_existent'"):
            TableauDeButcher.par_nom('non_existent')

    def test_insensibilite_a_la_casse(self):
        tableau = TableauDeButcher.par_nom('EULER_EXPLICITE')
        assert isinstance(tableau, TableauDeButcher)
        assert tableau.ordre == 1
