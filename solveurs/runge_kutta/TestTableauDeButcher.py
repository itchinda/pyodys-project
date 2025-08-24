import unittest
import numpy as np
from TableauDeButcher import TableauDeButcher

class TestTableauDeButcher(unittest.TestCase):
    
    # --- Cas de test 1: Données valides: Aucune exception attendue ---
    
    def test_valide_schema_explicite(self):
        """Teste avec un schéma RK4 explicite valide."""
        A = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        B = np.array([1/6, 1/3, 1/3, 1/6])
        C = np.array([0.0, 0.5, 0.5, 1.0])
        try:
            schema = TableauDeButcher(A, B, C, 4)
            self.assertIsInstance(schema, TableauDeButcher)
        except Exception as e:
            self.fail(f"La tentative de construction d'un tableau de Butcher avec des paramètres valides a échoué : {e}")

    def test_valid_sdirk_schema(self):
        """Teste avec un schéma SDIRK valide."""
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
        try:
            schema = TableauDeButcher(A, B, C, 4)
            self.assertIsInstance(schema, TableauDeButcher)
        except Exception as e:
            self.fail(f"La tentative de construction d'un tableau de Butcher de SDIRK avec des paramètres valides a échoué : {e}")
            
    # --- Cas de test 2: Types des entrées invalides (devrait lancer une exception de type TypeError) ---
    
    def test_type_invalide_A(self):
        """Teste le cas où A n'est pas une matrice numpy."""
        with self.assertRaisesRegex(TypeError, "A doit être une matrice numpy de dimension 2."):
            TableauDeButcher([1, 2], np.array([1]), np.array([1]), 1)
            
    def test_type_invalide_B(self):
        """Teste le cas où B n'est pas un vecteur/matrice numpy."""
        with self.assertRaisesRegex(TypeError, "B doit être un vecteur numpy de dimension 1 ou 2."):
            TableauDeButcher(np.array([[1]]), [1], np.array([1]), 1)
            
    def test_type_invalide_C(self):
        """Teste le cas où C n'est pas un vecteur numpy."""
        with self.assertRaisesRegex(TypeError, "C doit être un vecteur numpy de dimension 1 ou 2."):
            TableauDeButcher(np.array([[1]]), np.array([1]), [1], 1)
    
    def test_type_invalide_ordre(self):
        """Teste le cas où 'ordre' n'est pas un nombre."""
        with self.assertRaisesRegex(TypeError, "Le paramètre 'ordre' doit être de type entier ou réel."):
            TableauDeButcher(np.array([[1]]), np.array([1]), np.array([1]), "un")
            
    def test_entrees_non_reelles_ou_entieres_A(self):
        """Teste une matrice avec des données non numériques."""
        A = np.array([['a', 'b'], ['c', 'd']])
        with self.assertRaisesRegex(TypeError, "Les tableaux A, B, et C doivent contenir des nombres"):
            TableauDeButcher(A, np.array([1, 2]), np.array([1, 2]), 2)
            
    # --- Cas de test 3: Dimensions incompatibles (devrait lancer une exception de type ValueError) ---
    
    def test_matrice_A_non_carree(self):
        """Teste le cas où A n'est pas une matrice carrée."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "La matrice A doit être carrée."):
            TableauDeButcher(A, np.array([1, 2, 3]), np.array([1, 2, 3]), 3)

    def test_dimensions_incompatibles(self):
        """Teste la situation où les dimensions de B et C sont incompatibles avec celles de A."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([1, 2, 3])
        C = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, "Le vecteur B doit avoir une taille de 2."):
            TableauDeButcher(A, B, C, 2)
            
    def test_incorrect_B_shape(self):
        """Teste la construction du tableau avec des dimensions invalides pour B."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4], [5, 6]])
        C = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, "Le vecteur B doit avoir 2 colonnes et 1 ou 2 lignes."):
            TableauDeButcher(A, B, C, 2)
            
    # --- Cas de test 4: tests des propriétés ---
    
    def setUp(self):
        """Fixtures pour tester les propriétés. Ceci permet d'éviter de reconstruire les objets."""
        A_explicit = np.array([[0.0, 0.0], [0.5, 0.0]])
        B_explicit = np.array([0.5, 0.5])
        C_explicit = np.array([0.0, 0.5])
        self.explicit_schema = TableauDeButcher(A_explicit, B_explicit, C_explicit, 2)
        
        A_implicit = np.array([[0.5, 0.5], [0.5, 0.5]])
        B_implicit = np.array([0.5, 0.5])
        C_implicit = np.array([0.5, 0.5])
        self.implicit_schema = TableauDeButcher(A_implicit, B_implicit, C_implicit, 2)
        
        A_sdirk = np.array([[0.5, 0.0], [0.5, 0.5]])
        B_sdirk = np.array([0.5, 0.5])
        C_sdirk = np.array([0.5, 1.0])
        self.sdirk_schema = TableauDeButcher(A_sdirk, B_sdirk, C_sdirk, 2)

    def test_nombre_de_niveaux_property(self):
        self.assertEqual(self.explicit_schema.nombre_de_niveaux, 2)
        self.assertEqual(self.implicit_schema.nombre_de_niveaux, 2)

    def test_est_explicite_property(self):
        self.assertTrue(self.explicit_schema.est_explicite)
        self.assertFalse(self.implicit_schema.est_explicite)

    def test_est_implicite_property(self):
        self.assertFalse(self.explicit_schema.est_implicite)
        self.assertTrue(self.implicit_schema.est_implicite)

    def test_est_diagonalement_implicite_property(self):
        self.assertTrue(self.sdirk_schema.est_diagonalement_implicite)
        self.assertFalse(self.explicit_schema.est_diagonalement_implicite)
        self.assertFalse(self.implicit_schema.est_diagonalement_implicite)

# Note: All tests from TestParNom have been integrated into this class
class TestParNom(unittest.TestCase):
    
    def test_des_proprietes_des_schemas_predefinis(self):
        """Vérifie que par_nom instancie correctement les objets valides pour tous les schémas prédéfinis."""
        schemes_to_test = TableauDeButcher.SCHEMAS_DISPONIBLES
        for nom in schemes_to_test:
            try:
                tableau = TableauDeButcher.par_nom(nom)
                self.assertIsInstance(tableau, TableauDeButcher)
            except Exception as e:
                self.fail(f"Impossible de créer une instance du schéma '{nom}'. Exception lancée: {e}")

    def test_des_proprietes_euler_explicite(self):
        """Vérification des propriétés pour le schéma d'Euler Explicite."""
        tableau = TableauDeButcher.par_nom('euler_explicite')
        self.assertEqual(tableau.nombre_de_niveaux, 1)
        self.assertTrue(tableau.est_explicite)
        self.assertFalse(tableau.est_implicite)
        self.assertFalse(tableau.est_diagonalement_implicite)

    def test_des_proprietes_euler_implicite(self):
        """Vérification des propriétés pour le schéma d'Euler Implicite."""
        tableau = TableauDeButcher.par_nom('euler_implicite')
        self.assertEqual(tableau.nombre_de_niveaux, 1)
        self.assertTrue(tableau.est_implicite)
        self.assertFalse(tableau.est_explicite)
        self.assertTrue(tableau.est_diagonalement_implicite)

    def test_des_proprietes_rk4(self):
        """Vérification des propriétés pour le schéma de Runge-Kutta d'ordre 4 Explicite."""
        tableau = TableauDeButcher.par_nom('rk4')
        self.assertEqual(tableau.nombre_de_niveaux, 4)
        self.assertTrue(tableau.est_explicite)
        self.assertFalse(tableau.est_implicite)
        self.assertFalse(tableau.est_diagonalement_implicite)

    def test_des_proprietes_sdirk_ordre3_predefini(self):
        """Vérification des propriétés pour le schéma SDIRK d'ordre 3 à 4 niveaux prédéfini."""
        tableau = TableauDeButcher.par_nom('sdirk_ordre3_predefini')
        self.assertEqual(tableau.nombre_de_niveaux, 4)
        self.assertTrue(tableau.est_implicite)
        self.assertFalse(tableau.est_explicite)
        self.assertTrue(tableau.est_diagonalement_implicite)
        
    def test_nom_inconnu(self):
        """Vérifie que l'exception de type ValueError est lancée lorsque le type est inconnu."""
        with self.assertRaisesRegex(ValueError, r"Nom de schema inconnu: 'non_existent'"):
            TableauDeButcher.par_nom('non_existent')

    def test_insensibilite_a_la_casse(self):
        """Vérifie que le nom est insensible à la casse."""
        tableau = TableauDeButcher.par_nom('EULER_EXPLICITE')
        self.assertIsInstance(tableau, TableauDeButcher)
        self.assertEqual(tableau.ordre, 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)