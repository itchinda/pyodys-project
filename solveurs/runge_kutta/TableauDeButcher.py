import numpy as np
import json
import os
from solveurs.runge_kutta.available_butcher_table_data_to_json import available_butcher_table_data_to_json

# Creation du fichier JSON pour les schemas disponibles
available_butcher_table_data_to_json()

# On importe ensuite les donnees du fichier JSON
file_path = os.path.join(os.path.dirname(__file__), 'tableaux_de_butcher_disponibles.json')
with open(file_path, 'r') as f:
    _SCHEMAS_DATA = json.load(f)

# On convertit en NumPy arrays
for scheme_data in _SCHEMAS_DATA.values():
    scheme_data['A'] = np.array(scheme_data['A'])
    scheme_data['B'] = np.array(scheme_data['B'])
    scheme_data['C'] = np.array(scheme_data['C'])

class TableauDeButcher(object):
    SCHEMAS_DISPONIBLES = list(_SCHEMAS_DATA.keys())

    def __init__(self, A, B, C, ordre: int):
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise TypeError("A doit être une matrice numpy de dimension 2.")
        if not isinstance(B, np.ndarray) or B.ndim not in [1, 2]:
            raise TypeError("B doit être un vecteur numpy de dimension 1 ou 2.")
        if not isinstance(C, np.ndarray) or C.ndim not in [1, 2]:
            raise TypeError("C doit être un vecteur numpy de dimension 1 ou 2.")
        if not isinstance(ordre, (int, float)):
            raise TypeError("Le paramètre 'ordre' doit être de type entier ou réel.")
        if not all(np.issubdtype(arr.dtype, np.number) for arr in [A, B, C]):
            raise TypeError("Les tableaux A, B, et C doivent contenir des nombres (entiers ou réels).")
            
        rows_A, cols_A = A.shape
        if rows_A != cols_A:
            raise ValueError("La matrice A doit être carrée.")
        s = rows_A  # le nombre de niveaux
        
        if B.ndim == 1:
            if B.size != s:
                raise ValueError(f"Le vecteur B doit avoir une taille de {s}.")
        elif B.ndim == 2:
            rows_B, cols_B = B.shape
            if cols_B != s or rows_B not in [1, 2]:
                raise ValueError(f"Le vecteur B doit avoir {s} colonnes et 1 ou 2 lignes.")
        
        if C.ndim == 1:
            if C.size != s:
                raise ValueError(f"Le vecteur C doit avoir une taille de {s}.")
        elif C.ndim == 2:
            rows_C, cols_C = C.shape
            if rows_C != s or cols_C != 1:
                raise ValueError(f"Le vecteur C doit être une matrice {s}x1.")
        
        self.A = A
        self.B = B
        self.C = C
        self.ordre = ordre

    @property
    def nombre_de_niveaux(self):
        return self.A.shape[0]

    @property
    def est_explicite(self):
        return np.all(np.triu(self.A, 0) == 0)

    @property
    def est_implicite(self):
        return not self.est_explicite

    @property
    def est_diagonalement_implicite(self):
        if not np.all(np.triu(self.A, 1) == 0):
            return False
        
        diagonale = np.diag(self.A)
        return np.all(diagonale != 0) and np.allclose(diagonale, diagonale[0])

    def __str__(self):
        s = self.A.shape[0]
        output = f"Schéma de Runge-Kutta d'ordre {self.ordre}\n\n"
        
        # Determine max width for numbers to ensure alignment
        max_width = max(len(f'{x:.16g}') for x in np.concatenate([self.A.flatten(), self.B.flatten(), self.C.flatten()]))
        
        # Print the C vector and A matrix
        for i in range(s):
            c_val = f'{self.C[i]:>{max_width}.16g}'
            a_row = ' '.join(f'{val:>{max_width}.16g}' for val in self.A[i])
            output += f"{c_val} | {a_row}\n"
        
        # Print the line separating A and B
        separator = '-' * (max_width + 1) + '+' + '-' * (s * (max_width + 1) + 1)
        output += f"{separator}\n"
        
        # Print the B vectors
        if self.B.ndim == 1:
            b_row = ' '.join(f'{val:>{max_width}.16g}' for val in self.B)
            output += f"{'':>{max_width}}   {b_row}\n"
        elif self.B.ndim == 2:
            for row in self.B:
                b_row = ' '.join(f'{val:>{max_width}.16g}' for val in row)
                output += f"{'':>{max_width}}   {b_row}\n"
                
        return output

    @classmethod
    def par_nom(cls, nom):
        nom_lower = nom.lower()
        if nom_lower not in _SCHEMAS_DATA:
            raise ValueError(f"Nom de schema inconnu: '{nom}'. Voici les schemas disponibles: {', '.join(cls.SCHEMAS_DISPONIBLES)}")
        
        scheme_data = _SCHEMAS_DATA[nom_lower]
        return cls(scheme_data['A'], scheme_data['B'], scheme_data['C'], scheme_data['ordre'])
    
    @classmethod # Alias en anglais de par_nom
    def from_name(cls, name):
        return cls.par_nom(name)