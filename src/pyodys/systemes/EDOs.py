from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

class EDOs(ABC):
    """
    Classe abstraite pour les Equations Differentielle Ordinaires
    Chaque classe heritiere devra implemente la methode 'evalue'
    """
    
    def __init__(self, t_init: float, t_final:float, initial_state: ArrayLike):
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.initial_state = np.array(initial_state, dtype=np.float64)
        self.delta = 1e-5  # faible perturbation pour le calcul du jacobien numerique

    @abstractmethod
    def evalue(self, t, u):
        """
        Calcule du/dt = f(t, u).
        Chaque classe heritiere doit implementer la methode 'evalue'.
        """
        raise NotImplementedError("Chaque classe heritiere doit implementer la methode 'evalue'.")
        
    def jacobien(self, t, u):
        """
        Calcule la matrice Jacobienne numerique du systeme d'EDO en utilisant la difference finie centree.
        """
        n = len(u)
        Jacobien = np.zeros((n, n))

        u_temp = np.copy(u)

        for j in range(n):
            # Perturbation a droite
            u_temp[j] += self.delta
            f_u_right = self.evalue(t, u_temp)

            # Perturbation a gauche (on fait -2*delta car on vient juste de faire +delta)
            u_temp[j] -= 2 * self.delta
            f_u_left = self.evalue(t, u_temp)

            # Calcule la j-ieme colonne du Jacobien
            Jacobien[:, j] = (f_u_right - f_u_left) / (2 * self.delta)

            # Restaure la valeur originale de u_temp[j] avant la prochaine etape
            u_temp[j] = u[j] 

        return Jacobien
