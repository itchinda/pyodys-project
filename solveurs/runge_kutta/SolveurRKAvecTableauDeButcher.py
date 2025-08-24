from systemes.EDOs import EDOs
from .TableauDeButcher import TableauDeButcher
import numpy as np 
from scipy.linalg import lu, inv

class SolveurRKAvecTableauDeButcher(object):
    def __init__(self, tableau_de_butcher=TableauDeButcher.par_nom('rk4') ):
        if not isinstance(tableau_de_butcher, TableauDeButcher):
            raise TypeError("On devrait passer un objet de type TableauDeButcher.")
        self.tableau_de_butcher = tableau_de_butcher

    def _effectueUnPasDeTempsRKAvecTableauDeButcher(self, F:EDOs, tn: float, delta_t: float, U_np: np.ndarray):
        """
        Effectue un pas de Runge-Kutta en utilisant un tableau de  Butcher
        Args:
            F: le systeme a resoudre f = dU/dt.
            tn (float): le temps courrant.
            delta_t (float): Le pas de temps.
            U_np (np.ndarray): Le vector solution au temps tn.

        Returns:
            tuple: A tuple contenant:
                Un (np.ndarray): le vecteur solution au temps t(n+1).
                U_pred (np.ndarray): The predicted state vector.
                newton_pas_content (bool): Flag indiquant si Newton a converge ou pas.
        """
        nombre_de_niveaux = self.tableau_de_butcher.A.shape[1]
        nombre_d_equations = len(U_np)

        a = self.tableau_de_butcher.A
        c = self.tableau_de_butcher.C
        d = np.zeros_like(a[0,:])

        avec_prediction = self.tableau_de_butcher.B.shape[0] == 2
        if avec_prediction:
            b = self.tableau_de_butcher.B[0, :]
            d = self.tableau_de_butcher.B[1, :]
        else:
            b = self.tableau_de_butcher.B

        newton_pas_content = False
        U_chap = np.zeros((nombre_d_equations, nombre_de_niveaux))
        valeur_f = np.zeros((nombre_d_equations, nombre_de_niveaux))

        max_iteration_newton = 25
        min_iteration_newton = 4
        abs_tolerance = 1e-12
        rel_tolerance = 1e-6

        U_n = np.copy(U_np)
        U_pred = np.zeros_like(U_np)
        if avec_prediction:
            U_pred = np.copy(U_np)

        I = np.eye(nombre_d_equations)

        for k in range(nombre_de_niveaux):
            U_chap_k = U_np + np.sum(a[k, :k] * valeur_f[:, :k], axis=1)

            if a[k, k] != 0.0:
                tn_k = tn + c[k] * delta_t
                delta_t_x_akk = delta_t * a[k, k]
                U_newton = np.copy(U_chap_k)

                J = F.jacobien(tn_k, U_newton)
                A = I - delta_t_x_akk * J

                for iteration_newton in range(max_iteration_newton):
                    residu = U_newton - (U_chap_k + delta_t_x_akk * F.evalue(tn_k, U_newton))

                    # On resoud A * delta = residu
                    try:
                        delta = np.linalg.solve(A, residu)
                    except np.linalg.LinAlgError:
                        newton_pas_content = True
                        return U_n, U_pred, newton_pas_content
                    # Mettre a jour U_newton
                    U_newton = U_newton - delta

                    # verifie la convergence
                    convergence = (np.linalg.norm(delta) <= abs_tolerance) and (np.linalg.norm(delta / (U_newton + 1e-12)) <= rel_tolerance)
                    if convergence and iteration_newton >= min_iteration_newton:
                        break
                else:
                    newton_pas_content = True
                    return U_n, U_pred, newton_pas_content

                U_chap[:, k] = U_newton
            else:
                tn_k = tn + c[k] * delta_t
                U_chap[:, k] = U_chap_k

            valeur_f[:, k] = delta_t * F.evalue(tn_k, U_chap[:, k])

            U_n += b[k] * valeur_f[:, k]
            if avec_prediction:
                U_pred += d[k] * valeur_f[:, k]

        return U_n, U_pred, newton_pas_content

    def _testePasDeTemps(self):
        pass

    def resoud(self, systeme_edo:EDOs, pas_de_temps: float, nb_pas_de_temps_max: int):
        """
        Solves the ODE system by performing a series of time steps.

        Args:
            F (EDOs): The ODE system to solve.
            pas_de_temps (float): The step size.
            nb_pas_de_temps_max (int): The maximum number of steps.
            tn (float): The initial time.
            Un (np.ndarray): The initial state vector.

        Returns:
            tuple: A tuple containing lists of the solution times and state vectors.
                - temps (list): A list of the time points.
                - solutions (list): A list of the solution vectors at each time point.
        """
        temps = [systeme_edo.temps_initial]
        solutions = [systeme_edo.condition_initiale]
        
        U_courant = np.copy(systeme_edo.condition_initiale)
        temps_courant = systeme_edo.temps_initial
        
        for i in range(nb_pas_de_temps_max):
            # on fait un pas de temps
            U_n_plus_1, _, newton_pas_content = self._effectueUnPasDeTempsRKAvecTableauDeButcher(
                systeme_edo, temps_courant, pas_de_temps, U_courant
            )
            
            # Verifie la convergence de Newton
            if newton_pas_content:
                print(f"L'algorithme de Newton a echoue a converger au pas de tempa {i+1}. Arret de la simulation.")
                break
                
            # Mise a jour de la solution courante et du temps courant
            U_courant = U_n_plus_1
            temps_courant += pas_de_temps
            
            # On sauvegarde la solution du pas de temps dans le vecteur solution
            temps.append(temps_courant)
            solutions.append(U_courant)

        return np.array(temps), np.array(solutions)

    def solve(self, systeme_edo:EDOs, pas_de_temps: float, nb_pas_de_temps_max: int):
        """
        Alias de resoud().
        """
        return self.resoud(systeme_edo, pas_de_temps, nb_pas_de_temps_max)



