from ...systemes.EDOs import EDOs
from .TableauDeButcher import TableauDeButcher
import numpy as np 
from scipy.linalg import lu_factor, lu_solve

def wrms_norm(delta, u, atol=1e-12, rtol=1e-6):
    """
    Weighted Root Mean Square norm.
    delta : vector of Newton update
    u : current Newton iterate
    atol : absolute tolerance
    rtol : relative tolerance
    """
    scale = atol + rtol * np.abs(u)
    return np.sqrt(np.mean((delta / scale)**2))

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
                newton_not_happy (bool): Flag indiquant si Newton a converge ou pas.
        """
        nombre_de_niveaux = self.tableau_de_butcher.A.shape[1]
        nombre_d_equations = len(U_np)

        a = self.tableau_de_butcher.A
        c = self.tableau_de_butcher.C
        d = np.zeros_like(a[0,:])

        avec_prediction = self.tableau_de_butcher.est_avec_prediction
        if avec_prediction:
            b = self.tableau_de_butcher.B[0, :]
            d = self.tableau_de_butcher.B[1, :]
        else:
            b = self.tableau_de_butcher.B

        newton_not_happy = False
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

                success = False
                for refresh in range(2):
                    J = F.jacobien(tn_k, U_newton)
                    A = I - delta_t_x_akk * J

                    try:
                        LU_piv = lu_factor(A)                     # LU factorization once
                    except Exception:
                        newton_not_happy = True
                        return U_n, U_pred, newton_not_happy

                    for iteration_newton in range(max_iteration_newton):
                        residu = U_newton - (U_chap_k + delta_t_x_akk * F.evalue(tn_k, U_newton))

                        try:
                            delta = lu_solve(LU_piv, residu)
                        except:
                            newton_not_happy = True
                            return U_n, U_pred, newton_not_happy
                        U_newton -= delta
                        # verifie la convergence

                        #convergence = wrms_norm(delta, U_newton, abs_tolerance, rel_tolerance) < 1.0
                        #convergence = (np.linalg.norm(delta) <= abs_tolerance) and (np.linalg.norm(delta / (U_newton + 1e-12)) <= rel_tolerance)
                        scaled_error = np.abs(delta) / (abs_tolerance + np.abs(U_newton) * rel_tolerance)
                        convergence = np.linalg.norm(scaled_error, ord=np.inf) <= 1.0
                        if convergence and iteration_newton >= min_iteration_newton:
                            success=True
                            break
                    if success:
                        break
                else:
                    newton_not_happy = True
                    return U_n, U_pred, newton_not_happy

                U_chap[:, k] = U_newton
            else:
                tn_k = tn + c[k] * delta_t
                U_chap[:, k] = U_chap_k

            valeur_f[:, k] = delta_t * F.evalue(tn_k, U_chap[:, k])

            U_n += b[k] * valeur_f[:, k]
            if avec_prediction:
                U_pred += d[k] * valeur_f[:, k]

        return U_n, U_pred, newton_not_happy

    def resoud(self, systeme_EDOs:EDOs, initial_step_size: float, adaptive_time_stepping : bool = False, target_relative_error : float = 1.0e-5, min_step_size :float = 1.0e-8, max_step_size :float = 100.0):
        """
        Solves the ODE system by performing a series of time steps.

        Args:
            F (EDOs): The ODE system to solve.
            step_size (float): The step size.
            max_number_of_time_steps (int): The maximum number of steps.
            tn (float): The initial time.
            Un (np.ndarray): The initial state vector.

        Returns:
            tuple: A tuple containing lists of the solution times and state vectors.
                - temps (list): A list of the time points.
                - solutions (list): A list of the solution vectors at each time point.
        """
        if (not self.tableau_de_butcher.est_avec_prediction) and (adaptive_time_stepping):
            print('Warning: The selected solver does not support adaptive time stepping. Using fixed time steps instead. ⚠️')
            adaptive_time_stepping = False
        
        if not adaptive_time_stepping:
            return self._resoud_pas_de_temps_fixe(systeme_EDOs, initial_step_size)
        
        temps = [systeme_EDOs.t_init]
        solutions = [systeme_EDOs.initial_state]
        
        U_courant = np.copy(systeme_EDOs.initial_state)
        temps_courant = systeme_EDOs.t_init
        t_init = systeme_EDOs.t_init
        t_final = systeme_EDOs.t_final
        step_size = initial_step_size
        order = self.tableau_de_butcher.ordre

        number_of_time_steps = 0
        newton_failure_count = 0  # Counter for consecutive Newton failures
        max_newton_failures = 10  # Maximum number of failures before stopping

        while temps_courant < t_final:
            U_n_plus_1, U_pred, newton_not_happy = self._effectueUnPasDeTempsRKAvecTableauDeButcher(
                systeme_EDOs, temps_courant, step_size, U_courant
            )

            if newton_not_happy:
                newton_failure_count += 1
                print(f"Newton failed at time t = {temps_courant:.4f}. Reducing step size and retrying. Failure count: {newton_failure_count}")
                step_size = np.maximum(step_size / 2.0, min_step_size)  # Reduce step size
                if newton_failure_count >= max_newton_failures:
                    print(f"Maximum consecutive Newton failures ({max_newton_failures}) reached. Stopping the simulation.")
                    break
                continue  # Skip the rest of the loop and retry the same time step

            # Reset failure counter on success
            newton_failure_count = 0

            # Validate the step
            new_step_size, step_accepted = self._validePasDeTemps(U_n_plus_1, U_pred, step_size, target_relative_error, order, min_step_size, max_step_size, temps_courant, t_final)

            if step_accepted:
                # Step accepted: move to next time step
                U_courant = U_n_plus_1
                temps_courant += step_size
                temps.append(temps_courant)
                solutions.append(U_courant)
                step_size = new_step_size # Update step size for next iteration
                number_of_time_steps += 1
                if number_of_time_steps % 100 == 0:
                    print(f"Time step #{number_of_time_steps} completed. Current time: {temps_courant:.4f}")

            else:
                # Step rejected: retry with the new, smaller step size
                print(f"Time step {step_size} rejected at t = {temps_courant:.4f}. Retrying with step size: {new_step_size:.4e}")
                step_size = new_step_size

        print(f"The total number of time steps required to reach t_final = {t_final} is {number_of_time_steps}.")
        return np.array(temps), np.array(solutions)

    
    
    def _validePasDeTemps(self, U_approx, U_pred, step_size, target_relative_error, order, min_step_size, max_step_size, temps_courant, t_final):
        alpha = 0.1
        beta = 0.8  # safety factor
        eps = 1e-15
        relative_error = np.linalg.norm(U_approx - U_pred, 2.0) / (np.linalg.norm(U_pred, 2.0) + eps)
        
        # Calculate the new step size regardless of acceptance
        new_step_size = beta * step_size * (target_relative_error / (relative_error + eps))**(1.0 / order)

        # Determine if the step is accepted
        step_accepted = relative_error < (1 + alpha) * target_relative_error
        
        # Check for minimum and maximum step sizes
        if new_step_size < min_step_size:
            print(f'Warning! The computed step size {new_step_size:.4e} is less than the actual min step size: {min_step_size:.4e}. Using min step size.')
            new_step_size = min_step_size
        elif new_step_size > max_step_size:
            print(f'Warning! The computed step size {new_step_size:.4e} is greater than the actual max step size: {max_step_size:.4e}. Using max step size.')
            new_step_size = max_step_size
        
        # Final check to prevent overshooting the final time
        # This uses the OLD step_size to calculate the time for the next step.
        temps_apres_pas_courant = temps_courant + step_size
        
        if temps_apres_pas_courant + new_step_size > t_final:
            new_step_size = t_final - temps_apres_pas_courant
            # If the remaining time is zero or negative, the simulation is finished
            if new_step_size <= 0:
                step_accepted = True
                new_step_size = 0.0 # Make sure the next step is zero

        return new_step_size, step_accepted 

    def _resoud_pas_de_temps_fixe(self, systeme_EDOs:EDOs, step_size):
        """
        Solves the ODE system by performing a series of time steps.

        Args:
            systeme_EDOs (EDOs): The ODE system to solve.
            step_size (float): The step size.

        Returns:
            tuple: A tuple containing lists of the solution times and state vectors.
                - temps (list): A list of the time points.
                - solutions (list): A list of the solution vectors at each time point.
        """

        temps = [systeme_EDOs.t_init]
        solutions = [systeme_EDOs.initial_state]
        
        U_courant = np.copy(systeme_EDOs.initial_state)
        temps_courant = systeme_EDOs.t_init
        
        max_number_of_time_steps = int ((systeme_EDOs.t_final - systeme_EDOs.t_init) / step_size)

        for i in range(max_number_of_time_steps):
            # on fait un pas de temps
            U_n_plus_1, _, newton_not_happy = self._effectueUnPasDeTempsRKAvecTableauDeButcher(
                systeme_EDOs, temps_courant, step_size, U_courant
            )
            
            # Verifie la convergence de Newton
            if newton_not_happy:
                print(f"L'algorithme de Newton a echoue a converger au pas de temps {i+1}. Arret de la simulation.")
                break
                
            # Mise a jour de la solution courante et du temps courant
            U_courant = U_n_plus_1
            temps_courant += step_size
            
            # On sauvegarde la solution du pas de temps dans le vecteur solution
            temps.append(temps_courant)
            solutions.append(U_courant)

        return np.array(temps), np.array(solutions)
    
    def solve(self, systeme_EDOs:EDOs, initial_step_size: float, adaptive_time_stepping : bool = False, target_relative_error : float = 1.0e-5, min_step_size :float = 1.0e-8, max_step_size :float = 100.0):
        """
        Alias de resoud().
        """
        return self.resoud(systeme_EDOs, initial_step_size, adaptive_time_stepping, target_relative_error, min_step_size, max_step_size)



