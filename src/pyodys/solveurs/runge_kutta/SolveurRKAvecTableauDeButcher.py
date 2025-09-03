from ...systemes.EDOs import EDOs
from .TableauDeButcher import TableauDeButcher
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgError
import csv
import os


class PyOdysError(RuntimeError):
    """Exception raised when PyOdys fails to solve a problem."""
    def __init__(self, message):
        super().__init__(message)


def wrms_norm(delta, u, atol=1e-12, rtol=1e-6):
    """
    Weighted Root Mean Square norm.
    delta : Newton update vector
    u : current Newton iterate
    """
    scale = atol + rtol * np.abs(u)
    return np.sqrt(np.mean((delta / scale) ** 2))


class SolveurRKAvecTableauDeButcher(object):
    def __init__(self, tableau_de_butcher=TableauDeButcher.from_name("erk4"),
                 verbose: bool = True,
                 progress_interval_in_time: int = 100.0,
                 max_jacobian_refresh=1,
                 export_interval: int = None,
                 export_prefix=None):
        if not isinstance(tableau_de_butcher, TableauDeButcher):
            raise TypeError("On devrait passer un objet de type TableauDeButcher.")
        self.tableau_de_butcher = tableau_de_butcher
        self.verbose = verbose
        self.progress_interval_in_time = progress_interval_in_time
        self.max_jacobian_refresh = max_jacobian_refresh
        self.export_interval = export_interval
        self.export_prefix = export_prefix
        self.export_counter = 0
        self.schema_explicite = self.tableau_de_butcher.est_explicite
        self.schema_avec_prediction = self.tableau_de_butcher.est_avec_prediction

    def _print_verbose(self, message):
        if self.verbose:
            print(message)

    def _print_pyodys_error_message(self, message):
        print(message)

    def _export(self, temps, solutions: np.ndarray):
        if self.export_prefix is None:
            return
        self.export_counter += 1
        filename = f"{self.export_prefix}_{self.export_counter:05d}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        n_vars = solutions.shape[1] if solutions.ndim > 1 else 1
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["t"] + [f"u{i}" for i in range(n_vars)]
            writer.writerow(header)
            for t, u in zip(temps, solutions):
                row = [t] + (u.tolist() if n_vars > 1 else [u])
                writer.writerow(row)
        self._print_verbose(f"Exported {len(temps)} steps to {filename}")

    def _effectueUnPasDeTempsRKAvecTableauDeButcher(self, F: EDOs, tn: float,
                                                    delta_t: float, U_np: np.ndarray):
        """
        Effectue un pas de Runge-Kutta en utilisant un tableau de Butcher
        """
        s = self.tableau_de_butcher.A.shape[0]
        n_eq = len(U_np)

        a = self.tableau_de_butcher.A
        c = self.tableau_de_butcher.C
        d = np.zeros_like(a[0, :])

        if self.schema_avec_prediction:
            b = self.tableau_de_butcher.B[0, :]
            d = self.tableau_de_butcher.B[1, :]
        else:
            b = self.tableau_de_butcher.B

        newton_not_happy = False
        U_chap = np.zeros((n_eq, s))
        valeur_f = np.zeros((n_eq, s))

        max_iteration_newton = 10
        abs_tolerance = 1e-12
        rel_tolerance = 1e-8

        U_n = np.copy(U_np)
        U_pred = np.zeros_like(U_np)
        if self.schema_avec_prediction:
            U_pred = np.copy(U_np)

        I = np.eye(n_eq)

        for k in range(s):
            U_chap_k = U_np + np.sum(a[k, :k] * valeur_f[:, :k], axis=1)

            if a[k, k] != 0.0:
                tn_k = tn + c[k] * delta_t
                delta_t_x_akk = delta_t * a[k, k]
                U_newton = np.copy(U_chap_k)
                success = False

                for refresh in range(self.max_jacobian_refresh + 1):
                    try:
                        if self.schema_explicite:
                            A = I
                        else:
                            J = F.jacobien(tn_k, U_newton)
                            A = I - delta_t_x_akk * J
                        LU_piv = lu_factor(A)
                    except LinAlgError:
                        newton_not_happy = True
                        return U_n, U_pred, newton_not_happy

                    for iteration_newton in range(max_iteration_newton):
                        residu = U_newton - (
                            U_chap_k + delta_t_x_akk * F.evalue(tn_k, U_newton)
                        )

                        try:
                            delta = lu_solve(LU_piv, residu)
                        except LinAlgError:
                            self._print_verbose(f"Jacobian solve failed at stage {k}")
                            newton_not_happy = True
                            return U_n, U_pred, newton_not_happy
                        U_newton -= delta

                        #convergence = np.linalg.norm(delta) < abs_tolerance and np.linalg.norm(delta,2) / (np.linalg.norm(U_newton,2)+1e-12)  < rel_tolerance
                        #if convergence:
                        if wrms_norm(delta, U_newton, abs_tolerance, rel_tolerance) < 0.7:
                            success = True
                            break
                    if success:
                        break
                else:
                    newton_not_happy = True
                    self._print_verbose(
                        f"Newton failed at stage {k} even after Jacobian refresh"
                    )
                    return U_n, U_pred, newton_not_happy

                U_chap[:, k] = U_newton
            else:
                tn_k = tn + c[k] * delta_t
                U_chap[:, k] = U_chap_k

            valeur_f[:, k] = delta_t * F.evalue(tn_k, U_chap[:, k])
            U_n += b[k] * valeur_f[:, k]
            if self.schema_avec_prediction:
                U_pred += d[k] * valeur_f[:, k]

        return U_n, U_pred, newton_not_happy

    def resoud(self, systeme_EDOs: EDOs, initial_step_size: float,
               adaptive_time_stepping: bool = False, target_relative_error: float = 1.0e-5,
               min_step_size: float = 1.0e-8, max_step_size: float = 100.0):
        """
        Solve ODE system with adaptive or fixed time stepping
        """
        if (not self.tableau_de_butcher.est_avec_prediction) and adaptive_time_stepping:
            self._print_verbose(
                "Warning: The selected solver does not support adaptive time stepping. Using fixed time steps instead. ⚠️"
            )
            adaptive_time_stepping = False

        if not adaptive_time_stepping:
            return self._resoud_pas_de_temps_fixe(systeme_EDOs, initial_step_size)

        temps = [systeme_EDOs.t_init]
        solutions = [systeme_EDOs.initial_state]

        U_courant = np.copy(systeme_EDOs.initial_state)
        temps_courant = systeme_EDOs.t_init
        t_final = systeme_EDOs.t_final
        step_size = initial_step_size
        order = self.tableau_de_butcher.ordre

        number_of_time_steps = 0
        next_progress_in_time = systeme_EDOs.t_init + self.progress_interval_in_time
        newton_failure_count = 0
        max_newton_failures = 10

        while temps_courant < t_final:
            # tronquer pour ne pas dépasser t_final
            step_size = min(step_size, t_final - temps_courant)

            U_n_plus_1, U_pred, newton_not_happy = \
                self._effectueUnPasDeTempsRKAvecTableauDeButcher(
                    systeme_EDOs, temps_courant, step_size, U_courant
                )

            if newton_not_happy:
                newton_failure_count += 1
                self._print_verbose(
                    f"Newton failed at t = {temps_courant:.4f}. "
                    f"Reducing step size and retrying. Failure count: {newton_failure_count}"
                )
                step_size = max(step_size / 2.0, min_step_size)
                if newton_failure_count >= max_newton_failures:
                    message = (
                        f"Maximum consecutive Newton failures ({max_newton_failures}) reached. "
                        "Stopping the simulation."
                    )
                    self._print_verbose(message)
                    raise PyOdysError(message)
                continue  # retry immediately at same time

            # succès Newton
            newton_failure_count = 0

            new_step_size, step_accepted = self._validePasDeTemps(
                U_n_plus_1, U_pred, step_size, target_relative_error, order,
                min_step_size, max_step_size, temps_courant, t_final
            )

            # new_step_size, step_accepted = self.test_pas_de_temps_valide(U_n_plus_1, U_pred, step_size, order, target_relative_error, 
            #                  min_step_size, max_step_size)

            if step_accepted:
                U_courant = U_n_plus_1
                temps_courant += step_size
                temps.append(temps_courant)
                solutions.append(U_courant)
                step_size = new_step_size
                number_of_time_steps += 1
                if temps_courant >= next_progress_in_time:
                    self._print_verbose(
                        f"Time step #{number_of_time_steps} completed. Current time: {temps_courant:.4f}"
                    )
                    next_progress_in_time += self.progress_interval_in_time
                if self.export_interval and (number_of_time_steps % self.export_interval == 0):
                    self._export(np.array(temps[:-1]), np.array(solutions[:-1,:]))
                    temps = [temps[-1]]
                    solutions = [solutions[-1]]
            else:
                self._print_verbose(
                    f"Time step {step_size:.4e} rejected at t = {temps_courant:.4f}. "
                    f"Retrying with step size: {new_step_size:.4e}"
                )
                step_size = new_step_size

        self._print_verbose(
            f"The total number of time steps required to reach t_final = {t_final} is {number_of_time_steps}."
        )
        return np.array(temps), np.array(solutions)


    def _validePasDeTemps(self, U_approx, U_pred, step_size, target_relative_error,
                          order, min_step_size, max_step_size, temps_courant, t_final):
        alpha = 0.1
        beta = 0.9
        eps = 1e-15
        # WRMS-like erreur
        # scale = 1e-12 + 1e-6 * np.maximum(np.max(np.abs(U_approx)), np.max(np.abs(U_pred)))
        # err = np.sqrt(np.mean(((U_approx - U_pred) / scale) ** 2))
        # step_accepted = err <= (1.0 + alpha)

        # err = wrms_norm(U_approx - U_pred, U_pred, rtol=target_relative_error)
        # step_accepted = err < 1.0

        #err = np.linalg.norm(U_approx - U_pred, 2) / (np.linalg.norm( U_pred, 2) + eps)
        err = np.linalg.norm((U_approx - U_pred) / (np.abs(U_pred)+1e-12), ord=np.inf) #/ (np.linalg.norm( U_pred, 2) + eps)
        step_accepted = err < (1 + alpha) * target_relative_error
        new_step_size = beta * step_size * (target_relative_error / max(err, eps)) ** (1.0 / (order))

        if new_step_size < min_step_size:
            self._print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} < min step size {min_step_size:.4e}. Using min step size."
            )
            new_step_size = min_step_size
        elif new_step_size > max_step_size:
            self._print_verbose(
                f"Warning! Computed step size {new_step_size:.4e} > max step size {max_step_size:.4e}. Using max step size."
            )
            new_step_size = max_step_size

        temps_apres_pas_courant = temps_courant + step_size
        if temps_apres_pas_courant + new_step_size > t_final:
            new_step_size = max(t_final - temps_apres_pas_courant, 0.0)
            if new_step_size <= 0:
                step_accepted = True
                new_step_size = 0.0
        return new_step_size, step_accepted

    def _resoud_pas_de_temps_fixe(self, systeme_EDOs: EDOs, step_size):
        U_courant = np.copy(systeme_EDOs.initial_state)
        temps_courant = systeme_EDOs.t_init
        next_progress_in_time = systeme_EDOs.t_init + self.progress_interval_in_time
        max_number_of_time_steps = int((systeme_EDOs.t_final - systeme_EDOs.t_init) / step_size)
        if self.export_interval:
            temps = np.empty(self.export_interval+1, dtype=float)
            solutions = np.empty((self.export_interval+1, len(systeme_EDOs.initial_state)), dtype=float)
        else :
            temps = np.empty(max_number_of_time_steps+1, dtype=float)
            solutions = np.empty((max_number_of_time_steps+1, len(systeme_EDOs.initial_state)), dtype=float)
        temps[0] = systeme_EDOs.t_init
        solutions[0,:] = np.copy(U_courant)

        k = 0

        for n in range(max_number_of_time_steps):
            U_n_plus_1, _, newton_not_happy = self._effectueUnPasDeTempsRKAvecTableauDeButcher(
                systeme_EDOs, temps_courant, step_size, U_courant
            )
            if newton_not_happy:
                message = f"Newton failed at time step {n+1} even after Jacobian refresh."
                self._print_verbose(message)
                raise PyOdysError(message)

            U_courant = U_n_plus_1
            temps_courant += step_size
            temps[n+1] = temps_courant
            solutions[n+1,:] = U_courant
            k+=1

            if self.export_interval and k == self.export_interval == 0:
                self._export(np.array(temps), np.array(solutions))
                temps = [temps[-1]]
                solutions = [solutions[-1]]

            if temps_courant >= next_progress_in_time:
                self._print_verbose(
                    f"Time step #{n+1} completed. Current time: {temps_courant:.4f}"
                )
                next_progress_in_time += self.progress_interval_in_time

        return np.array(temps), np.array(solutions)

    def solve(self, systeme_EDOs: EDOs, initial_step_size: float,
              adaptive_time_stepping: bool = False, target_relative_error: float = 1.0e-5,
              min_step_size: float = 1.0e-8, max_step_size: float = 100.0):
        return self.resoud(systeme_EDOs, initial_step_size, adaptive_time_stepping,
                           target_relative_error, min_step_size, max_step_size)

    