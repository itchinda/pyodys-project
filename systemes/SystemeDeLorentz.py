import numpy as np
from .EDOs import EDOs


class SystemeDeLorentz(EDOs):
    def __init__(self, temps_initial, condition_initiale, sigma=10, rho=28, beta=2.667):
        # Appel du constructeur Parent
        super().__init__(temps_initial, condition_initiale)
        # Parametre specifiques au system de Lorentz
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def evalue(self, t, u):
        # u, le vecteur d'etat a l'instant t est [x, y, z]
        x, y, z = u
        
        # Define the derivatives dx/dt, dy/dt, dz/dt
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        
        # Retourne les derivees dans un array numpy
        return np.array([dxdt, dydt, dzdt])
    
    def jacobien(self, t, u):
        x, y, z = u
        Jacobien = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, - self.beta]
        ])
        return Jacobien
    

if __name__=='__main__':
    ode = SystemeDeLorentz(0.0, [0.0,0.0,0.0])
    print(ode.jacobien(0.0, [0.0,0.0,0.0]))
    