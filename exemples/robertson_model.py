import numpy as np
import matplotlib.pyplot as plt
import time
from systemes.EDOs import EDOs
from solveurs.runge_kutta.TableauDeButcher import TableauDeButcher
from solveurs.runge_kutta.SolveurRKAvecTableauDeButcher import SolveurRKAvecTableauDeButcher

# Define Lorenz System
class RobertsonModel(EDOs):
    def __init__(self, t_init, t_final, initial_state, k1=0.04, k2=3.0e7, k3=1.0e4):
        # Call the parent constructor
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
    def evalue(self, t, u):
        # u, state at time t: u = [x, y, z]
        x, y, z = u
        
        # Define the derivatives dx/dt, dy/dt, dz/dt
        dxdt = -self.k1 * x + self.k3 * y * z
        dydt = self.k1 * x - self.k2 * y**2.0 - self.k3 * y * z
        dzdt = self.k2 * y**2.0
        
        # Returns the derivatives in a Numpy Array
        return np.array([dxdt, dydt, dzdt])
    
    def jacobien(self, t, u):
        x, y, z = u
        Jacobien = np.array([
            [-self.k1, self.k3 * z, self.k3 * y],
            [ self.k1, -2.0*self.k2 * y - self.k3 * z, -self.k3 * y],
            [0.0, 2.0*self.k2 * y, 0.0]
        ])
        return Jacobien


if __name__ == '__main__':
    # Initial conditions
    t0 = 0.0
    tf = 1.0e2
    u0 = [1.0, 0.0, 0.0]
    systeme = RobertsonModel(t0, tf, u0)

    # RK4 solver
    time_step = 0.0001
    max_number_of_time_steps = int((tf - t0) / time_step)
    solveur_rk4 = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name('rk4'))
    start=time.time()
    t_rk4, sol_rk4 = solveur_rk4.solve(systeme, time_step, max_number_of_time_steps)
    elapsed=time.time()-start
    print(f"Python EDOs runtime: {elapsed:.4f} seconds")
    # Create a single figure with two subplots
    fig = plt.figure(figsize=(14, 6)) # Adjust figure size to accommodate two plots side-by-side
    
    # 1. First subplot: Time series plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.semilogx(t_rk4, sol_rk4[:,0], 'b.-', markersize=2, label='x(t) RK4')
    ax1.semilogx(t_rk4, 1e4*sol_rk4[:,1], 'r-', markersize=2, label='10^4 y(t) RK4')
    ax1.semilogx(t_rk4, sol_rk4[:,2], 'm-', markersize=2, label='z(t) RK4')
    ax1.set_title("Robertson Model: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    
    plt.tight_layout()
    plt.show()