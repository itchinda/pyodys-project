import numpy as np
import matplotlib.pyplot as plt
from EDOsSolverModules import EDOs
from EDOsSolverModules import TableauDeButcher
from EDOsSolverModules import SolveurRKAvecTableauDeButcher

# Define Lorenz System
class LorenzSystem(EDOs):
    def __init__(self, t_init, t_final, initial_state, sigma=10, rho=28, beta=2.667):
        # Call the parent constructor
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def evalue(self, t, u):
        # u, state at time t: u = [x, y, z]
        x, y, z = u
        
        # Define the derivatives dx/dt, dy/dt, dz/dt
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        
        # Returns the derivatives in a Numpy Array
        return np.array([dxdt, dydt, dzdt])
    
    def jacobien(self, t, u):
        x, y, z = u
        Jacobien = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, - self.beta]
        ])
        return Jacobien


if __name__ == '__main__':
    # Initial conditions
    t0 = 0.0
    tf = 50.0
    u0 = [0.0, 1.0, 0.0]
    systeme = LorenzSystem(t0, tf, u0)

    # RK4 solver
    step_size = 0.001
    solveur_sdirk = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name('sdirk_hairer_norsett_wanner_45'))

    t_rk4, sol_rk4 = solveur_sdirk.solve(systeme, step_size, adaptive_time_stepping=True, target_relative_error=1e-10, min_step_size=1e-8, max_step_size=1e5)

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(14, 6)) # Adjust figure size to accommodate two plots side-by-side
    
    # 1. First subplot: Time series plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t_rk4, sol_rk4[:,0], 'b.-', markersize=2, label='x(t) RK4')
    ax1.plot(t_rk4, sol_rk4[:,1], 'r-', markersize=2, label='y(t) RK4')
    ax1.plot(t_rk4, sol_rk4[:,2], 'm-', markersize=2, label='z(t) RK4')
    ax1.set_title("Lorenz System: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Second subplot: 3D attractor plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(sol_rk4[:, 0], sol_rk4[:, 1], sol_rk4[:, 2], lw=0.5)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_title("Lorenz Attractor (RK4)")
    
    plt.tight_layout()
    plt.show()