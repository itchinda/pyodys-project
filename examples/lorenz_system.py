import argparse
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

    parser = argparse.ArgumentParser(description="Solve the Lorenz System.")
    parser.add_argument('--method', '-m', 
                        type=str, 
                        default='dopri5',
                        help='The Runge-Kutta method to use.')
    parser.add_argument('--step-size', '-s', 
                        type=float, 
                        default=1e-4,
                        help='The initial time step size.')
    parser.add_argument('--final-time', '-t', 
                        type=float, 
                        default=50.0,
                        help='The final time for the simulation.')
    parser.add_argument('--tolerance', '-tol', 
                        type=float,
                        default=1e-8,
                        help='The target relative error for adaptive time stepping.')
    parser.add_argument('--no-adaptive-stepping', 
                        action='store_false', 
                        dest='adaptive_stepping',
                        help='Disable adaptive time stepping.')
    parser.add_argument('--min-step-size','-n', 
                        type=float,
                        default=1e-12,
                        help='The minimum time step size for adaptive stepping.')
    parser.add_argument('--max-step-size', '-x',
                        type=float,
                        default=1.0,
                        help='The maximum time step size for adaptive stepping.')
    
    args = parser.parse_args()

    # Initial conditions
    t0 = 0.0
    tf = args.final_time
    u0 = [0.0, 1.0, 0.0]
    systeme = LorenzSystem(t_init=t0, 
                           t_final=args.final_time, 
                           initial_state=u0)

    # solver
    solver = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name(args.method))

    time, solution = solver.solve(
        systeme_EDOs=systeme, 
        initial_step_size=args.step_size, 
        adaptive_time_stepping=args.adaptive_stepping,
        target_relative_error=args.tolerance, 
        min_step_size=args.min_step_size, 
        max_step_size=args.max_step_size
        )

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(14, 6)) # Adjust figure size to accommodate two plots side-by-side
    
    # 1. First subplot: Time series plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(time, solution[:,0], 'b.-', markersize=2, label='x(t)')
    ax1.plot(time, solution[:,1], 'r-', markersize=2, label='y(t)')
    ax1.plot(time, solution[:,2], 'm-', markersize=2, label='z(t)')
    ax1.set_title("Lorenz System: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Second subplot: 3D attractor plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(solution[:, 0], solution[:, 1], solution[:, 2], lw=0.5)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_title("Lorenz Attractor")
    
    plt.tight_layout()
    plt.show()