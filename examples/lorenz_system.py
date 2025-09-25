import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyodys import ODEProblem, PyodysSolver

# Define Lorenz System
class LorenzSystem(ODEProblem):
    def __init__(self, t_init, t_final, initial_state, sigma=10, rho=28, beta=2.667):
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def evaluate_at(self, t, u):
        x, y, z = u
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])
    
    def __call__(self, t, u):
        return self.evaluate_at(t, u)
    
    def jacobian_at(self, t, u):
        x, y, z = u
        jacobian_at = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, - self.beta]
        ])
        return jacobian_at

def extract_args():
    parser = argparse.ArgumentParser(description="Solve the Robertson System.")
    parser.add_argument('--method', '-m', 
                        type=str, 
                        default='dopri54',
                        help='The Runge-Kutta method to use.')
    parser.add_argument('--fixed-step', '-s', 
                        type=float, 
                        default=None,
                        help='The fixed step size. Must be provided for non adaptive stepping.')
    parser.add_argument('--first-step', '-f', 
                        type=float, 
                        default=1e-8,
                        help='The initial time step size.')
    parser.add_argument('--final-time', '-t', 
                        type=float, 
                        default=50.0,
                        help='The final time for the simulation.')
    parser.add_argument('--rtol', '-rt', 
                        type=float,
                        default=1e-10,
                        help='The target relative error for adaptive time stepping.')
    parser.add_argument('--atol', '-at', 
                        type=float,
                        default=1e-10,
                        help='The target absolute error for adaptive time stepping.')
    parser.add_argument('--no-adaptive', 
                        action='store_false', 
                        dest='adaptive',
                        help='Disable adaptive time stepping.')
    parser.add_argument('--min-step','-n', 
                        type=float,
                        default=1e-10,
                        help='The minimum time step size for adaptive stepping.')
    parser.add_argument('--max-step', '-x',
                        type=float,
                        default=1e-2,
                        help='The maximum time step size for adaptive stepping.')
    parser.add_argument('--save-csv', 
                        action='store_true', 
                        help='Save the results to a CSV file.')
    parser.add_argument('--save-png', 
                        action='store_true', 
                        help='Save the results to a png file.')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print progress info.')


    return parser.parse_args()

if __name__ == '__main__':

    args = extract_args()

    # Initial conditions
    t0 = 0.0
    tf = args.final_time
    u0 = [0.0, 1.0, 0.0]
    lorenz_system = LorenzSystem(t_init=t0, 
                           t_final=args.final_time, 
                           initial_state=u0)

    # solver
    solver = PyodysSolver(
                    method = args.method,
                    fixed_step=args.fixed_step,
                    first_step=args.first_step, 
                    adaptive=args.adaptive,
                    rtol=args.rtol,
                    atol=args.atol,
                    min_step=args.min_step, 
                    max_step=args.max_step,
                    verbose=args.verbose if args.verbose else False
    )

    times, solutions = solver.solve( ode_problem = lorenz_system )

    if args.save_csv:
        print("Saving data to CSV...")
        results = np.column_stack((times, solutions))
        header = "time,x(t),y(t),z(t)"
        np.savetxt('lorenz_system.csv', results, delimiter=',', header=header, comments='')
        print("Data saved to lorenz_system.csv")

    fig = plt.figure(figsize=(14, 6))
    # 1. First subplot: Time series plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(times, solutions[:,0], 'b.-', markersize=2, label='x(t)')
    ax1.plot(times, solutions[:,1], 'r-', markersize=2, label='y(t)')
    ax1.plot(times, solutions[:,2], 'm-', markersize=2, label='z(t)')
    ax1.set_title("Lorenz System: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Second subplot: 3D Lorenz attractor plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(solutions[:, 0], solutions[:, 1], solutions[:, 2], lw=0.5)
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    ax2.set_title("Lorenz Attractor")
    
    plt.tight_layout()
    plt.show()