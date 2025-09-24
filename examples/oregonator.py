import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pyodys import ODEProblem, PyodysSolver

# Define Robertson System
class RobertsonModel(ODEProblem):
    def __init__(self, t_init, t_final, initial_state, k1=0.04, k2=3.0e7, k3=1.0e4):
        # Call the parent constructor
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
 
        self.nb_of_jac_call=0
        
    def evaluate_at(self, t, u):
        # u, state at time t: u = [x, y, z]
        x, y, z = u
        
        # Define the derivatives dx/dt, dy/dt, dz/dt
        dxdt = 77.27*(y + x*(1 - 8.375e-6*x -y))
        dydt = 1/77.27 * (z - (1+x)*y)
        dzdt = 0.161*(x - z)
        
        # Returns the derivatives in a Numpy Array
        return np.array([dxdt, dydt, dzdt])
    
    def jacobian_at(self, t, u):
        x, y, z = u
        jacobian_at = np.array([
            [77.27*((1 - 8.375e-6*x -y) - 8.375e-6*x ), 77.27*(1 + x*(-1)), 0.0],
            [1/77.27*(-y), 1/77.27*(-(1+x)), 1/77.27],
            [0.161, 0.0, -0.161]
        ])
        self.nb_of_jac_call+=1

        return jacobian_at

def extract_args():
    parser = argparse.ArgumentParser(description="Solve the Robertson System.")
    parser.add_argument('--method', '-m', 
                        type=str, 
                        default='esdirk64',
                        help='The Runge-Kutta method to use.')
    parser.add_argument('--first-step', '-s', 
                        type=float, 
                        default=None,
                        help='The initial time step size.')
    parser.add_argument('--final-time', '-t', 
                        type=float, 
                        default=700.0,
                        help='The final time for the simulation.')
    parser.add_argument('--rtol', '-rt', 
                        type=float,
                        default=1e-8,
                        help='The target relative error for adaptive time stepping.')
    parser.add_argument('--atol', '-at', 
                        type=float,
                        default=1e-8,
                        help='The target absolute error for adaptive time stepping.')
    parser.add_argument('--no-adaptive', 
                        action='store_false', 
                        dest='adaptive',
                        help='Enable adaptive time stepping.')
    parser.add_argument('--min-step','-n', 
                        type=float,
                        default=1e-8,
                        help='The minimum time step size for adaptive stepping.')
    parser.add_argument('--max-step', '-x',
                        type=float,
                        default=1e4,
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
    u0 = [1.0, 2.0, 3.0]
    system = RobertsonModel(t0, tf, u0)

    # solver
    solver = PyodysSolver(
        method = args.method,
        first_step = args.first_step,
        adaptive = args.adaptive,
        rtol = args.rtol,
        atol = args.atol,
        min_step = args.min_step,
        max_step = args.max_step,
        verbose=args.verbose
    )

    # Solve the system
    start=time.time()
    times, solutions = solver.solve( system )
    elapsed=time.time()-start
    print(f"Python EDOs runtime: {elapsed:.4f} seconds")

    print(f"Nb of Jacobian call: {system.nb_of_jac_call}")
    
    if args.save_csv or args.save_png:
        output_directory = "oregonator_results"
        try:
            os.mkdir(output_directory)
            print(f"Directory {output_directory} created successfully.")
        except FileExistsError:
            print(f"Directory {output_directory} already exists.")
        except FileNotFoundError:
            print("The parent directory does not exist.")
            os.makedirs(output_directory, exist_ok=True)

    if args.save_csv:
        print("Saving data to CSV...")
        filename = "oregonator.csv"
        full_result_path = os.path.join(output_directory, filename)
        results_to_save = np.column_stack((times, solutions))
        header = "time,x(t),y(t),z(t)"
        np.savetxt(full_result_path, 
                   results_to_save, 
                   delimiter=',', 
                   header=header, 
                   comments='')
        print(f"CSV saved successfully at: {os.path.abspath(full_result_path)}")
    
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(times, solutions[:,0], 'b.-', markersize=2, label='x(t)')
    ax1.plot(times, 10*solutions[:,1], 'r-', markersize=2, label='10y(t)')
    ax1.plot(times, solutions[:,2], 'm-', markersize=2, label='z(t)')
    ax1.set_title("Oregonator: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    if args.save_png:
        print("Saving to PNG...")
        filename = "oregonator.png"
        full_result_path = os.path.join(output_directory, filename)
        plt.savefig(full_result_path)
        print(f"PNG saved successfully at: {os.path.abspath(full_result_path)}")

    plt.tight_layout()
    plt.show()