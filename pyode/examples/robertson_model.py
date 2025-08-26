import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pyode import EDOs
from pyode import TableauDeButcher
from pyode import SolveurRKAvecTableauDeButcher

# Define Robertson System
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

def extract_args():
    parser = argparse.ArgumentParser(description="Solve the Robertson System.")
    parser.add_argument('--method', '-m', 
                        type=str, 
                        default='sdirk_norsett_thomson_34',
                        help='The Runge-Kutta method to use.')
    parser.add_argument('--step-size', '-s', 
                        type=float, 
                        default=1e-4,
                        help='The initial time step size.')
    parser.add_argument('--final-time', '-t', 
                        type=float, 
                        default=1.0,
                        help='The final time for the simulation.')
    parser.add_argument('--tolerance', '-tol', 
                        type=float,
                        default=1e-6,
                        help='The target relative error for adaptive time stepping.')
    parser.add_argument('--no-adaptive-stepping', 
                        action='store_false', 
                        dest='adaptive_stepping',
                        help='Disable adaptive time stepping.')
    parser.add_argument('--min-step-size','-n', 
                        type=float,
                        default=1e-8,
                        help='The minimum time step size for adaptive stepping.')
    parser.add_argument('--max-step-size', '-x',
                        type=float,
                        default=1e4,
                        help='The maximum time step size for adaptive stepping.')
    parser.add_argument('--save-csv', 
                        action='store_true', 
                        help='Save the results to a CSV file.')
    parser.add_argument('--save-png', 
                        action='store_true', 
                        help='Save the results to a png file.')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = extract_args()

    # Initial conditions
    t0 = 0.0
    tf = args.final_time
    u0 = [1.0, 0.0, 0.0]
    system = RobertsonModel(t0, tf, u0)

    # solver
    method = args.method
    solver = SolveurRKAvecTableauDeButcher(TableauDeButcher.from_name(method))

    # Solve the system
    start=time.time()
    times, solutions = solver.solve(
        system,
        initial_step_size=args.step_size,
        adaptive_time_stepping=args.adaptive_stepping,
        target_relative_error=args.tolerance,
        min_step_size=args.min_step_size,
        max_step_size=args.max_step_size
    )
    elapsed=time.time()-start
    print(f"Python EDOs runtime: {elapsed:.4f} seconds")
    
    if args.save_csv or args.save_png:
        output_directory = "robertson_model_results"
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
        filename = "robertson_model.csv"
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
    ax1.semilogx(times, solutions[:,0], 'b.-', markersize=2, label='x(t)')
    ax1.semilogx(times, 1e4*solutions[:,1], 'r-', markersize=2, label='10^4 y(t)')
    ax1.semilogx(times, solutions[:,2], 'm-', markersize=2, label='z(t)')
    ax1.set_title("Robertson Model: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    if args.save_png:
        print("Saving to PNG...")
        filename = "robertson_model.png"
        full_result_path = os.path.join(output_directory, filename)
        plt.savefig(full_result_path)
        print(f"PNG saved successfully at: {os.path.abspath(full_result_path)}")

    plt.tight_layout()
    plt.show()