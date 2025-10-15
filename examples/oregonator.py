import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pyodys import ODEProblem, PyodysSolver, extract_args

# Define the well-known ergonator System
class ErgonatorSystem(ODEProblem):
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


if __name__ == '__main__':
    
    args = extract_args(
        description = "Solve the so-called Oregonator System.",
        method = 'esdirk64',
        fixed_step = None,
        first_step = None,
        final_time = 700,
        rtol = 1e-10,
        atol = 1e-10,
        min_step = 1e-8
    )

    # Initial conditions
    t0 = 0.0
    tf = args.final_time
    u0 = [1.0, 2.0, 3.0]
    system = ErgonatorSystem(t0, tf, u0)

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
