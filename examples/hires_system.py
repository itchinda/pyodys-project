import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from pyodys import ODEProblem, PyodysSolver, extract_args


# Define HIRES System
class HIRESModel(ODEProblem):
    def __init__(self, t_init, t_final, initial_state):
        super().__init__(t_init, t_final, initial_state)

    def evaluate_at(self, t, u):
        """
        HIRES stiff system (7 equations).
        See: Hairer & Wanner (1996) Solving Ordinary Differential Equations II.
        """
        y = u
        dydt = np.zeros(8)

        dydt[0] = -1.71*y[0] + 0.43*y[1] + 8.32*y[2] + 0.0007
        dydt[1] =  1.71*y[0] - 8.75*y[1]
        dydt[2] = -10.03*y[2] + 0.43*y[3] + 0.035*y[4]
        dydt[3] =  8.32*y[1] + 1.71*y[2] - 1.12*y[3]
        dydt[4] = -1.745*y[4] + 0.43*y[5] + 0.43*y[6]
        dydt[5] = -280.0*y[5]*y[7] + 0.69*y[3] + 1.71*y[4] - 0.43*y[5] + 0.69*y[6]
        dydt[6] =  280.0*y[5]*y[7] - 1.81*y[6]
        dydt[7] = -dydt[6]

        return dydt

    def jacobien(self, t, u):
        """
        Analytical Jacobian of HIRES system.
        """
        y = u
        J = np.zeros((8, 8))

        J[0,:] = [-1.71, 0.43, 8.32, 0,     0,     0,     0,     0]
        J[1,:] = [ 1.71, -8.75, 0,   0,     0,     0,     0,     0]
        J[2,:] = [ 0,    0, -10.03, 0.43, 0.035, 0,     0,     0]
        J[3,:] = [ 0,    8.32, 1.71, -1.12, 0,   0,     0,     0]
        J[4,:] = [ 0,    0,    0,   0, -1.745, 0.43, 0.43, 0]
        J[5,:] = [ 0,    0,    0,   0.69, 1.71, -280*y[7] - 0.43, 0.69, -280*y[5]]
        J[6,:] = [ 0,    0,    0,   0,     0, 280*y[7], -1.81, 280*y[5]]
        J[7,:] = [ 0,    0,    0,   0,     0, -280*y[7], 1.81, -280*y[5]]

        return J


if __name__ == '__main__':

    args = extract_args(
        description = "Solve the HIRES System.",
        method = 'esdirk64',
        fixed_step = None,
        first_step = None,
        final_time = 324,
        rtol = 1e-10,
        atol = 1e-10,
        min_step = 1e-8,
        max_step = 324
    )

    # Initial conditions (from Hairer & Wanner II, p. 5)
    t0 = 0.0
    tf = args.final_time
    u0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]

    hires_system = HIRESModel(t0, tf, u0)

    # solver
    solver = PyodysSolver(
                method = args.method,
                fixed_step = args.fixed_step,
                adaptive = args.adaptive,
                first_step = args.first_step,
                min_step = args.min_step,
                max_step = args.max_step,
                rtol = args.rtol,
                atol = args.atol,
                verbose=args.verbose
            )

    # Solve the system
    start=time.time()
    times, solutions = solver.solve( hires_system )
    elapsed=time.time()-start
    print(f"Python HIRES runtime: {elapsed:.4f} seconds")

    # Output folder
    if args.save_csv or args.save_png:
        output_directory = "hires_model_results"
        os.makedirs(output_directory, exist_ok=True)

    # Save CSV
    if args.save_csv:
        filename = "hires_model.csv"
        full_result_path = os.path.join(output_directory, filename)
        results_to_save = np.column_stack((times, solutions))
        header = "time," + ",".join([f"y{i+1}(t)" for i in range(8)])
        np.savetxt(full_result_path, 
                   results_to_save, 
                   delimiter=',', 
                   header=header, 
                   comments='')
        print(f"CSV saved successfully at: {os.path.abspath(full_result_path)}")

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(8):
        ax.plot(times, solutions[:,i], label=f"y{i+1}(t)")
    ax.set_title("HIRES Model: Solutions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    if args.save_png:
        filename = "hires_model.png"
        full_result_path = os.path.join(output_directory, filename)
        plt.savefig(full_result_path)
        print(f"PNG saved successfully at: {os.path.abspath(full_result_path)}")

    plt.tight_layout()
    plt.show()
