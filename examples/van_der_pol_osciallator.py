import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pyodys import ODEProblem, PyodysSolver, extract_args

# Define Van-der-pol System
class VanDerPol(ODEProblem):
    def __init__(self, t_init, t_final, initial_state, mu=10):
        # Call the parent constructor
        super().__init__(t_init, t_final, initial_state)
        # Specific Lorenz System Parameters
        self.mu  = mu
        self.nb_of_jac_call=0
        
    def evaluate_at(self, t, u):
        # u, state at time t: u = [x, y]
        x, y = u
        
        # Define the derivatives dx/dt, dy/dt
        dxdt = y
        dydt = self.mu * ((1-x**2)*y ) - x
        
        # Returns the derivatives in a Numpy Array
        return np.array([dxdt, dydt])
    
    def jacobian_at(self, t, u):
        x, y = u
        jacobian_at = np.array([
            [0.0, 1],
            [  self.mu*(-2*x*y - 1), self.mu*(1-x**2)]
        ])
        self.nb_of_jac_call+=1

        return jacobian_at

if __name__ == '__main__':
    
    args = extract_args(
        description = "Solve the Van der Pol System.",
        method = 'esdirk64',
        fixed_step = None,
        first_step = None,
        final_time = 40,
        rtol = 1e-10,
        atol = 1e-10,
        min_step = 1e-8,
        max_step = 100
    )

    # Initial conditions
    t0 = 0.0
    tf = args.final_time
    u0 = [2.0, 0.0]
    mu = 10
    system = VanDerPol(t0, tf, u0, mu)

    # solver
    solver = PyodysSolver(
                    method = args.method,
                    fixed_step= args.fixed_step,
                    first_step = args.first_step,
                    adaptive = args.adaptive,
                    rtol = args.rtol,
                    atol = args.atol,
                    min_step = args.min_step,
                    max_step = args.max_step,
                    nsteps_max=100000,
                    verbose=args.verbose
    )


    # Solve the system
    start=time.time()
    times, solutions = solver.solve( system )
    elapsed=time.time()-start
    print(f"Python EDOs runtime: {elapsed:.4f} seconds")

    print(f"Nb of Jacobian call: {system.nb_of_jac_call}")
    
    if args.save_csv or args.save_png:
        output_directory = "van_der_pol_results"
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
        filename = "van_der_pol.csv"
        full_result_path = os.path.join(output_directory, filename)
        results_to_save = np.column_stack((times, solutions))
        header = "time,x(t),y(t)"
        np.savetxt(full_result_path, 
                   results_to_save, 
                   delimiter=',', 
                   header=header, 
                   comments='')
        print(f"CSV saved successfully at: {os.path.abspath(full_result_path)}")
    
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(times, solutions[:,0], 'b.-', markersize=2, label='x(t)')
    ax1.plot(times, solutions[:,1], 'r-', markersize=2, label='y(t)')
    ax1.set_title("Van-Der-Pol Oscillator: Solutions")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    if args.save_png:
        print("Saving to PNG...")
        filename = "van_der_pol.png"
        full_result_path = os.path.join(output_directory, filename)
        plt.savefig(full_result_path)
        print(f"PNG saved successfully at: {os.path.abspath(full_result_path)}")

    plt.tight_layout()
    plt.show()


    # Plot the phase portrait
    plt.figure(figsize=(7, 6))
    plt.plot(solutions[:, 0], solutions[:, 1], 'b-')
    plt.title(fr"Van-Der-Pol Oscillator Phase Portrait ($\mu = {mu}$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()