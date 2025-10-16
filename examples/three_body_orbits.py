import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pyodys import ODEProblem, PyodysSolver, extract_args

# Define the Three-Body system
# Reference: https://arxiv.org/pdf/1709.04775
class ThreeBodySystem(ODEProblem):
    def __init__(self, t_init, t_final, initial_state, G=1.0, m1=1.0, m2=1.0, m3=1.0):
        super().__init__(t_init, t_final, initial_state)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G

    def evaluate_at(self, t, u):
        # State vector at time t: u = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = u

        # Position vectors
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r3 = np.array([x3, y3])
        
        # Distances between bodies
        r12 = np.linalg.norm(r1 - r2)
        r13 = np.linalg.norm(r1 - r3)
        r23 = np.linalg.norm(r2 - r3)

        # Velocities
        dx1dt, dy1dt = vx1, vy1
        dx2dt, dy2dt = vx2, vy2
        dx3dt, dy3dt = vx3, vy3

        # Accelerations from Newton's law of gravitation
        dvx1dt = self.G*(self.m2*(x2-x1)/(r12**3) + self.m3*(x3-x1)/(r13**3))
        dvy1dt = self.G*(self.m2*(y2-y1)/(r12**3) + self.m3*(y3-y1)/(r13**3))
        dvx2dt = self.G*(self.m1*(x1-x2)/(r12**3) + self.m3*(x3-x2)/(r23**3))
        dvy2dt = self.G*(self.m1*(y1-y2)/(r12**3) + self.m3*(y3-y2)/(r23**3))
        dvx3dt = self.G*(self.m1*(x1-x3)/(r13**3) + self.m2*(x2-x3)/(r23**3))
        dvy3dt = self.G*(self.m1*(y1-y3)/(r13**3) + self.m2*(y2-y3)/(r23**3))

        # Return derivatives as a single array
        return np.array([
            dx1dt, dy1dt, dx2dt, dy2dt, dx3dt, dy3dt,
            dvx1dt, dvy1dt, dvx2dt, dvy2dt, dvx3dt, dvy3dt
        ], dtype=float)

# Function to compute the total energy of the system
def total_energy(u, m1, m2, m3, G=1.0):
    # Extract positions and velocities
    r1, r2, r3 = u[:2], u[2:4], u[4:6]
    v1, v2, v3 = u[6:8], u[8:10], u[10:12]

    # Kinetic energy
    KE = 0.5*(m1*np.dot(v1,v1) + m2*np.dot(v2,v2) + m3*np.dot(v3,v3))

    # Potential energy
    r12 = np.linalg.norm(r1 - r2)
    r13 = np.linalg.norm(r1 - r3)
    r23 = np.linalg.norm(r2 - r3)
    PE = -G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)

    return KE + PE

if __name__ == '__main__':
    
    # Initial simulation parameters
    t0 = 0.0
    G = 1.0
    m1, m2 = 1.0, 1.0

    # Controlling parameters from Ref. https://arxiv.org/pdf/1709.04775
    # The following blocks define alternative scenarios. Uncomment to run a different case.

    # Scenario 1
    # tf = 50.0
    # m3 = 0.5
    # v1, v2 = 0.2009656237,  0.2431076328

    # Scenario 2 (currently active)
    tf = 100.0
    m3 = 0.5
    v1, v2 = 0.2138410831, 0.0542938396

    # Scenario 3
    # tf = 100.0
    # m3 = 0.75
    # v1, v2 = 0.4101378717, 0.1341894173

    # Scenario 4
    # tf = 100.0
    # m3 = 2
    # v1, v2 = 0.6649107583, 0.8324167864

    # Scenario 5
    # tf = 16.0
    # m3 = 2
    # v1, v2 = 0.3057224330, 0.5215124257

    # Scenario 6
    # tf = 100.0
    # m3 = 4
    # v1, v2 = 0.9911981217, 0.7119472124

    # Initial positions
    x10, y10 = -1, 0
    x20, y20 = 1, 0
    x30, y30 = 0, 0

    # Initial velocities
    vx10, vy10 = v1, v2
    vx20, vy20 = v1, v2
    # Ensure zero total momentum
    vx30, vy30 = -v1*(m1+m2)/m3, -v2*(m1+m2)/m3

    # Initial state vector
    u0 = [x10, y10, x20, y20, x30, y30, vx10, vy10, vx20, vy20, vx30, vy30]
    system = ThreeBodySystem(t0, tf, u0, G, m1, m2, m3)

    # Set up solver
    solver = PyodysSolver(
        method="esdirk64",
        fixed_step=1e-4,
        first_step=1e-8,
        adaptive=True,
        rtol=1e-10,
        atol=1e-10,
        verbose=True
    )

    # Solve the system
    start = time.time()
    times, solutions = solver.solve(system)
    elapsed = time.time() - start
    print(f"Python ODEs runtime: {elapsed:.4f} seconds")

    # Compute total energy over time
    energies = np.array([total_energy(u, system.m1, system.m2, system.m3, system.G) 
                         for u in solutions])
    print(f"Max total energy: {np.max(energies)}, Min total energy: {np.min(energies)}")

    # Create 2x2 subplot grid
    fig, ax = plt.subplots(2, 2, figsize=(16, 6))

    # Phase portrait (x vs y)
    ax[0][0].plot(solutions[:, 0], solutions[:, 1], 'b-', alpha=0.75, label='Body 1')
    ax[0][0].plot(solutions[:, 2], solutions[:, 3], 'r-', alpha=0.75, label='Body 2')
    ax[0][0].plot(solutions[:, 4], solutions[:, 5], 'k-', alpha=0.75, label='Body 3')
    ax[0][0].set_xlabel("$x$")
    ax[0][0].set_ylabel("$y$")
    ax[0][0].set_title("Three-Body System Orbits")
    ax[0][0].grid(True)
    ax[0][0].legend()

    # Total energy over time
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda y, _: f'{y:.3e}')
    ax[0][1].plot(times, energies, 'k-')
    ax[0][1].set_xlabel("Time")
    ax[0][1].set_ylabel("Total Energy")
    ax[0][1].set_title(
        r"Three-Body System: "
        r"$E_\mathrm{tot} = \sum_{i=1}^{3} \frac{1}{2} m_i \|\boldsymbol{v}_i\|^2 "
        r"- \sum_{i=1}^{3} \sum_{j>i}^{3} G \frac{m_i m_j}{\left\|\mathbf{r}_i - \mathbf{r}_j\right\|}$"
    )
    ax[0][1].grid(True)
    ax[0][1].yaxis.set_major_formatter(formatter)

    # x-coordinates over time
    ax[1][0].plot(times, solutions[:, 0], 'b-', label='Body 1')
    ax[1][0].plot(times, solutions[:, 2], 'r-', label='Body 2')
    ax[1][0].plot(times, solutions[:, 4], 'k-', label='Body 3')
    ax[1][0].set_xlabel("Time")
    ax[1][0].set_ylabel("$x$")
    ax[1][0].set_title("Three-Body System: $x$-coordinate vs Time")
    ax[1][0].grid(True)
    ax[1][0].legend()

    # y-coordinates over time
    ax[1][1].plot(times, solutions[:, 1], 'b-', label='Body 1')
    ax[1][1].plot(times, solutions[:, 3], 'r-', label='Body 2')
    ax[1][1].plot(times, solutions[:, 5], 'k-', label='Body 3')
    ax[1][1].set_xlabel("Time")
    ax[1][1].set_ylabel("$y$")
    ax[1][1].set_title("Three-Body System: $y$-coordinate vs Time")
    ax[1][1].grid(True)
    ax[1][1].legend()

    plt.tight_layout()
    plt.show()
