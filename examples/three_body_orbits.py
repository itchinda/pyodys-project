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
    tf = 100.0
    m3 = 4
    v1, v2 = 0.9911981217, 0.7119472124

    # Scenario 7
    # tf = 50
    # v1, v2 = 0.3420307307, 0.1809369236
    # m3 = 0.5

    # scenario 8
    # tf = 100
    # v1, v2 = 0.5337490177, 0.3041674607
    # m3 = 0.75

    # scenario 9
    # tf = 100
    # v1, v2 = 0.3477173243, 0.0739384079
    # m3 = 0.75

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
        method="dirk65",
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


    fig, ax = plt.subplots(3, 2, figsize=(14, 9))
    # Prepare subplots
    ax_anim = ax[0][0]   # This will now host the animation
    ax_orbits = ax[0][1]
    ax_x = ax[1][0]
    ax_y = ax[1][1]
    ax_velocity = ax[2][0]
    ax_energy = ax[2][1]

    # --- ANIMATED ORBIT ---
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.grid(True)
    ax_anim.set_title("Three-Body Gravitational Motion")
    ax_anim.set_xlabel("$x$")
    ax_anim.set_ylabel("$y$")

    # Data arrays
    x_all = solutions[:, [0, 2, 4]]
    y_all = solutions[:, [1, 3, 5]]
    x_min, x_max = np.min(x_all) * 1.2, np.max(x_all) * 1.2
    y_min, y_max = np.min(y_all) * 1.2, np.max(y_all) * 1.2
    ax_anim.set_xlim(x_min, x_max)
    ax_anim.set_ylim(y_min, y_max)

    # Body markers, trails, and COM
    colors = ['b', 'r', 'k']
    points = [ax_anim.plot([], [], 'o', color=c, markersize=8)[0] for c in colors]
    trails = [ax_anim.plot([], [], '-', color=c, lw=1, alpha=0.6)[0] for c in colors]
    COM, = ax_anim.plot([], [], 'go', markersize=5, alpha=0.4, label='COM')
    ax_anim.legend()

    # Phase portrait (x vs y)
    ax_orbits.plot(solutions[:, 0], solutions[:, 1], 'b-', alpha=0.75, label='Body 1')
    ax_orbits.plot(solutions[:, 2], solutions[:, 3], 'r-', alpha=0.75, label='Body 2')
    ax_orbits.plot(solutions[:, 4], solutions[:, 5], 'k-', alpha=0.75, label='Body 3')
    ax_orbits.set_xlabel("$x$")
    ax_orbits.set_ylabel("$y$")
    ax_orbits.set_title("Three-Body System Orbits")
    ax_orbits.grid(True)
    ax_orbits.legend()

    # --- X positions ---
    ax_x.plot(times, solutions[:, 0], 'b-', label='Body 1')
    ax_x.plot(times, solutions[:, 2], 'r-', label='Body 2')
    ax_x.plot(times, solutions[:, 4], 'k-', label='Body 3')
    ax_x.set_xlabel("Time")
    ax_x.set_ylabel("$x$")
    ax_x.set_title("$x$-coordinates")
    ax_x.grid(True)
    ax_x.legend()

    # --- Y positions ---
    ax_y.plot(times, solutions[:, 1], 'b-', label='Body 1')
    ax_y.plot(times, solutions[:, 3], 'r-', label='Body 2')
    ax_y.plot(times, solutions[:, 5], 'k-', label='Body 3')
    ax_y.set_xlabel("Time")
    ax_y.set_ylabel("$y$")
    ax_y.set_title("$y$-coordinates")
    ax_y.grid(True)
    ax_y.legend()


    # Phase portrait (x vs y)
    ax_velocity.plot(times, np.sqrt( solutions[:, 6]**2 + solutions[:, 7]**2 ), 'b-', alpha=0.75, label='Body 1')
    ax_velocity.plot(times, np.sqrt( solutions[:, 8]**2 + solutions[:, 9]**2 ), 'r-', alpha=0.75, label='Body 2')
    ax_velocity.plot(times, np.sqrt( solutions[:,10]**2 + solutions[:,11]**2 ), 'k-', alpha=0.75, label='Body 3')
    ax_velocity.set_xlabel("Time")
    ax_velocity.set_ylabel(r"$\|\boldsymbol{v}\|$")
    ax_velocity.set_title("Three-Body Velocity")
    ax_velocity.grid(True)
    ax_velocity.legend()

     # --- Total energy ---
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda y, _: f'{y:.3e}')
    ax_energy.plot(times, energies, 'k-')
    ax_energy.set_xlabel("Time")
    ax_energy.set_ylabel(r"$E_\mathrm{tot}$ ")
    ax_energy.set_title(r"Energy Conservation: "
                        r"$E_\mathrm{tot} = \sum_{i=1}^{3} \frac{1}{2} m_i \|\boldsymbol{v}_i\|^2 " 
                        r"- \sum_{i=1}^{3} \sum_{j>i}^{3} G \frac{m_i m_j}{\left\|\mathbf{r}_i - \mathbf{r}_j\right\|}$"
                    )
    ax_energy.grid(True)
    ax_energy.yaxis.set_major_formatter(formatter)

    # --- ANIMATION UPDATE ---
    import matplotlib.animation as animation

    def update(frame):
        if frame == 0:
            for p, t in zip(points, trails):
                p.set_data([], [])
                t.set_data([], [])
            COM.set_data([], [])
            return points + trails + [COM]

        for i, (p, t) in enumerate(zip(points, trails)):
            xi, yi = x_all[:frame, i], y_all[:frame, i]
            p.set_data([xi[-1]], [yi[-1]])
            t.set_data(xi, yi)

        # Center of mass
        total_m = system.m1 + system.m2 + system.m3
        cx = (system.m1 * x_all[frame-1, 0] +
              system.m2 * x_all[frame-1, 1] +
              system.m3 * x_all[frame-1, 2]) / total_m
        cy = (system.m1 * y_all[frame-1, 0] +
              system.m2 * y_all[frame-1, 1] +
              system.m3 * y_all[frame-1, 2]) / total_m
        COM.set_data([cx], [cy])

        return points + trails + [COM]

    # --- Launch animation within same figure ---
    interval = 1000 * (times[-1] - times[0]) / len(times)
    step = 5  # skip every 5th frame
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(times), step), interval=interval, blit=True
    )

    plt.tight_layout()

    # from matplotlib.animation import PillowWriter

    # # Save animation as GIF
    # output_path = "three_body_animation.gif"
    # writer = PillowWriter(fps=30)  # frames per second
    # ani.save(output_path, writer=writer)
    # #ani.save(output_path, writer="ffmpeg", fps=15)
    # print(f"Animation saved as: {os.path.abspath(output_path)}")
    plt.show()
