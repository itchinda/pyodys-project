import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from pyodys import ODEProblem, PyodysSolver, hermite_interpolate

# =====================================================
# --- New Rescaling Function ---
# =====================================================

def rescale_moon_position(positions_array, scale_factor):
    """
    Applies distance scaling to the Moon's position (index 2) relative to the Earth (index 1).
    
    Args:
        positions_array (np.array): A 2D array of positions (N_steps, 3_bodies, 2_coords).
                                    (..., 0) = Sun, (..., 1) = Earth, (..., 2) = Moon.
        scale_factor (float): The scaling factor 'f'. The distance is multiplied by (1 + f).

    Returns:
        np.array: The new array with the Moon's coordinates scaled.
    """
    # Create a copy to modify
    positions_viz = positions_array.copy() 
    
    # Earth and Moon positions are at indices 1 and 2
    P_earth = positions_viz[:, 1, :]
    P_moon = positions_viz[:, 2, :]
    
    # Vector from Earth to Moon: V = Pm - Pe
    vec_em = P_moon - P_earth
    
    # Apply the scaling: Pm_new = Pe + (1 + f) * V
    # This multiplies the Earth-Moon distance by (1 + f)
    P_moon_new = P_earth + vec_em * (1.0 + scale_factor)
    
    # Update the Moon's position in the visualization array
    positions_viz[:, 2, :] = P_moon_new
    
    return positions_viz


# =====================================================
# 1. Define normalized Three-Body ODE
# =====================================================
class ThreeBodySystem(ODEProblem):
    def __init__(self, t_init, t_final, initial_state, G=1.0, m1=1.0, m2=1.0, m3=1.0):
        super().__init__(t_init, t_final, initial_state)
        self.m1, self.m2, self.m3, self.G = m1, m2, m3, G

    def evaluate_at(self, t, u):
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = u
        r1, r2, r3 = np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])
        r12, r13, r23 = np.linalg.norm(r1 - r2), np.linalg.norm(r1 - r3), np.linalg.norm(r2 - r3)

        # velocities
        dx1dt, dy1dt = vx1, vy1
        dx2dt, dy2dt = vx2, vy2
        dx3dt, dy3dt = vx3, vy3

        # accelerations
        dvx1dt = self.G * (self.m2 * (x2-x1)/r12**3 + self.m3 * (x3-x1)/r13**3)
        dvy1dt = self.G * (self.m2 * (y2-y1)/r12**3 + self.m3 * (y3-y1)/r13**3)
        dvx2dt = self.G * (self.m1 * (x1-x2)/r12**3 + self.m3 * (x3-x2)/r23**3)
        dvy2dt = self.G * (self.m1 * (y1-y2)/r12**3 + self.m3 * (y3-y2)/r23**3)
        dvx3dt = self.G * (self.m1 * (x1-x3)/r13**3 + self.m2 * (x2-x3)/r23**3)
        dvy3dt = self.G * (self.m1 * (y1-y3)/r13**3 + self.m2 * (y2-y3)/r23**3)

        return np.array([
            dx1dt, dy1dt, dx2dt, dy2dt, dx3dt, dy3dt,
            dvx1dt, dvy1dt, dvx2dt, dvy2dt, dvx3dt, dvy3dt
        ])


# =====================================================
# 2. Energy helper
# =====================================================
def total_energy(u, m1, m2, m3, G=1.0):
    r1, r2, r3 = u[:2], u[2:4], u[4:6]
    v1, v2, v3 = u[6:8], u[8:10], u[10:12]
    KE = 0.5 * (m1*np.dot(v1,v1) + m2*np.dot(v2,v2) + m3*np.dot(v3,v3))
    r12, r13, r23 = np.linalg.norm(r1-r2), np.linalg.norm(r1-r3), np.linalg.norm(r2-r3)
    PE = -G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    return KE + PE


# =====================================================
# 3. Normalized setup (Earth-Moon-Sun)
# =====================================================
def earth_moon_sun_normalized():
    # ... (Real constants and normalization constants remain the same) ...
    AU = 1.496e11         # m
    day = 86400           # s
    G_SI = 6.67430e-11
    M_sun = 1.98847e30
    M_earth = 5.972e24
    M_moon = 7.3477e22

    # Canonical scaling
    L0 = AU
    M0 = M_sun
    T0 = np.sqrt(L0**3 / (G_SI*M0))  # T0 is the time unit (~581 days)
    G = 1.0  # Normalized G

    # Normalized masses
    m1 = 1.0                              # Sun
    m2 = M_earth / M_sun                  # Earth
    m3 = M_moon / M_sun                   # Moon
    
    # Normalized Earth-Moon distance
    a_em = 0.00257 
    
    # 1. Earth-Moon System Parameters
    M_em = m2 + m3                                # Total E-M mass
    # Distance of Earth's center from the E-M Barycenter (B-C)
    r2_bc = a_em * m3 / M_em
    # Distance of Moon's center from the E-M Barycenter (B-C)
    r3_bc = a_em * m2 / M_em 

    # 2. Initial POSITIONS
    # Assume the E-M Barycenter is at x_bc = 1.0, y_bc = 0.0 (1 AU from Sun)
    x_bc, y_bc = 1.0, 0.0
    
    # Earth position relative to the Sun
    x1, y1 = 0.0, 0.0 
    x2, y2 = x_bc - r2_bc, y_bc 
    # Moon position relative to the Sun
    x3, y3 = x_bc + r3_bc, y_bc 

    # 3. Initial VELOCITIES
    
    # Velocity of the E-M Barycenter orbiting the Sun at 1 AU
    v_bc = np.sqrt(G * m1 / x_bc) 

    # Circular orbital velocity of the Earth/Moon around the E-M B-C
    # Use the full E-M mass (m2+m3) and the respective barycenter distances
    # This is Kepler's Third Law (or simplified Two-Body equation) applied to the E-M system
    v_rel_em = np.sqrt(G * M_em / a_em)
    
    v2_rel_bc = v_rel_em * m3 / M_em  # Earth velocity relative to B-C
    v3_rel_bc = v_rel_em * m2 / M_em  # Moon velocity relative to B-C

    # Sun velocity (assume Sun is stationary or nearly so, for simplicity)
    vx1, vy1 = 0.0, 0.0
    
    # Earth absolute velocity (B-C velocity + relative velocity)
    vx2, vy2 = 0.0, v_bc - v2_rel_bc  
    
    # Moon absolute velocity (B-C velocity + relative velocity)
    # The Moon is on the outside, so its relative velocity adds to the orbital velocity
    vx3, vy3 = 0.0, v_bc + v3_rel_bc 
    
    # NOTE: The formula np.sqrt(G * M_em / a_em) is the KEY correction.
    # The term v_rel_em is the orbital speed required for the Earth-Moon separation.

    u0 = np.array([x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3])
    t0, tf = 0.0, 4.0 * 2*np.pi  # ~2 years of simulation
    return t0, tf, u0, G, m1, m2, m3


# =====================================================
# 4. Run simulation
# =====================================================
if __name__ == "__main__":
    t0, tf, u0, G, m1, m2, m3 = earth_moon_sun_normalized()
    system = ThreeBodySystem(t0, tf, u0, G, m1, m2, m3)

    solver = PyodysSolver(method="dirk65",
                          rtol=1e-13,
                          atol=1e-13,
                          verbose=True
                        )

    start = time.time()
    times, solutions = solver.solve(system)
    print(f"Integration time: {time.time()-start:.3f}s")

    energies = np.array([total_energy(u, m1, m2, m3, G) for u in solutions])
    print(f"Relative ΔE = {(np.max(energies)-np.min(energies))/np.mean(energies):.3e}")

    # --- Rescale Trajectories for Visualization ---
    # Reshape solutions from (N_steps, 12_vars) to (N_steps, 3_bodies, 2_coords)
    physical_pos = solutions[:, 0:6].reshape(len(times), 3, 2)
    
    # Define a scaling factor for visualization: f = 50.0 means 51x distance
    VISUAL_SCALING_FACTOR = 30.0 
    
    # Get the scaled positions
    scaled_pos = rescale_moon_position(physical_pos, VISUAL_SCALING_FACTOR)
    
    # Extract coordinates for plotting: Sun (0), Earth (1), Moon (2)
    x_viz, y_viz = scaled_pos[:, :, 0], scaled_pos[:, :, 1]
    # ------------------------------------------------------------------

    # =====================================================
    # 5. Plot orbits + energy
    # =====================================================
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Orbit plot (Using scaled positions)
    ax[0,0].plot(x_viz[:,0], y_viz[:,0], 'y-', label='Sun', markersize=3)
    ax[0,0].plot(x_viz[:,1], y_viz[:,1], 'b-', label='Earth', markersize=3)
    ax[0,0].plot(x_viz[:,2], y_viz[:,2], 'gray', label=f'Moon (Scaled $\\times${1+VISUAL_SCALING_FACTOR:.0f})', linewidth=1)
    
    # Add a marker for the Earth and Moon at the end
    ax[0,0].plot(x_viz[-1,1], y_viz[-1,1], 'bo', markersize=5)
    ax[0,0].plot(x_viz[-1,2], y_viz[-1,2], 'o', color='gray', markersize=2)
    
    ax[0,0].set_title("Normalized Three-Body Orbits (Sun-Earth-Moon, SCALED Moon)")
    ax[0,0].set_xlabel("$x$ (AU)")
    ax[0,0].set_ylabel("$y$ (AU)")
    ax[0,0].axis("equal")
    ax[0,0].legend()
    ax[0,0].grid(True)

    # Total Energy
    ax[0,1].plot(times, energies, 'k-')
    ax[0,1].set_title("Total Energy (Normalized Units)")
    ax[0,1].set_xlabel("t")
    ax[0,1].set_ylabel("E")

    # Positions (x-coordinates)
    x_phys, y_phys = physical_pos[:, :, 0], physical_pos[:, :, 1]
    ax[1,0].plot(times, x_phys[:,1], 'b-', label='Earth x')
    ax[1,0].plot(times, x_phys[:,2], 'gray', label='Moon x')
    ax[1,0].set_title("x-coordinates (Physical)")
    ax[1,0].legend()
    ax[1,0].grid(True)

    # Positions (y-coordinates)
    ax[1,1].plot(times, y_phys[:,1], 'b-', label='Earth y')
    ax[1,1].plot(times, y_phys[:,2], 'gray', label='Moon y')
    ax[1,1].set_title("y-coordinates (Physical)")
    ax[1,1].legend()
    ax[1,1].grid(True)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 6. Optional Animation (NOW USES SCALED POSITIONS)
    # =====================================================
    fig_anim, ax_anim = plt.subplots(figsize=(6,6))
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title(f"Sun-Earth-Moon (Moon Distance Scaled $\\times${1+VISUAL_SCALING_FACTOR:.0f})")
    
    # Set limits based on the scaled positions to ensure the Moon is visible
    ax_anim.set_xlim(np.min(x_viz)*1.1, np.max(x_viz)*1.1)
    ax_anim.set_ylim(np.min(y_viz)*1.1, np.max(y_viz)*1.1)

    colors = ['y','b','gray']
    labels = ['Sun','Earth',f'Moon (Scaled $\\times${1+VISUAL_SCALING_FACTOR:.0f})']
    markersizes = [14, 8, 5]
    
    # Points list: All 3 bodies need a marker
    points = [ax_anim.plot([], [], 'o', color=c, markersize=m, label=l)[0] for c, l, m in zip(colors, labels, markersizes)]
    trails = [ax_anim.plot([], [], '-', color=c, lw=1, alpha=0.6)[0] for c in colors[:2]] 
    #                                                                           ^^^^ Only the first 2 colors

    ax_anim.legend()

    def update(frame):
        for i in range(3):
            points[i].set_data([x_viz[frame,i]], [y_viz[frame,i]])
            
        for i in range(2): 
             trails[i].set_data(x_viz[:frame,i], y_viz[:frame,i])
             
        return points + trails

    ani = animation.FuncAnimation(fig_anim, update, frames=len(times), interval=20, blit=True)
    plt.show()