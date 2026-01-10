import numpy as np
import math

# --- Constants (Teori) ---
G = 6.674e-11
M = 5.972e24          # Earth Mass
MU = G * M            # Standard Gravitational Parameter
EARTH_RADIUS = 6_371_000

def get_acceleration(r_position: np.ndarray) -> np.ndarray:
    """Calculates gravitational acceleration based on Newton's Law."""
    dist = np.linalg.norm(r_position)
    if dist == 0:
        return np.zeros_like(r_position)
    # F = G*M*m / r^2  ->  a = G*M / r^2
    # Vector form: a = -mu * r / |r|^3
    return -MU * r_position / (dist**3)

def rk4_step(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta 4 Integration.
    This is the 'Method' part of your report.
    """
    # k1
    a1 = get_acceleration(r)
    v1 = v

    # k2
    r2 = r + 0.5 * v1 * dt
    v2 = v + 0.5 * a1 * dt
    a2 = get_acceleration(r2)

    # k3
    r3 = r + 0.5 * v2 * dt
    v3 = v + 0.5 * a2 * dt
    a3 = get_acceleration(r3)

    # k4
    r4 = r + v3 * dt
    v4 = v + a3 * dt
    a4 = get_acceleration(r4)

    # Final weighted average
    r_new = r + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    v_new = v + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)

    return r_new, v_new