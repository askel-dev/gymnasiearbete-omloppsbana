"""Simple real-time orbit visualisation using Matplotlib."""

from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Physical constants & initial state ------------------------------------------------------

G = 6.674e-11  # gravitation constant (SI)
M = 5.972e24  # central mass (Earth) in kg

# Starting position (m): ~ Earth radius + 600 km, on the x-axis
INITIAL_POSITION = np.array([7.0e6, 0.0], dtype=float)

# Starting velocity (m/s): almost circular low Earth orbit in y-direction
INITIAL_VELOCITY = np.array([0.0, 7800.0], dtype=float)

# --- Numerical configuration ----------------------------------------------------------------

DT = 0.25  # physics timestep in seconds
REAL_TIME_SPEED = 60  # simulate 60 s of physics per second of wall time
TRAIL_LENGTH = 2000  # number of points retained in the orbit trail
VEL_VECTOR_SCALE = 500.0  # scale of the velocity vector overlay


def acceleration(position: np.ndarray) -> np.ndarray:
    """Compute the gravitational acceleration at ``position``."""

    distance = np.linalg.norm(position)
    if distance == 0:
        return np.zeros_like(position)
    return -G * M * position / distance**3


def rk4_step(position: np.ndarray, velocity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance the system one Runge–Kutta 4 step."""

    a1 = acceleration(position)
    k1_r = velocity
    k1_v = a1

    a2 = acceleration(position + 0.5 * dt * k1_r)
    k2_r = velocity + 0.5 * dt * k1_v
    k2_v = a2

    a3 = acceleration(position + 0.5 * dt * k2_r)
    k3_r = velocity + 0.5 * dt * k2_v
    k3_v = a3

    a4 = acceleration(position + dt * k3_r)
    k4_r = velocity + dt * k3_v
    k4_v = a4

    next_position = position + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    next_velocity = velocity + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return next_position, next_velocity


@dataclass
class SimulationState:
    """Mutable simulation data used by the Matplotlib animation callback."""

    position: np.ndarray = field(default_factory=lambda: INITIAL_POSITION.copy())
    velocity: np.ndarray = field(default_factory=lambda: INITIAL_VELOCITY.copy())
    time: float = 0.0
    trail_x: deque[float] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    trail_y: deque[float] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))

    def integrate(self, n_steps: int) -> None:
        """Advance the simulation ``n_steps`` times using RK4 integration."""

        for _ in range(max(1, n_steps)):
            self.position, self.velocity = rk4_step(self.position, self.velocity, DT)
            self.time += DT

    def record_trail_point(self) -> None:
        """Store the latest position while keeping the trail length bounded."""

        self.trail_x.append(self.position[0])
        self.trail_y.append(self.position[1])


def setup_axes() -> tuple[plt.Figure, plt.Axes]:
    """Create and configure a square Matplotlib axis for the orbit plot."""

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.set_aspect("equal", "box")
    axis_scale = 1.2 * np.linalg.norm(INITIAL_POSITION)
    axes.set_xlim(-axis_scale, axis_scale)
    axes.set_ylim(-axis_scale, axis_scale)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Omloppsbana i realtid (RK4)")
    return figure, axes


def create_animation() -> FuncAnimation:
    """Build the Matplotlib animation object visualising the orbit."""

    state = SimulationState()
    figure, axes = setup_axes()

    (earth,) = axes.plot([0], [0], "o", markersize=8, label="Jorden")
    (satellite,) = axes.plot([], [], "o", markersize=5, label="Satellit")
    (trail,) = axes.plot([], [], lw=1, alpha=0.8, label="Bana (spår)")
    (velocity_line,) = axes.plot([], [], lw=1, alpha=0.6, label="Hastighet")
    time_text = axes.text(0.02, 0.98, "", transform=axes.transAxes, va="top")
    axes.legend(loc="upper right")

    axis_scale = axes.get_xlim()[1]

    artists = (earth, satellite, trail, velocity_line, time_text)

    def init():  # noqa: D401 - Matplotlib callback signature
        """Reset the artists before the animation starts."""

        satellite.set_data([], [])
        trail.set_data([], [])
        velocity_line.set_data([], [])
        time_text.set_text("")
        return artists

    def update(_frame):  # noqa: D401 - Matplotlib callback signature
        """Advance the simulation and update the plot artists."""

        state.integrate(int(REAL_TIME_SPEED))
        satellite.set_data([state.position[0]], [state.position[1]])

        state.record_trail_point()
        trail.set_data(state.trail_x, state.trail_y)

        vx, vy = state.velocity
        end_x = state.position[0] + vx * VEL_VECTOR_SCALE
        end_y = state.position[1] + vy * VEL_VECTOR_SCALE
        velocity_line.set_data([state.position[0], end_x], [state.position[1], end_y])

        max_extent = max(
            np.abs(state.position[0]), np.abs(state.position[1]), axis_scale
        )
        axes.set_xlim(-max_extent, max_extent)
        axes.set_ylim(-max_extent, max_extent)

        distance = np.linalg.norm(state.position)
        speed = np.linalg.norm(state.velocity)
        time_text.set_text(
            "\n".join(
                (
                    f"t = {state.time:,.0f} s",
                    f"|r| = {distance/1e6:.2f} Mm",
                    f"|v| = {speed:.1f} m/s",
                )
            )
        )

        return artists

    return FuncAnimation(
        figure,
        update,
        init_func=init,
        interval=33,
        blit=True,
    )


def main() -> None:
    """Run the Matplotlib animation in an interactive window."""

    animation = create_animation()
    # Hold a reference to avoid garbage collection of the animation object.
    _ = animation
    plt.show()


if __name__ == "__main__":
    main()