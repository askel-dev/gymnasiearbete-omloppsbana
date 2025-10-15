"""Parameter sweep that classifies orbit types for different launch setups."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Physics constants matching the interactive simulator
G = 6.674e-11
M = 5.972e24
MU = G * M
EARTH_RADIUS = 6_371_000
R0 = np.array([7_000_000.0, 0.0])

# Sweep parameters
SPEED_RANGE = (6_500.0, 12_500.0)
ANGLE_RANGE = (0.0, math.pi / 2.0)
GRID_POINTS = 25
DT = 0.2
MAX_STEPS = 20_000
ESCAPE_RADIUS_FACTOR = 20.0
ENERGY_TOL = 1e-4

FIGURES_DIR = Path("figures")


def accel(r: np.ndarray) -> np.ndarray:
    rmag = math.hypot(r[0], r[1])
    return -MU * r / (rmag ** 3)


def rk4_step(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = accel(r)
    k1_r = v
    k1_v = a1

    a2 = accel(r + 0.5 * dt * k1_r)
    k2_r = v + 0.5 * dt * k1_v
    k2_v = a2

    a3 = accel(r + 0.5 * dt * k2_r)
    k3_r = v + 0.5 * dt * k2_v
    k3_v = a3

    a4 = accel(r + dt * k3_r)
    k4_r = v + dt * k3_v
    k4_v = a4

    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next


def energy_specific(r: np.ndarray, v: np.ndarray) -> float:
    rmag = math.hypot(r[0], r[1])
    vmag2 = v[0] * v[0] + v[1] * v[1]
    return 0.5 * vmag2 - MU / rmag


def classify_point(speed: float, angle_rad: float, escape_radius: float) -> int:
    r = R0.copy()
    v = np.array([
        speed * math.cos(angle_rad),
        speed * math.sin(angle_rad),
    ])

    for _ in range(MAX_STEPS):
        r, v = rk4_step(r, v, DT)
        rmag = math.hypot(r[0], r[1])
        if rmag <= EARTH_RADIUS:
            return 0  # impact counts as bound orbit in this classification
        eps = energy_specific(r, v)
        if eps > 0.0 and rmag > escape_radius:
            return 2
    eps = energy_specific(r, v)
    if eps < -ENERGY_TOL:
        return 0
    if eps > ENERGY_TOL:
        return 2
    return 1


def run_sweep() -> np.ndarray:
    speeds = np.linspace(SPEED_RANGE[0], SPEED_RANGE[1], GRID_POINTS)
    angles = np.linspace(ANGLE_RANGE[0], ANGLE_RANGE[1], GRID_POINTS)
    escape_radius = ESCAPE_RADIUS_FACTOR * float(np.linalg.norm(R0))

    results = np.zeros((angles.size, speeds.size), dtype=int)
    total = angles.size * speeds.size
    print(f"Simulerar {total} kombinationer ...")
    processed = 0
    for i, angle in enumerate(angles):
        for j, speed in enumerate(speeds):
            results[i, j] = classify_point(speed, angle, escape_radius)
            processed += 1
        print(f"  Rad {i + 1}/{angles.size} klar ({processed}/{total})")
    return speeds, angles, results


def plot_heatmap(speeds: np.ndarray, angles: np.ndarray, results: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#2f9e44", "#ffd43b", "#f03e3e"])
    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [speeds.min(), speeds.max(), math.degrees(angles.min()), math.degrees(angles.max())]
    im = ax.imshow(
        results,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
    )
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Ellips", "Parabel", "Hyperbel"])
    ax.set_xlabel("Starthastighet [m/s]")
    ax.set_ylabel("Startvinkel [grader]")
    ax.set_title("Bantyp per starthastighet och vinkel")
    fig.tight_layout()
    output_path = FIGURES_DIR / "sweep_heatmap.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Heatmap sparad till {output_path}")


def main() -> None:
    speeds, angles, results = run_sweep()
    plot_heatmap(speeds, angles, results)


if __name__ == "__main__":
    main()
