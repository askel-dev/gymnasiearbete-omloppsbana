"""Parameter sweep that classifies orbit types for different launch setups (fast or simulated)."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ===========================
# PHYSICS CONSTANTS
# ===========================
G = 6.674e-11
M = 5.972e24
MU = G * M
EARTH_RADIUS = 6_371_000
R0 = np.array([7_000_000.0, 0.0])  # 7000 km orbit start distance

# ===========================
# SWEEP SETTINGS
# ===========================
SPEED_RANGE = (6_500.0, 12_500.0)   # m/s
ANGLE_RANGE = (0.0, math.pi / 2.0)  # radians
SPEED_POINTS = 300                  # horizontal resolution (x-axis)
ANGLE_POINTS = 20                   # vertical resolution (y-axis)

MODE = "fast"       # "fast" = analytical (1 sec), "simulated" = RK4 (slower)
DT = 1.0
MAX_STEPS = 5000
ESCAPE_RADIUS_FACTOR = 8.0

ENERGY_TOL = 3e5  # ±300 000 J/kg → synlig gul zon

FIGURES_DIR = Path("figures")

# ===========================
# PHYSICS HELPERS
# ===========================
def accel(r: np.ndarray) -> np.ndarray:
    rmag = math.hypot(r[0], r[1])
    return -MU * r / (rmag ** 3)

def rk4_step(r: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = accel(r); k1_r = v; k1_v = a1
    a2 = accel(r + 0.5 * dt * k1_r); k2_r = v + 0.5 * dt * k1_v; k2_v = a2
    a3 = accel(r + 0.5 * dt * k2_r); k3_r = v + 0.5 * dt * k2_v; k3_v = a3
    a4 = accel(r + dt * k3_r);      k4_r = v + dt * k3_v;      k4_v = a4
    r_next = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_next, v_next

def energy_specific(r: np.ndarray, v: np.ndarray) -> float:
    rmag = math.hypot(r[0], r[1])
    vmag2 = v[0]**2 + v[1]**2
    return 0.5 * vmag2 - MU / rmag

# ===========================
# CLASSIFICATION
# ===========================
def classify_fast(speed: float) -> int:
    """Classify orbit type from start energy only."""
    eps0 = 0.5 * speed**2 - MU / np.linalg.norm(R0)
    if eps0 < -ENERGY_TOL:
        return 0  # Ellips
    elif eps0 > ENERGY_TOL:
        return 2  # Hyperbel
    else:
        return 1  # Parabel (inom ±ENERGY_TOL)

def classify_simulated(speed: float, angle_rad: float, escape_radius: float) -> int:
    """Numerical RK4 integration for classification."""
    r = R0.copy()
    v = np.array([speed * math.cos(angle_rad), speed * math.sin(angle_rad)])
    for _ in range(MAX_STEPS):
        r, v = rk4_step(r, v, DT)
        rmag = math.hypot(r[0], r[1])
        if rmag <= EARTH_RADIUS:
            return 0
        eps = energy_specific(r, v)
        if eps > 0.0 and rmag > escape_radius:
            return 2
    eps = energy_specific(r, v)
    if eps < -ENERGY_TOL:
        return 0
    elif eps > ENERGY_TOL:
        return 2
    else:
        return 1

# ===========================
# MAIN SWEEP
# ===========================
def run_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speeds = np.linspace(*SPEED_RANGE, SPEED_POINTS)
    angles = np.linspace(*ANGLE_RANGE, ANGLE_POINTS)
    escape_radius = ESCAPE_RADIUS_FACTOR * np.linalg.norm(R0)
    results = np.zeros((angles.size, speeds.size), dtype=int)

    total = angles.size * speeds.size
    print(f"\n--- Kör parameter-svep ({MODE}) ---")
    print(f"Totalt {total} punkter ({SPEED_POINTS} hastigheter × {ANGLE_POINTS} vinklar)")

    processed = 0
    for i, angle in enumerate(angles):
        for j, speed in enumerate(speeds):
            if MODE == "fast":
                results[i, j] = classify_fast(speed)
            else:
                results[i, j] = classify_simulated(speed, angle, escape_radius)
            processed += 1
        if i % 2 == 0:
            pct = 100 * processed / total
            print(f"  {i+1}/{angles.size} rader klara ({pct:.1f}%)")

    return speeds, angles, results

# ===========================
# PLOTTNING
# ===========================
def plot_heatmap(speeds: np.ndarray, angles: np.ndarray, results: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#2f9e44", "#ffd43b", "#f03e3e"])  # green, yellow, red

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [speeds.min(), speeds.max(), math.degrees(angles.min()), math.degrees(angles.max())]
    im = ax.imshow(results, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, vmin=-0.5, vmax=2.5)
    cbar = fig.colorbar(im, ticks=[0,1,2])
    cbar.ax.set_yticklabels(["Ellips", "Parabel", "Hyperbel"])
    ax.set_xlabel("Starthastighet [m/s]")
    ax.set_ylabel("Startvinkel [grader]")
    ax.set_title(f"Bantyp per starthastighet och vinkel ({MODE})")
    fig.tight_layout()
    out = FIGURES_DIR / f"sweep_heatmap_{MODE}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\n✅ Heatmap sparad till {out}")

# ===========================
# MAIN
# ===========================
def main() -> None:
    v_esc = math.sqrt(2 * MU / np.linalg.norm(R0))
    print(f"Flykthastighet vid {np.linalg.norm(R0)/1000:.0f} km: {v_esc:.2f} m/s")
    print(f"Energitolerans: ±{ENERGY_TOL:.1e} J/kg")
    speeds, angles, results = run_sweep()
    plot_heatmap(speeds, angles, results)

if __name__ == "__main__":
    main()
