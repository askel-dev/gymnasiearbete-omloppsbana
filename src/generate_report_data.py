"""Generate simulation data for report figures."""
import json
import csv
import math
from pathlib import Path
from datetime import datetime
import numpy as np

from physics import G, M_EARTH, MU_EARTH, EARTH_RADIUS, rk4_step, euler_step


def run_simulation(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    duration: float,
    integrator: str,
    planet_name: str,
    planet_radius: float,
    planet_mass: float,
    scenario_name: str,
) -> Path:
    """Run a headless simulation and save data."""
    mu = G * planet_mass
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data") / "runs" / f"{timestamp}_{scenario_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metadata
    meta = {
        "scenario_key": scenario_name.lower().replace(" ", "_"),
        "scenario_name": scenario_name,
        "planet_name": planet_name,
        "planet_radius": float(planet_radius),
        "R0": r0.tolist(),
        "V0": v0.tolist(),
        "v0": float(np.linalg.norm(v0)),
        "G": G,
        "M": float(planet_mass),
        "mu": float(mu),
        "integrator": integrator,
        "dt_phys": dt,
        "code_version": "report_generation",
    }
    
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Run simulation
    r = r0.copy()
    v = v0.copy()
    t = 0.0
    
    timeseries = []
    events = []
    
    prev_r_mag = np.linalg.norm(r)
    dr_sign_prev = 0
    
    step_func = rk4_step if integrator == "RK4" else euler_step
    
    steps = int(duration / dt)
    log_every = 20  # Log every 20 steps
    
    for step in range(steps):
        # Step physics
        r, v = step_func(r, v, dt, mu)
        t += dt
        
        # Compute derived quantities
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Specific energy
        energy = 0.5 * v_mag**2 - mu / r_mag
        
        # Specific angular momentum (2D: h = r × v, only z component)
        h = r[0] * v[1] - r[1] * v[0]
        
        # Eccentricity vector: e = (v × h) / mu - r / |r|
        # In 2D: v × h = (v_x, v_y, 0) × (0, 0, h) = (v_y * h, -v_x * h, 0)
        e_vec = np.array([v[1] * h / mu - r[0] / r_mag, -v[0] * h / mu - r[1] / r_mag])
        e = np.linalg.norm(e_vec)
        
        # Log timeseries (every 20 steps)
        if step % log_every == 0:
            timeseries.append({
                "t": t,
                "x": r[0],
                "y": r[1],
                "vx": v[0],
                "vy": v[1],
                "r": r_mag,
                "v": v_mag,
                "energy": energy,
                "h": h,
                "e": e,
                "dt_eff": dt,
            })
        
        # Detect apsis events
        dr = r_mag - prev_r_mag
        if dr_sign_prev < 0 and dr >= 0:
            # Pericenter
            events.append({
                "t": t,
                "type": "pericenter",
                "r": r_mag,
                "v": v_mag,
                "details": json.dumps({"e": e, "energy": energy})
            })
        elif dr_sign_prev > 0 and dr <= 0:
            # Apocenter
            events.append({
                "t": t,
                "type": "apocenter",
                "r": r_mag,
                "v": v_mag,
                "details": json.dumps({"e": e, "energy": energy})
            })
        
        dr_sign_prev = 1 if dr > 0 else (-1 if dr < 0 else 0)
        prev_r_mag = r_mag
        
        # Check for impact
        if r_mag <= planet_radius:
            events.append({
                "t": t,
                "type": "impact",
                "r": r_mag,
                "v": v_mag,
                "details": json.dumps({"penetration": planet_radius - r_mag, "energy": energy})
            })
            break
        
        # Check for escape
        if energy > 0 and r_mag > 8 * np.linalg.norm(r0):
            events.append({
                "t": t,
                "type": "escape",
                "r": r_mag,
                "v": v_mag,
                "details": json.dumps({"e": e, "energy": energy})
            })
            break
    
    # Write timeseries
    with open(run_dir / "timeseries.csv", "w", newline="") as f:
        if timeseries:
            writer = csv.DictWriter(f, fieldnames=timeseries[0].keys())
            writer.writeheader()
            writer.writerows(timeseries)
    
    # Write events
    with open(run_dir / "events.csv", "w", newline="") as f:
        if events:
            writer = csv.DictWriter(f, fieldnames=["t", "type", "r", "v", "details"])
            writer.writeheader()
            writer.writerows(events)
        else:
            # Write header even if no events
            writer = csv.DictWriter(f, fieldnames=["t", "type", "r", "v", "details"])
            writer.writeheader()
    
    # Update last_run.txt
    last_run_file = Path("data") / "runs" / "last_run.txt"
    last_run_file.write_text(run_dir.name, encoding="utf-8")
    
    print(f"Simulation saved to {run_dir}")
    return run_dir


def main():
    """Generate key scenarios for the report."""
    print("Generating simulation data for report figures...")
    
    # Earth parameters
    planet_name = "Earth"
    planet_radius = EARTH_RADIUS
    planet_mass = M_EARTH
    mu = MU_EARTH
    
    # Orbital altitude (600 km above surface)
    h = 600_000  # m
    r0_mag = planet_radius + h
    r0 = np.array([r0_mag, 0.0])
    
    # Circular velocity at this altitude
    v_circ = math.sqrt(mu / r0_mag)
    
    # Escape velocity at this altitude
    v_esc = math.sqrt(2 * mu / r0_mag)
    
    print(f"\nOrbital altitude: {h/1000:.0f} km (r = {r0_mag/1e6:.3f} Mm)")
    print(f"Circular velocity: {v_circ:.2f} m/s")
    print(f"Escape velocity: {v_esc:.2f} m/s")
    
    scenarios = [
        # 1. Elliptical orbit with RK4
        {
            "name": "Elliptical_RK4",
            "v0": np.array([0.0, v_circ * 1.15]),  # 15% above circular
            "dt": 0.25,
            "duration": 10000,  # seconds
            "integrator": "RK4",
        },
        # 2. Same elliptical orbit with Euler (to show drift)
        {
            "name": "Elliptical_Euler",
            "v0": np.array([0.0, v_circ * 1.15]),
            "dt": 0.25,
            "duration": 10000,
            "integrator": "Euler",
        },
        # 3. Near-parabolic orbit (just below escape)
        {
            "name": "Near_Parabolic_RK4",
            "v0": np.array([0.0, v_esc * 0.99]),
            "dt": 0.25,
            "duration": 15000,
            "integrator": "RK4",
        },
        # 4. Hyperbolic escape
        {
            "name": "Hyperbolic_Escape_RK4",
            "v0": np.array([0.0, v_esc * 1.1]),
            "dt": 0.25,
            "duration": 8000,
            "integrator": "RK4",
        },
    ]
    
    run_dirs = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n=== Scenario {i}/{len(scenarios)}: {scenario['name']} ===")
        v0_mag = np.linalg.norm(scenario['v0'])
        print(f"Initial velocity: {v0_mag:.2f} m/s ({v0_mag/v_circ:.3f} × v_circ)")
        
        run_dir = run_simulation(
            r0=r0,
            v0=scenario['v0'],
            dt=scenario['dt'],
            duration=scenario['duration'],
            integrator=scenario['integrator'],
            planet_name=planet_name,
            planet_radius=planet_radius,
            planet_mass=planet_mass,
            scenario_name=scenario['name'],
        )
        run_dirs.append(run_dir)
    
    print(f"\n{'='*60}")
    print("All simulations complete!")
    print(f"Generated {len(run_dirs)} runs in data/runs/")
    print("\nNext steps:")
    print("1. Run analyze_run.py on each directory to generate figures")
    print("2. Copy figures to report_assets/figures/")
    
    return run_dirs


if __name__ == "__main__":
    main()
