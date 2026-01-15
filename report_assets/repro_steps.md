# Reproducibility Guide

This document provides exact commands to reproduce all figures and results in the Gymnasiearbete report.

## Prerequisites

Ensure all dependencies are installed:

```bash
pip install pygame numpy matplotlib
```

## Step 1: Generate Parameter Sweep Heatmap

Run the parameter sweep analysis:

```bash
python src/sweep.py
```

**Output**: `figures/sweep_heatmap_fast.png`

**Duration**: ~10 seconds

**Result**: Heatmap showing orbit classification (elliptical/parabolic/hyperbolic) for 6,000 combinations of initial velocity and angle.

---

## Step 2: Generate Simulation Data

Run the automated simulation generator:

```bash
python src/generate_report_data.py
```

**Output**: Creates 4 run directories in `data/runs/`:
- `<timestamp>_Elliptical_RK4`
- `<timestamp>_Elliptical_Euler`
- `<timestamp>_Near_Parabolic_RK4`
- `<timestamp>_Hyperbolic_Escape_RK4`

**Duration**: ~10-15 seconds total

Each directory contains:
- `meta.json` - simulation parameters
- `timeseries.csv` - position, velocity, energy, etc. over time
- `events.csv` - pericenter, apocenter, impact, or escape events

---

## Step 3: Analyze Simulations and Generate Figures

For each generated run, analyze and create plots:

```bash
# Replace <timestamp> with actual timestamp from Step 2

# Elliptical orbit with RK4
python src/analyze_run.py data/runs/<timestamp>_Elliptical_RK4

# Same orbit with Euler (for comparison)
python src/analyze_run.py data/runs/<timestamp>_Elliptical_Euler

# Near-parabolic orbit
python src/analyze_run.py data/runs/<timestamp>_Near_Parabolic_RK4

# Hyperbolic escape
python src/analyze_run.py data/runs/<timestamp>_Hyperbolic_Escape_RK4
```

**Output**: For each run, creates `figs/` subdirectory with:
- `orbit_xy.png` - trajectory in xy-plane
- `energy.png` - specific energy over time
- `specific_angular_momentum.png` - |h| over time
- `eccentricity.png` - eccentricity over time
- `radius.png` - orbital radius with apsis markers

**Duration**: ~5-10 seconds per run

---

## Step 4: Copy Figures to Report Assets

```bash
# Create report directories if they don't exist
mkdir report
mkdir report_assets
mkdir report_assets\figures

# Copy heatmap
copy figures\sweep_heatmap_fast.png report_assets\figures\heatmap_parameter_sweep.png

# Copy RK4 elliptical orbit figures
copy data\runs\<timestamp>_Elliptical_RK4\figs\orbit_xy.png report_assets\figures\fig1_elliptical_orbit_rk4.png
copy data\runs\<timestamp>_Elliptical_RK4\figs\energy.png report_assets\figures\fig2_energy_rk4.png
copy data\runs\<timestamp>_Elliptical_RK4\figs\specific_angular_momentum.png report_assets\figures\fig3_angular_momentum_rk4.png
copy data\runs\<timestamp>_Elliptical_RK4\figs\eccentricity.png report_assets\figures\fig4_eccentricity_rk4.png
copy data\runs\<timestamp>_Elliptical_RK4\figs\radius.png report_assets\figures\fig5_radius_rk4.png

# Copy Euler energy drift comparison
copy data\runs\<timestamp>_Elliptical_Euler\figs\energy.png report_assets\figures\fig6_energy_euler.png

# Copy hyperbolic escape trajectory
copy data\runs\<timestamp>_Hyperbolic_Escape_RK4\figs\orbit_xy.png report_assets\figures\fig7_hyperbolic_orbit.png
```

---

## Alternative: Use Existing Data

If you want to reproduce the exact figures from the report generation timestamp `20260115_220544`:

```bash
# Heatmap (already exists)
copy figures\sweep_heatmap_fast.png report_assets\figures\heatmap_parameter_sweep.png

# RK4 figures
copy data\runs\20260115_220544_Elliptical_RK4\figs\*.png report_assets\figures\
copy data\runs\20260115_220545_Elliptical_Euler\figs\energy.png report_assets\figures\fig6_energy_euler.png
copy data\runs\20260115_220548_Hyperbolic_Escape_RK4\figs\orbit_xy.png report_assets\figures\fig7_hyperbolic_orbit.png
```

---

## Verification

To verify that your results match the documented values:

### Check Key Physical Constants

Open `src/physics.py` and verify:
```python
G = 6.674e-11          # Gravitational constant
M_EARTH = 5.972e24     # Earth mass (kg)
EARTH_RADIUS = 6_371_000  # Earth radius (m)
```

### Check Calculated Values

The script `src/generate_report_data.py` prints:
- Circular velocity: should be ~7,561.46 m/s
- Escape velocity: should be ~10,693.51 m/s

### Check Energy Conservation

RK4 simulations should show energy drift < 10⁻⁶ relative to initial energy.

View in analysis output:
```
Relativ energidrift dE/E = <value>
```

For RK4, this should be near zero (< 1e-6).
For Euler, you'll see noticeable drift (> 1e-4).

---

## Simulation Parameters Summary

### Scenario 1: Elliptical RK4
- **r₀**: 6,971,000 m
- **v₀**: 8,695.68 m/s (1.15 × v_circ)
- **Integrator**: RK4
- **dt**: 0.25 s
- **Duration**: 10,000 s

### Scenario 2: Elliptical Euler
- Same as Scenario 1 but with Euler integrator

### Scenario 3: Near-Parabolic
- **v₀**: 10,586.58 m/s (0.99 × v_esc)
- **Duration**: 15,000 s

### Scenario 4: Hyperbolic Escape
- **v₀**: 11,762.87 m/s (1.10 × v_esc)
- **Duration**: 8,000 s

---

## Troubleshooting

### Unicode Errors on Windows

If you see `UnicodeEncodeError` in console output, this is a Windows terminal encoding issue. The figures are still generated correctly - just the final print statement fails. You can ignore these errors or fix them by:

1. Opening `src/sweep.py` and `src/analyze_run.py`
2. Replacing special Unicode characters (✓, Δ) with ASCII equivalents

### Missing Dependencies

If import errors occur:
```bash
pip install --upgrade pygame numpy matplotlib
```

### Path Issues

On Windows, use backslashes `\` in paths.
On Linux/Mac, use forward slashes `/`.

The Python scripts handle both automatically.

---

## Timeline

Total time to reproduce all results: **< 2 minutes**

- Parameter sweep: ~10 s
- Generate 4 simulations: ~15 s
- Analyze 4 runs: ~40 s (10s each)
- Copy files: instant

---

## Contact / Issues

If reproduction fails, verify:
1. Python version (3.10+)
2. All dependencies installed
3. Working directory is the repository root
4. `src/` directory contains all .py files

All code is version-controlled in this repository for full reproducibility.
