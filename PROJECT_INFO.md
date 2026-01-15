# OrbitLab – Project Information

## Quick Facts

| Property | Value |
|----------|-------|
| **Project Name** | OrbitLab |
| **Full Title** | OrbitLab: Hur påverkas en omloppsbana av startförhållanden? |
| **Version** | 1.5 |
| **Type** | Gymnasiearbete (Swedish High School Thesis) |
| **Subject** | Physics, Computer Science, Numerical Methods |
| **Language** | Swedish (report), Python (code) |
| **Author** | Axel Jönsson |
| **Year** | 2026 |
| **License** | MIT (code), Copyright (report) |

## Project Summary

OrbitLab is an interactive orbit simulator that explores how initial conditions (velocity and direction) determine a satellite's trajectory around celestial bodies. The project compares numerical integration methods (Euler vs RK4) and validates results against theoretical predictions.

## Key Features

✅ **Multi-Planet Support**: 10 celestial bodies (Earth, Mars, Jupiter, Neptune, etc.)  
✅ **Dual Integration**: Euler (1st order) and RK4 (4th order) methods  
✅ **Real-Time Visualization**: Pygame-based interactive display  
✅ **Automatic Logging**: CSV export of position, velocity, energy, momentum  
✅ **Analysis Tools**: Energy drift, orbit classification, parameter sweeps  
✅ **Procedural Textures**: Unique planet appearances generated from seed  
✅ **Reproducible Science**: All results verified from actual simulation data

## Technical Specifications

### Implementation
- **Language**: Python 3.10+
- **Physics Engine**: Custom 2D N-body simulator
- **Graphics**: Pygame 2.5
- **Analysis**: NumPy 1.26, Matplotlib 3.8
- **Integration Methods**: 
  - Euler (1st order, O(dt²) error)
  - Runge-Kutta 4 (4th order, O(dt⁵) error)

### Performance
- **Timestep**: 0.25 seconds (configurable)
- **Typical Run**: 10,000 seconds simulated in ~1 second real-time
- **Parameter Sweep**: 6,000 conditions evaluated in ~10 seconds
- **Energy Conservation**: < 10⁻⁶ relative drift (RK4)

### Physics Model
- **Gravitational Law**: Newton's law of universal gravitation
- **Coordinate System**: 2D Cartesian (x, y plane)
- **Assumptions**: 
  - Two-body problem (planet + satellite)
  - Point masses
  - No atmospheric drag
  - No relativistic effects

## Research Questions

The project investigates:

1. **How does initial velocity affect orbit shape?**
   - Result: Energy sign determines elliptical (ε<0) vs hyperbolic (ε>0)

2. **What is the escape velocity threshold?**
   - Result: v_esc = 10,693.51 m/s at 600 km altitude (verified)

3. **How do Euler and RK4 compare?**
   - Result: RK4 is ~2000× more accurate for same timestep

4. **Are conserved quantities preserved?**
   - Result: Energy and angular momentum conserved to 10⁻⁶

## Project Structure

```
OrbitLab/
├── src/                          # Source code
│   ├── main.py                   # Pygame simulator
│   ├── physics.py                # Integrators (Euler, RK4)
│   ├── planet_generator.py       # Procedural planet textures
│   ├── logging_utils.py          # Data recording
│   ├── analyze_run.py            # Post-simulation analysis
│   └── sweep.py                  # Parameter space exploration
├── data/runs/                    # Logged simulation data
├── report/                       # Gymnasiearbete report
│   ├── FINAL_REPORT.md          # Full thesis (~8,500 words)
│   └── README.md                # Report documentation
├── report_assets/
│   ├── figures/                 # All 8 figures (PNG, 150 DPI)
│   ├── figure_params.md         # Exact parameters per figure
│   └── repro_steps.md           # Reproduction instructions
├── README.md                    # This overview
├── BRANDING.md                  # Visual identity guidelines
└── CONVERSION_GUIDE.md          # Markdown → DOCX instructions
```

## Key Results

### Verified Constants
- G = 6.674 × 10⁻¹¹ m³/(kg·s²)
- M_Earth = 5.972 × 10²⁴ kg
- R_Earth = 6.371 × 10⁶ m
- μ = G·M = 3.986 × 10¹⁴ m³/s²

### Orbital Parameters (600 km altitude)
- Circular velocity: 7,561.46 m/s
- Escape velocity: 10,693.51 m/s
- Orbital radius: 6,971,000 m

### Numerical Performance
- RK4 energy drift: < 10⁻⁶ (0.0001%)
- Euler energy drift: ~2 × 10⁻³ (0.22%)
- Angular momentum conservation: < 0.001%

### Orbit Classification
- Elliptical: ε < -300,000 J/kg (bound)
- Parabolic: -300,000 ≤ ε ≤ 300,000 J/kg (marginal)
- Hyperbolic: ε > 300,000 J/kg (unbound)

## Educational Value

OrbitLab demonstrates:

1. **Orbital Mechanics**: Kepler's laws, energy conservation, two-body problem
2. **Numerical Methods**: Euler vs RK4, timestep sensitivity, error accumulation
3. **Scientific Programming**: Data logging, visualization, reproducibility
4. **Physics Validation**: Theoretical predictions vs simulation results
5. **Engineering Trade-offs**: Accuracy vs computation time

## Use Cases

### For Students
- Learn orbital mechanics interactively
- Experiment with different initial conditions
- Visualize abstract physics concepts
- Understand numerical integration methods

### For Teachers
- Demonstrate Kepler's laws dynamically
- Compare numerical methods side-by-side
- Show energy conservation principles
- Illustrate sensitivity to initial conditions

### For Developers
- Reference implementation of RK4
- Example of physics simulation in Pygame
- Data logging and analysis pipeline
- Reproducible scientific workflow

## Citations & References

If you use OrbitLab in your work:

**APA Style:**
```
Jönsson, A. (2026). OrbitLab: Hur påverkas en omloppsbana av startförhållanden? 
[Computer software and thesis]. https://github.com/[username]/OrbitLab
```

**Bibtex:**
```bibtex
@software{jonsson2026orbitlab,
  author = {Jönsson, Axel},
  title = {OrbitLab: Interactive Orbit Simulator},
  year = {2026},
  publisher = {GitHub},
  type = {Gymnasiearbete},
  url = {https://github.com/[username]/OrbitLab}
}
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install pygame numpy matplotlib
   ```

2. **Run the simulator:**
   ```bash
   python src/main.py
   ```

3. **Try different scenarios:**
   - Press `P` to cycle through planets
   - Press `1-5` to select predefined scenarios
   - Click "Integrator" to switch between RK4/Euler

4. **Analyze results:**
   ```bash
   python src/analyze_run.py data/runs/[latest_run]
   ```

5. **Generate parameter sweep:**
   ```bash
   python src/sweep.py
   ```

## Future Enhancements

Potential improvements for OrbitLab v2.0:

- [ ] 3D visualization with perspective camera
- [ ] Symplectic integrators (Verlet, leapfrog)
- [ ] Three-body problem (Earth-Moon-satellite)
- [ ] Atmospheric drag model
- [ ] J₂ perturbations (Earth oblateness)
- [ ] Hohmann transfer orbit calculator
- [ ] Mission planning tools
- [ ] WebAssembly port for browser

## Contact & Support

**Author**: Axel Jönsson  
**Project**: Gymnasiearbete, Naturvetenskapsprogrammet  
**Year**: 2026

For questions about:
- **Physics**: See `report/FINAL_REPORT.md` sections 1-2
- **Code**: Check docstrings in `src/*.py` files
- **Reproducibility**: Follow `report_assets/repro_steps.md`
- **Figures**: See `report_assets/figure_params.md`

## Acknowledgments

- **Python Software Foundation** – Python language and ecosystem
- **Pygame Community** – Real-time graphics library
- **NumPy/Matplotlib Teams** – Scientific computing tools
- **NASA** – Planetary data and orbital mechanics resources
- **Isaac Newton** – Laws of motion and gravitation
- **Johannes Kepler** – Orbital mechanics foundations
- **Carl David Tolmé Runge & Wilhelm Kutta** – RK4 integration method

## License

**Code** (`src/`): MIT License – free to use, modify, distribute  
**Report** (`report/`): Copyright © 2026 Axel Jönsson  
**Data & Figures**: CC BY 4.0 – attribution required

---

**Built with ❤️ for understanding the universe through code**

*OrbitLab – Because every great orbit starts with the right initial conditions.*
