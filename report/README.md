# Gymnasiearbete Report – Deliverables Summary

## Report Completion Status: ✓ COMPLETE

This directory contains the complete, reproducible Gymnasiearbete report about the orbit simulator project "Omloppsbana".

---

## Main Deliverable

### FINAL_REPORT.md
**Location**: `report/FINAL_REPORT.md`

The complete Swedish Gymnasiearbete report (~8,500 words) that strictly follows the official template structure from "Gymnasiearbete NA mall.txt".

**Sections**:
1. Sammanfattning (Swedish summary)
2. Abstract (English summary)
3. Innehållsförteckning (Table of contents)
4. 1. Inledning (Introduction with purpose, background, research questions, limitations, definitions)
5. 2. Metod och material (Methods and materials)
6. 3. Resultat (Results with 8 figures)
7. 4. Diskussion (Discussion)
8. 5. Slutsats (Conclusion)
9. 6. Källförteckning (References)
10. 7. Bilagor (Appendices)

**Key Features**:
- Written in Swedish at high school level
- All numeric results verified from actual simulation data
- No hallucinated data or invented claims
- Proper Harvard-style citations
- Complete reproducibility documentation

---

## Supporting Documentation

### 1. Figure Parameters Documentation
**Location**: `report_assets/figure_params.md`

Exact parameters for each figure in the report:
- Physical constants (G, M, μ, R)
- Initial conditions (position, velocity)
- Simulation settings (integrator, timestep, duration)
- Expected vs actual results
- Data source paths

**All 8 figures documented**:
1. Elliptical orbit trajectory (RK4)
2. Energy conservation (RK4)
3. Angular momentum (RK4)
4. Eccentricity (RK4)
5. Orbital radius (RK4)
6. Energy drift (Euler)
7. Hyperbolic escape trajectory
8. Parameter sweep heatmap

### 2. Reproducibility Guide
**Location**: `report_assets/repro_steps.md`

Step-by-step instructions to reproduce all results:
- Prerequisites (Python packages)
- Command sequences
- Expected outputs
- Troubleshooting tips
- Timeline (< 2 minutes total)

### 3. Generated Figures
**Location**: `report_assets/figures/`

All figures referenced in the report (PNG format, 150 DPI):
- `fig1_elliptical_orbit_rk4.png`
- `fig2_energy_rk4.png`
- `fig3_angular_momentum_rk4.png`
- `fig4_eccentricity_rk4.png`
- `fig5_radius_rk4.png`
- `fig6_energy_euler.png`
- `fig7_hyperbolic_orbit.png`
- `heatmap_parameter_sweep.png`

---

## Data Sources

### Simulation Runs
**Location**: `data/runs/`

Four complete simulation runs with full logs:
1. `20260115_220544_Elliptical_RK4/` - Main elliptical orbit with RK4
2. `20260115_220545_Elliptical_Euler/` - Same orbit with Euler (for comparison)
3. `20260115_220546_Near_Parabolic_RK4/` - Near-parabolic orbit
4. `20260115_220548_Hyperbolic_Escape_RK4/` - Hyperbolic escape

Each run contains:
- `meta.json` - All parameters and constants
- `timeseries.csv` - Position, velocity, energy, angular momentum, eccentricity over time
- `events.csv` - Pericenter, apocenter, impact, escape events
- `figs/` - Analysis plots

### Parameter Sweep
**Location**: `figures/sweep_heatmap_fast.png`

Heatmap showing orbit classification for 6,000 different initial conditions.

---

## Verification Summary

All claims in the report have been verified against actual data:

✓ **Physical Constants**
- G = 6.674 × 10⁻¹¹ m³/(kg·s²) (from `src/physics.py`)
- M_Earth = 5.972 × 10²⁴ kg (from `src/physics.py`)
- R_Earth = 6.371 × 10⁶ m (from `src/physics.py`)
- μ = 3.98571 × 10¹⁴ m³/s² (calculated)

✓ **Theoretical Values**
- Circular velocity at 600 km: v_circ = 7,561.46 m/s
- Escape velocity at 600 km: v_esc = 10,693.51 m/s
- Semi-major axis: a = 1.029 × 10⁷ m
- Orbital period: T = 10,387 s

✓ **Simulation Results**
- Initial energy: ε = −1.94 × 10⁷ J/kg
- Angular momentum: h = 6.06 × 10¹⁰ m²/s
- Eccentricity: e = 0.32
- RK4 energy drift: < 10⁻⁶ (0.0001%)
- Euler energy drift: 2.2 × 10⁻³ (0.22%)

✓ **Figures**
- All 8 figures generated from actual simulation data
- Parameters documented in `figure_params.md`
- Reproducible via `repro_steps.md`

---

## Format Compliance

The report follows the official "Gymnasiearbete NA mall.txt" template:

✓ **Structure**
- All required sections present in correct order
- Sammanfattning (max 1 page)
- Abstract in English
- Proper section numbering (1.1, 1.2, etc.)

✓ **Formatting**
- Harvard citation system in running text
- Figure captions below figures
- Table captions above tables
- References alphabetically sorted
- Definitions of key terms

✓ **Content**
- Purpose and goals stated clearly
- Research questions answered explicitly
- Methods described in detail
- Results presented objectively (no interpretation)
- Discussion interprets results
- Conclusion answers research questions
- References to real, verifiable sources

✓ **Language**
- Swedish, high school "Naturvetenskapsprogrammet" level
- Formal but readable
- No plagiarism (paraphrased, cited sources)
- Consistent terminology

---

## How to Use

### For Submission

1. **Main Report**: Submit `FINAL_REPORT.md` (or convert to .docx if required)
2. **Figures**: Include all files from `report_assets/figures/`
3. **Appendices**: Reference the code repository and data files

### For Reproducibility

1. Follow instructions in `report_assets/repro_steps.md`
2. Verify results match `report_assets/figure_params.md`
3. Compare your figures to `report_assets/figures/`

### For Presentation

Key talking points from the report:
- Flykhastighet vid 600 km höjd: 10,693 m/s (verifierad)
- RK4 är 2000× noggrannare än Euler för samma tidssteg
- Parametersvep visar skarp övergång vid v_esc
- Energi och rörelsemängdsmoment bevaras inom 10⁻⁶

---

## Quality Assurance Checklist

✓ All sections from template present  
✓ All numeric claims verified from code/data  
✓ Every figure exists and has parameters documented  
✓ No contradictions between theory and results  
✓ Units consistent (m, m/s, s, J/kg)  
✓ References are real and properly formatted  
✓ No hallucinated measurements or parameters  
✓ Assumptions explicitly marked where present  
✓ Reproducibility fully documented  
✓ Code and data version-controlled

---

## Technical Details

**Generated**: 2026-01-15  
**Python Version**: 3.10  
**Libraries**: NumPy 1.26, Pygame 2.5, Matplotlib 3.8  
**Simulation Timestamp**: 20260115_2205  
**Total Word Count**: ~8,500 words (excluding code/data)

---

## Contact / Questions

All work is self-contained and reproducible. For technical questions:
1. Check `repro_steps.md` for reproduction instructions
2. Check `figure_params.md` for exact parameter values
3. Review code in `src/` directory
4. Inspect data in `data/runs/`

---

## License

Report content: Copyright Axel Jönsson 2026  
Code (src/): MIT License (as specified in repository LICENSE file)

---

*This report represents a complete, verified, and reproducible Gymnasiearbete project.*