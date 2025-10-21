# Run Logging Overview

This document summarises what the simulator records during a run, how the
information is captured, and the orbital mechanics underpinning each field.

## Directory layout

Every run initialises a dedicated folder inside `data/runs/` using the pattern
`<timestamp>_run/`. The folder contains three primary artefacts:

- `meta.json` — high level configuration and constants for the session.
- `timeseries.csv` — downsampled orbital state vectors and derived scalars.
- `events.csv` — sparse log of interesting events such as pericentre passages or
  impacts.

The helper `RunLogger` class in `src/orbit_sim/core/logging_utils.py` buffers writes to avoid
introducing stutter in the Pygame loop. Time-series samples are flushed in
chunks of 200 rows and events in chunks of 50 rows, or immediately when the
simulation closes.

## Logging cadence

The simulator advances the dynamics using a fixed physics time step `dt_phys`
(0.25 s by default) but may run multiple substeps per frame. Logging is
throttled to one record every `LOG_EVERY_STEPS = 20` physics updates. This
limits `timeseries.csv` to roughly one sample per five simulated seconds, which
captures the orbital evolution without overwhelming the disk. Whenever an event
is detected, the current state is logged immediately so that the event is
aligned with the corresponding time-series row.

## Metadata (`meta.json`)

`meta.json` records the inputs that produced the run:

| Field | Meaning |
| --- | --- |
| `R0`, `V0` | Initial position (m) and velocity (m/s) vectors. |
| `v0` | Magnitude of the initial velocity. |
| `G`, `M`, `mu` | Gravitational constant, Earth mass, and the standard gravitational parameter (`mu = G·M`). |
| `integrator` | Name of the time integrator (`"RK4"`). |
| `dt_phys` | Base physics time step used per integration update. |
| `start_speed` | Real-time speed multiplier active when the run starts. |
| `log_strategy` | Sampling rule that decides when a time-series row is emitted. |
| `code_version` | Semantic tag for the logging implementation. |

The metadata is written once at the start of each run, ensuring that downstream
analysis scripts can reproduce the same parameters.

## Time series (`timeseries.csv`)

Each row stores both raw state and derived quantities:

| Column | Definition | Rationale |
| --- | --- | --- |
| `t` | Simulation time in seconds. | Provides a monotonic clock. |
| `x`, `y` | Cartesian position components in metres. | Enables plan-view plots and radius calculations. |
| `vx`, `vy` | Velocity components in metres per second. | Required for energy, angular momentum, and orbit fits. |
| `r` | Orbital radius `r = √(x² + y²)`. | Highlights pericentre/apocentre distances. |
| `v` | Speed `v = √(vx² + vy²)`. | Useful for kinetic energy and escape checks. |
| `energy` | Specific mechanical energy `ε = v²/2 − μ/r`. | Sign indicates ellipse (ε < 0), parabola (≈0), or hyperbola (ε > 0). |
| `h` | Specific angular momentum magnitude `h = ‖r × v‖`. | Constant for two-body motion; drift reveals numerical error. |
| `e` | Scalar eccentricity obtained from the eccentricity vector `e⃗ = (v × h⃗)/μ − r⃗/r`. | Classifies orbit shape and orientation. |
| `dt_eff` | Effective physics step used for the sample (seconds). | Records adaptive behaviour when substepping. |

### Derivations

- **Specific mechanical energy**: combining kinetic and gravitational potential
  terms in the restricted two-body problem yields `ε = v²/2 − μ/r`. This value
  stays constant for ideal orbits; departures indicate perturbations or
  numerical integration error.
- **Specific angular momentum**: with position vector `r⃗ = (x, y, 0)` and
  velocity `v⃗ = (vx, vy, 0)`, the two-body angular momentum vector is
  `h⃗ = r⃗ × v⃗`. Its magnitude is `‖h⃗‖ = √(h_x² + h_y² + h_z²)` and should be
  conserved.
- **Eccentricity**: the eccentricity vector is defined as
  `e⃗ = (v⃗ × h⃗)/μ − r⃗/r`. Taking its norm produces the scalar eccentricity `e`,
  where `e < 1` corresponds to ellipses, `e = 1` to parabolas, and `e > 1` to
  hyperbolas.

## Events (`events.csv`)

Event rows capture discrete changes in orbital regime. The logger monitors the
radius derivative `Δr = r_n − r_{n−1}` and flags the following cases:

| Type | Condition | Stored details |
| --- | --- | --- |
| `pericenter` | `Δr` changes sign from negative to non-negative (local minimum). | JSON blob with instantaneous eccentricity and energy. |
| `apocenter` | `Δr` changes sign from positive to non-positive (local maximum). | JSON blob with instantaneous eccentricity and energy. |
| `impact` | `r ≤ R_earth`. | Penetration depth (`R_earth − r`) and energy. |
| `escape` | `ε > 0` and `r > escape_radius_limit`. | Eccentricity and energy at escape. |

Each event records the time, type, orbital radius, speed, and contextual JSON to
support later analysis. Because events are rare, the logger writes them to a
separate buffer with a lower flush threshold.

## Why this information matters

Collecting state, energy, and angular momentum allows the analysis tooling to:

- Validate numerical stability by checking how well conserved quantities behave.
- Estimate orbital elements (period, eccentricity, pericentre distance) without
  rerunning the simulation.
- Detect regime changes (capture, impact, escape) and measure when they occur.
- Produce report-ready plots and heatmaps that explain how initial conditions
  map to orbital classes.

The combination of buffered logging and derived quantities supports real-time
visualisation while preserving high-quality data for post-processing.
