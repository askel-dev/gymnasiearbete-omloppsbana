# Omloppsbana – v1.0

Interaktiv RK4-simulator för omloppsbanor med inbyggd datalogging, analys och parameterstudier.

## Förutsättningar

* Python 3.10+
* `pygame`, `numpy`, `matplotlib`

Installera beroenden via pip om de saknas:

```bash
pip install pygame numpy matplotlib
```

## Kör simulatorn och logga data

Starta Pygame-appen:

```bash
python src/orbit_pygame.py
```

* `Start Simulation` startar en körning. En ny loggmapp skapas automatiskt under `data/runs/<timestamp>_run/`.
* Knappar och tangentbord beter sig som tidigare (paus, reset, zoom, hastighet, kamera).
* Programmet loggar **var 20:e fysiksteg** (`log_strategy = "every_20_steps"`). Loggningen är buffrad och påverkar inte bilduppdateringen.
* Avsluta med `ESC` eller `Quit` – loggern flushas och stängs ned.

Varje körning innehåller följande filer:

```
data/runs/<timestamp>_run/
├── meta.json          # grundparametrar (R0, V0, mu, dt_phys, m.m.)
├── timeseries.csv     # kolumner: t, x, y, vx, vy, r, v, energy, h, e, dt_eff
└── events.csv         # kolumner: t, type, r, v, details (JSON-sträng)
```

Den senast skapade körningen pekas ut av `data/runs/last_run.txt`.

## Analysera en körning

Generera figurer och summering:

```bash
python src/analyze_run.py data/runs/<timestamp>_run
```

Utelämna argumentet för att analysera senaste körningen:

```bash
python src/analyze_run.py
```

Skriptet läser `meta.json`, `timeseries.csv` och `events.csv`, skriver en rapport till terminalen (klassificering, semi-huvudaxel, perioder, ΔE/E) och sparar figurer i `data/runs/<run_id>/figs/`:

* `orbit_xy.png` – bana i xy-planet
* `energy.png` – specifik energi över tid med relativ drift
* `specific_angular_momentum.png` – |h|(t)
* `eccentricity.png` – excentricitet över tid
* `radius.png` – radie med markerade peri-/apocentra

## Parameter-svep och heatmap

Utför ett svep över starthastigheter (6 500–12 500 m/s) och vinklar (0–90°) och spara en heatmap över bantyp:

```bash
python src/sweep.py
```

Resultatet sparas som `figures/sweep_heatmap.png` där färgerna representerar ellips/parabel/hyperbel.

## Version & loggstrategi

Den här versionen är **v1.0**. Logging sker med strategin `every_20_steps` (buffrad, flush vid ≥200 tidsrader eller ≥50 events).
