```
╔═══════════════════════════════════════╗
║                                       ║
║           O r b i t L a b             ║
║                                       ║
║   En interaktiv omloppsbana-simulator ║
║                                       ║
╚═══════════════════════════════════════╝
```

# OrbitLab – v1.5

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Code Size](https://img.shields.io/github/languages/code-size/username/OrbitLab?color=orange)

**En interaktiv omloppsbana-simulator**

Interaktiv RK4/Euler-simulator för omloppsbanor med stöd för flera planeter, procedurellt genererade texturer, inbyggd datalogging, analys och parameterstudier.

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulator
python src/main.py
```

**That's it!** The simulator should launch immediately.

## 📋 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 512 MB | 1 GB+ |
| **Storage** | 50 MB | 100 MB |
| **Display** | 1024x768 | 1920x1080 |
| **OS** | Windows 10, macOS 10.15, Linux | Latest versions |

## 🔧 Installation

### Option 1: Quick Install (Recommended)

```bash
# Clone and install in one go
git clone https://github.com/yourusername/OrbitLab.git
cd OrbitLab
pip install -r requirements.txt
python src/main.py
```

### Option 2: Manual Install

```bash
# Install dependencies individually
pip install pygame>=2.5.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0

# Run the simulator
python src/main.py
```

### Option 3: Using Virtual Environment (Cleanest)

```bash
# Create virtual environment
python -m venv orbitlab_env

# Activate it
# Windows:
orbitlab_env\Scripts\activate
# macOS/Linux:
source orbitlab_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```

## Kör simulatorn

Starta Pygame-appen:

```bash
python src/main.py
```

### Huvudmeny

Vid start visas en meny där du kan:
- Trycka `SPACE` för att starta simuleringen
- Trycka `P` för att byta planet
- Trycka `1-5` för att välja scenario (startar simuleringen direkt)
- Klicka på "Integrator"-knappen för att växla mellan RK4 och Euler

### Planeter

Simulatorn stöder 10 olika himlakroppar med realistiska fysikaliska egenskaper:

| Planet   | Radie (km) | Typ              |
|----------|------------|------------------|
| Earth    | 6 371      | Terrestrial      |
| Mars     | 3 390      | Barren           |
| Moon     | 1 737      | Barren           |
| Jupiter  | 69 911     | Gas Giant        |
| Neptune  | 24 622     | Ice Giant        |
| Venus    | 6 052      | Terrestrial      |
| Saturn   | 58 232     | Gas Giant        |
| Uranus   | 25 362     | Ice Giant        |
| Mercury  | 2 440      | Barren           |
| Io       | 1 822      | Molten           |

Varje planet renderas med procedurell textur baserad på sin typ (kontinenter, kratrar, gasband, lavaflöden m.m.).

### Scenarion

Varje planet har 5 fördefinierade scenarion:

1. **Low Orbit** – Cirkulär omloppsbana nära ytan
2. **Suborbital** – För låg hastighet, faller tillbaka
3. **Escape** – Överstiger flykthastigheten
4. **Parabolic** – Precis vid flyktgränsen
5. **Retrograde** – Cirkulär hastighet men bakåtriktad

Scenarierna anpassas automatiskt efter varje planets gravitationsparameter.

## Tangentbordskommandon

### Allmänt
| Tangent      | Funktion                          |
|--------------|-----------------------------------|
| `ESC`        | Avsluta programmet                |
| `F11`        | Växla fullskärm/fönsterläge       |
| `1-5`        | Byt scenario (nummer)             |
| `N`          | Nästa scenario                    |
| `P`          | Byt planet                        |

### Under simulering
| Tangent      | Funktion                          |
|--------------|-----------------------------------|
| `SPACE`      | Pausa/återuppta                   |
| `R`          | Återställ aktuellt scenario       |
| `+` / `-`    | Zooma in/ut                       |
| `←` / `→`    | Sänk/höj simuleringshastighet     |
| `↑` / `↓`    | Dubblera/halvera hastigheten      |
| `C`          | Växla kameraläge                  |
| `V`          | Visa/dölj hastighetspil           |
| `G`          | Visa/dölj koordinatrutnät         |

### Mus
| Åtgärd           | Funktion                       |
|------------------|--------------------------------|
| Scrollhjul       | Zooma in/ut (centrerad på mus) |
| Vänsterklick+dra | Panorera kameran (fritt läge)  |

## Kameralägen

- **Earth** (standard) – Centrerad på planeten med liten offset uppåt
- **Satellite** – Följer satelliten
- **Free** – Manuell panorering och zoom

## Integratorer

Simulatorn stöder två numeriska metoder:

- **RK4** (Runge-Kutta 4) – Hög noggrannhet, bevarar energi väl
- **Euler** – Enklare, snabbare, men ackumulerar fel över tid

Välj integrator via menyknappen innan start.

## Datalogging

Simuleringen loggar automatiskt data till `data/runs/<timestamp>_run/`:

```
data/runs/<timestamp>_run/
├── meta.json          # grundparametrar och konfiguration
├── timeseries.csv     # tidsserier: t, x, y, vx, vy, r, v, energy, h, e, dt_eff
└── events.csv         # händelser: t, type, r, v, details (JSON)
```

### Metadata (`meta.json`)

| Fält                  | Beskrivning                                    |
|-----------------------|------------------------------------------------|
| `scenario_key`        | Scenariots nyckel (t.ex. "leo")                |
| `scenario_name`       | Scenariots visningsnamn                        |
| `scenario_description`| Beskrivning av scenariot                       |
| `planet_name`         | Aktuell planet (t.ex. "Earth")                 |
| `planet_radius`       | Planetens radie i meter                        |
| `R0`, `V0`            | Startposition och starthastighet (vektorer)    |
| `v0`                  | Starthastighet (magnitud)                      |
| `G`, `M`, `mu`        | Gravitationskonstant, massa, μ = G·M           |
| `integrator`          | Vald integrator ("RK4" eller "Euler")          |
| `dt_phys`             | Fysik-tidssteg (standard 0.25 s)               |
| `start_speed`         | Simuleringshastighet vid start                 |
| `log_strategy`        | Loggningsstrategi (t.ex. "every_20_steps")     |
| `code_version`        | Version av koden                               |

### Händelsetyper

| Typ         | Villkor                                    |
|-------------|--------------------------------------------|
| `pericenter`| Lokal minimum i radie (närmast planeten)   |
| `apocenter` | Lokal maximum i radie (längst från planet) |
| `impact`    | Radie ≤ planetradie (kollision)            |
| `escape`    | Positiv energi och radie > flyktgräns      |

Den senast skapade körningen pekas ut av `data/runs/last_run.txt`.

## Analysera en körning

Generera figurer och sammanfattning:

```bash
python src/analyze_run.py data/runs/<timestamp>_run
```

Utelämna argumentet för att analysera senaste körningen:

```bash
python src/analyze_run.py
```

Skriptet läser loggfilerna och producerar:

### Terminal-rapport
- Klassificering (elliptisk/parabolisk/hyperbolisk)
- Semi-huvudaxel (för slutna banor)
- Teoretisk och simulerad period
- Relativ energidrift ΔE/E
- Sammanfattning av händelser

### Genererade figurer (`figs/`)

| Fil                            | Innehåll                                   |
|--------------------------------|--------------------------------------------|
| `orbit_xy.png`                 | Bana i xy-planet med planetmarkering       |
| `energy.png`                   | Specifik energi över tid med driftrapport  |
| `specific_angular_momentum.png`| Specifikt rörelsemängdsmoment |h|(t)        |
| `eccentricity.png`             | Excentricitet över tid                     |
| `radius.png`                   | Radie med markerade peri-/apocentra        |

## Parameter-svep och heatmap

Utför ett svep över starthastigheter och vinklar:

```bash
python src/sweep.py
```

Resultatet sparas som `figures/sweep_heatmap_<mode>.png` där färgerna representerar:
- **Grön** – Elliptisk bana
- **Gul** – Parabolisk (nära flyktgränsen)
- **Röd** – Hyperbolisk (flykt)

Svepet kan köras i två lägen:
- `fast` – Analytisk klassificering baserad på energi (standard)
- `simulated` – Numerisk RK4-integration för varje punkt

## Visuella funktioner

### Kollisionseffekter
Vid nedslag på planeten visas:
- Expanderande chockvågsring
- Informationsöverlägg med nedslagshastighet och vinkel
- Möjlighet att trycka `R` för reset eller `N` för nästa scenario

### Banprediktion
För slutna banor (ellipser) ritas en förutsagd bana som gradvis avtäcks under första omloppet.

### Apsis-markörer
Peri- och apocentra markeras med interaktiva markörer. Håll muspekaren över för att se höjd över ytan.

## Projektstruktur

```
├── src/
│   ├── main.py              # Huvudprogram med Pygame-loop
│   ├── physics.py           # RK4/Euler-integratorer, gravitationsberäkningar
│   ├── planet_generator.py  # Procedurell planetgenerering och presets
│   ├── logging_utils.py     # Buffrad datalogging
│   ├── analyze_run.py       # Analysverktyg för körningar
│   └── sweep.py             # Parametersvep för banklassificering
├── data/
│   └── runs/                # Loggade körningar
├── figures/                 # Genererade heatmaps från sweep
├── LOGGING_OVERVIEW.md      # Detaljerad dokumentation av loggformat
└── README.md                # Denna fil
```

## Versionshistorik

### v1.5 (aktuell)
- Stöd för 10 olika planeter med realistiska egenskaper
- Procedurell texturering baserad på planettyp
- Dynamiska scenarion per planet
- Valbar integrator (RK4/Euler)
- Tre kameralägen (Planet/Satellit/Fri)
- Interaktiva apsis-markörer med höjdvisning
- Kollisionsanimationer och informationsöverlägg
- Koordinatrutnät med skalbar upplösning
- Hastighetspil med skalning
- Musbaserad zoom centrerad på pekaren
- Utökad metadata i loggfiler

### v1.0
- Grundläggande RK4-simulator för Jorden
- Datalogging med buffrad skrivning
- Analysverktyg och parametersvep
- Fem fördefinierade scenarion
