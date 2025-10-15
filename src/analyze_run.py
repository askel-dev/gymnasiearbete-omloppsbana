"""Analyze a recorded simulation run and generate diagnostic figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMESERIES_FILENAME = "timeseries.csv"
EVENTS_FILENAME = "events.csv"
META_FILENAME = "meta.json"
FIGS_SUBDIR = "figs"
ENERGY_TOL = 1e-4  # joule per kilogram tolerance for orbit classification
EARTH_RADIUS = 6_371_000


def load_timeseries(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                columns.setdefault(key, []).append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def load_events(path: Path) -> List[dict]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        events: List[dict] = []
        for row in reader:
            if not row:
                continue
            event = {
                "t": float(row["t"]),
                "type": row["type"],
                "r": float(row["r"]),
                "v": float(row["v"]),
            }
            details_raw = row.get("details", "")
            if details_raw:
                try:
                    event["details"] = json.loads(details_raw)
                except json.JSONDecodeError:
                    event["details"] = details_raw
            events.append(event)
    return events


def ensure_fig_dir(run_dir: Path) -> Path:
    fig_dir = run_dir / FIGS_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def classify_orbit(energy_last: float) -> str:
    if energy_last < -ENERGY_TOL:
        return "elliptisk"
    if energy_last > ENERGY_TOL:
        return "hyperbolisk"
    return "parabolisk"


def summarize_events(events: List[dict]) -> Dict[str, int]:
    summary: Dict[str, int] = {"pericenter": 0, "apocenter": 0, "impact": 0, "escape": 0}
    for event in events:
        if event["type"] in summary:
            summary[event["type"]] += 1
    return summary


def estimate_period(events: List[dict]) -> float | None:
    pericenters = [event["t"] for event in events if event["type"] == "pericenter"]
    if len(pericenters) >= 2:
        return pericenters[-1] - pericenters[-2]
    return None


def plot_orbit(fig_dir: Path, ts: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ts["x"], ts["y"], color="#6bc5c0", lw=1.5, label="Satellit")
    ax.scatter([0.0], [0.0], color="#4a86f7", s=60, label="Jorden")
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(EARTH_RADIUS * np.cos(theta), EARTH_RADIUS * np.sin(theta), color="#4a86f7", alpha=0.3)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Bana (x–y)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "orbit_xy.png", dpi=150)
    plt.close(fig)


def plot_energy(fig_dir: Path, ts: Dict[str, np.ndarray], rel_drift: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts["t"], ts["energy"], color="#ffa94d")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Specifik energi [J/kg]")
    ax.set_title(f"Specifik energi – relativ drift ΔE/E = {rel_drift:.2e}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "energy.png", dpi=150)
    plt.close(fig)


def plot_h(fig_dir: Path, ts: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts["t"], np.abs(ts["h"]), color="#94d82d")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|h| [m²/s]")
    ax.set_title("|h| över tid")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "specific_angular_momentum.png", dpi=150)
    plt.close(fig)


def plot_eccentricity(fig_dir: Path, ts: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts["t"], ts["e"], color="#9775fa")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e [-]")
    ax.set_title("Excentricitet över tid")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "eccentricity.png", dpi=150)
    plt.close(fig)


def plot_radius(fig_dir: Path, ts: Dict[str, np.ndarray], events: List[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts["t"], ts["r"], color="#4dabf7")
    for event in events:
        if event["type"] == "pericenter":
            ax.axvline(event["t"], color="#d9480f", linestyle="--", alpha=0.5, label="Pericenter")
        elif event["type"] == "apocenter":
            ax.axvline(event["t"], color="#1864ab", linestyle=":", alpha=0.5, label="Apocenter")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # deduplicate labels
        seen = {}
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen[label] = True
                unique_handles.append(handle)
                unique_labels.append(label)
        ax.legend(unique_handles, unique_labels)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("r [m]")
    ax.set_title("Radie över tid")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "radius.png", dpi=150)
    plt.close(fig)


def print_summary(
    run_dir: Path,
    meta: dict,
    orbit_class: str,
    a: float | None,
    T_theo: float | None,
    T_sim: float | None,
    rel_drift: float,
    event_summary: Dict[str, int],
) -> None:
    print(f"Run: {run_dir.name}")
    print(f" Klassificering: {orbit_class}")
    if a is not None:
        print(f" Semi-huvudaxel a = {a:.3e} m")
    else:
        print(" Semi-huvudaxel a: ej definierad (icke-sluten bana)")
    if T_theo is not None:
        print(f" Teoretisk period T_theo = {T_theo:.3f} s")
    else:
        print(" Teoretisk period T_theo: ej tillgänglig")
    if T_sim is not None:
        print(f" Simulerad period (senaste två pericenter) T_sim = {T_sim:.3f} s")
    else:
        print(" Simulerad period: kräver minst två pericenter")
    print(f" Relativ energidrift ΔE/E = {rel_drift:.3e}")
    print(
        " Event-sammanfattning:" +
        ", ".join(f" {etype}: {count}" for etype, count in event_summary.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analysera en loggad körning och skapa figurer.")
    parser.add_argument("run_dir", nargs="?", help="Sökväg till en specifik run-mapp")
    args = parser.parse_args()

    base_runs_dir = Path("data") / "runs"
    if args.run_dir:
        run_path = Path(args.run_dir)
        if not run_path.is_dir():
            run_path = base_runs_dir / args.run_dir
    else:
        last_run_file = base_runs_dir / "last_run.txt"
        if not last_run_file.exists():
            parser.error("Ingen run angiven och last_run.txt saknas.")
        run_id = last_run_file.read_text(encoding="utf-8").strip()
        run_path = base_runs_dir / run_id

    if not run_path.is_dir():
        parser.error(f"Kunde inte hitta körningsmapp: {run_path}")

    meta_path = run_path / META_FILENAME
    ts_path = run_path / TIMESERIES_FILENAME
    ev_path = run_path / EVENTS_FILENAME

    if not meta_path.exists() or not ts_path.exists() or not ev_path.exists():
        parser.error("Körningsmappen saknar nödvändiga filer (meta/timeseries/events).")

    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ts = load_timeseries(ts_path)
    events = load_events(ev_path)

    fig_dir = ensure_fig_dir(run_path)

    if not ts:
        parser.error("Timeseries.csv är tom – kan inte analysera.")

    energy = ts.get("energy", np.array([]))
    rel_drift = 0.0
    if energy.size:
        denom = energy[0] if abs(energy[0]) > 1e-12 else 1.0
        rel_drift = (energy[-1] - energy[0]) / denom

    orbit_class = classify_orbit(float(energy[-1])) if energy.size else "okänd"
    mu = float(meta.get("mu", 0.0))
    energy_last = float(energy[-1]) if energy.size else 0.0
    a = None
    T_theo = None
    if mu > 0 and energy_last < 0:
        a = -mu / (2.0 * energy_last)
        T_theo = 2.0 * math.pi * math.sqrt(a**3 / mu)

    T_sim = estimate_period(events)
    event_summary = summarize_events(events)

    plot_orbit(fig_dir, ts)
    plot_energy(fig_dir, ts, rel_drift)
    plot_h(fig_dir, ts)
    plot_eccentricity(fig_dir, ts)
    plot_radius(fig_dir, ts, events)

    print_summary(run_path, meta, orbit_class, a, T_theo, T_sim, rel_drift, event_summary)


if __name__ == "__main__":
    main()
