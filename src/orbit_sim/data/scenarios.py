"""Scenario definitions for preset simulation starting conditions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    velocity: tuple[float, float]
    description: str

    def velocity_vector(self) -> np.ndarray:
        return np.array(self.velocity, dtype=float)


SCENARIO_DEFINITIONS: tuple[Scenario, ...] = (
    Scenario(
        key="leo",
        name="LEO",
        velocity=(0.0, 7_600.0),
        description="Classic circular low Earth orbit (~7.6 km/s prograde).",
    ),
    Scenario(
        key="suborbital",
        name="Suborbital",
        velocity=(0.0, 6_000.0),
        description="Too slow for orbit – dramatic re-entry arc (~6.0 km/s).",
    ),
    Scenario(
        key="escape",
        name="Escape",
        velocity=(0.0, 11_500.0),
        description="High-energy burn that easily exceeds escape velocity (~11.5 km/s).",
    ),
    Scenario(
        key="parabolic",
        name="Parabolic",
        velocity=(0.0, 10_000.0),
        description="Close to the escape threshold (~10.0 km/s).",
    ),
    Scenario(
        key="retrograde",
        name="Retrograde",
        velocity=(0.0, -7_600.0),
        description="Same magnitude as LEO but flipped for retrograde flight.",
    ),
)

SCENARIOS: dict[str, Scenario] = {scenario.key: scenario for scenario in SCENARIO_DEFINITIONS}
SCENARIO_DISPLAY_ORDER: list[str] = [scenario.key for scenario in SCENARIO_DEFINITIONS]
DEFAULT_SCENARIO_KEY = SCENARIO_DISPLAY_ORDER[0]
SCENARIO_FLASH_DURATION = 2.0


__all__ = [
    "DEFAULT_SCENARIO_KEY",
    "SCENARIO_DEFINITIONS",
    "SCENARIO_DISPLAY_ORDER",
    "SCENARIOS",
    "SCENARIO_FLASH_DURATION",
    "Scenario",
]
