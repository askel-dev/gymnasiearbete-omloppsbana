"""Data models for the orbit simulation state."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Body:
    """Celestial body used as the primary reference frame."""

    name: str
    radius: float
    mu: float


@dataclass
class Satellite:
    """Mutable state for the simulated satellite."""

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )

    def copy(self) -> "Satellite":
        return Satellite(position=self.position.copy(), velocity=self.velocity.copy())


@dataclass
class SimState:
    """High level simulation state container."""

    body: Body
    satellite: Satellite
    time: float = 0.0
    paused: bool = False

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.satellite.position = position.copy()
        self.satellite.velocity = velocity.copy()
        self.time = 0.0
        self.paused = False


__all__ = ["Body", "Satellite", "SimState"]
