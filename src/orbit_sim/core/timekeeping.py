"""Utilities for keeping fixed physics timesteps."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class FrameTimer:
    """High resolution timer based on :func:`time.perf_counter`."""

    last_time: float = field(default_factory=time.perf_counter)

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        return dt


@dataclass
class FixedStepAccumulator:
    """Accumulates simulation time and yields fixed step sizes."""

    step: float
    max_substeps: int
    value: float = 0.0

    def accrue(self, delta: float) -> None:
        if delta > 0.0:
            self.value += delta

    def clear(self) -> None:
        self.value = 0.0

    def consume(self) -> tuple[int, float]:
        if self.value <= 0.0:
            return 0, 0.0
        steps_needed = max(1, math.ceil(self.value / self.step))
        steps_to_run = min(steps_needed, self.max_substeps)
        dt = self.value / steps_to_run
        self.value = 0.0
        return steps_to_run, dt


__all__ = ["FixedStepAccumulator", "FrameTimer"]
