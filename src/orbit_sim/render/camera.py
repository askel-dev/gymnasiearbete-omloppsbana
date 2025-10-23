from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class CameraState:
    center: np.ndarray
    target: np.ndarray
    ppm: float
    ppm_target: float


class Camera:
    """Camera helper handling zoom and panning."""

    def __init__(
        self,
        size: tuple[int, int],
        ppm: float,
        *,
        min_ppm: float,
        max_ppm: float,
    ) -> None:
        self._size = size
        self._min_ppm = min_ppm
        self._max_ppm = max_ppm
        ppm = _clamp(ppm, min_ppm, max_ppm)
        self._state = CameraState(
            center=np.array([0.0, 0.0], dtype=float),
            target=np.array([0.0, 0.0], dtype=float),
            ppm=ppm,
            ppm_target=ppm,
        )
        self._pan_anchor: tuple[int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def update_size(self, size: tuple[int, int]) -> None:
        self._size = size

    @property
    def min_ppm(self) -> float:
        return self._min_ppm

    @property
    def max_ppm(self) -> float:
        return self._max_ppm

    @property
    def ppm(self) -> float:
        return self._state.ppm

    @property
    def ppm_target(self) -> float:
        return self._state.ppm_target

    @property
    def center(self) -> np.ndarray:
        return self._state.center

    @property
    def target(self) -> np.ndarray:
        return self._state.target

    def set_center(self, position: tuple[float, float]) -> None:
        self._state.center[:] = position
        self._state.target[:] = position

    def set_target(self, position: tuple[float, float]) -> None:
        self._state.target[:] = position

    def set_zoom(self, ppm: float) -> None:
        clamped = _clamp(ppm, self._min_ppm, self._max_ppm)
        self._state.ppm = clamped
        self._state.ppm_target = clamped

    def set_zoom_target(self, ppm: float) -> None:
        self._state.ppm_target = _clamp(ppm, self._min_ppm, self._max_ppm)

    def zoom_by_factor(self, factor: float) -> None:
        self.set_zoom_target(self._state.ppm_target * factor)

    def update(self, smoothing: float = 0.1) -> None:
        state = self._state
        state.ppm += (state.ppm_target - state.ppm) * smoothing
        state.ppm = _clamp(state.ppm, self._min_ppm, self._max_ppm)
        state.center += (state.target - state.center) * smoothing

    def begin_pan(self, position: tuple[int, int]) -> None:
        self._pan_anchor = position

    def pan(self, position: tuple[int, int]) -> None:
        if self._pan_anchor is None:
            return
        dx = position[0] - self._pan_anchor[0]
        dy = position[1] - self._pan_anchor[1]
        if dx == 0 and dy == 0:
            return
        ppm = max(self.ppm, 1e-9)
        self._state.center[0] -= dx / ppm
        self._state.center[1] += dy / ppm
        self._state.target[:] = self._state.center
        self._pan_anchor = position

    def end_pan(self) -> None:
        self._pan_anchor = None

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        width, height = self._size
        cx, cy = self._state.center
        sx = width // 2 + int((x - cx) * self._state.ppm)
        sy = height // 2 - int((y - cy) * self._state.ppm)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        width, height = self._size
        cx, cy = self._state.center
        x = (sx - width / 2.0) / max(self._state.ppm, 1e-9) + cx
        y = (height / 2.0 - sy) / max(self._state.ppm, 1e-9) + cy
        return x, y

    def view_rect(self) -> tuple[float, float, float, float]:
        width, height = self._size
        ppm = max(self._state.ppm, 1e-9)
        half_width_world = width / (2.0 * ppm)
        half_height_world = height / (2.0 * ppm)
        cx, cy = self._state.center
        return (
            cx - half_width_world,
            cy - half_height_world,
            cx + half_width_world,
            cy + half_height_world,
        )
