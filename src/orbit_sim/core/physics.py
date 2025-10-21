"""Physics helpers for the orbit simulation."""
from __future__ import annotations

import math

import numpy as np

from .config import PHYSICS_CFG, PhysicsCfg


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* between *lo* and *hi*."""

    return max(lo, min(hi, value))


def in_atmosphere(r: np.ndarray, cfg: PhysicsCfg = PHYSICS_CFG) -> bool:
    """Return ``True`` if the position ``r`` lies inside the atmosphere."""

    return float(np.linalg.norm(r)) <= cfg.atmosphere_boundary_radius


def atmosphere_depth_ratio(r_magnitude: float, cfg: PhysicsCfg = PHYSICS_CFG) -> float:
    """Normalised penetration depth into the atmosphere."""

    if cfg.atm_altitude <= 0.0:
        return 0.0
    depth = (cfg.atmosphere_boundary_radius - r_magnitude) / cfg.atm_altitude
    return clamp(depth, 0.0, 1.0)


def accel(r: np.ndarray, v: np.ndarray, cfg: PhysicsCfg = PHYSICS_CFG) -> np.ndarray:
    """Compute total acceleration for position ``r`` and velocity ``v``."""

    rmag = float(np.linalg.norm(r))
    if rmag <= 0.0:
        return np.zeros_like(r)

    gravitational = -cfg.mu * r / (rmag**3)
    if rmag <= cfg.atmosphere_boundary_radius:
        depth = atmosphere_depth_ratio(rmag, cfg)
        drag_coeff = cfg.atm_drag_coeff * (0.2 + 0.8 * depth)
        gravitational += -drag_coeff * v
    return gravitational


def rk4_step(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    cfg: PhysicsCfg = PHYSICS_CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the position/velocity state with a classical RK4 step."""

    a1 = accel(r, v, cfg)
    k1_r = v
    k1_v = a1

    a2 = accel(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, cfg)
    k2_r = v + 0.5 * dt * k1_v
    k2_v = a2

    a3 = accel(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, cfg)
    k3_r = v + 0.5 * dt * k2_v
    k3_v = a3

    a4 = accel(r + dt * k3_r, v + dt * k3_v, cfg)
    k4_r = v + dt * k3_v
    k4_v = a4

    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next


def energy_specific(r: np.ndarray, v: np.ndarray, cfg: PhysicsCfg = PHYSICS_CFG) -> float:
    """Specific orbital energy for position ``r`` and velocity ``v``."""

    rmag = float(np.linalg.norm(r))
    vmag2 = float(v[0] * v[0] + v[1] * v[1])
    return 0.5 * vmag2 - cfg.mu / rmag


def eccentricity(r: np.ndarray, v: np.ndarray, cfg: PhysicsCfg = PHYSICS_CFG) -> float:
    """Return the orbital eccentricity for state ``(r, v)``."""

    r3 = np.array([r[0], r[1], 0.0])
    v3 = np.array([v[0], v[1], 0.0])
    h = np.cross(r3, v3)
    e_vec = np.cross(v3, h) / cfg.mu - r3 / np.linalg.norm(r3)
    return float(np.linalg.norm(e_vec[:2]))


def compute_orbit_prediction(
    r_init: np.ndarray,
    v_init: np.ndarray,
    cfg: PhysicsCfg = PHYSICS_CFG,
) -> tuple[float | None, list[tuple[float, float]]]:
    """Predict the next orbital path using the configured physics constants."""

    eps = energy_specific(r_init, v_init, cfg)
    if eps >= 0.0:
        return None, []

    a = -cfg.mu / (2.0 * eps)
    period = 2.0 * math.pi * math.sqrt(a**3 / cfg.mu)
    estimated_samples = max(360, int(period / cfg.orbit_prediction_interval))
    num_samples = max(2, min(cfg.max_orbit_prediction_samples, estimated_samples))
    dt = period / num_samples

    r = r_init.copy()
    v = v_init.copy()
    points: list[tuple[float, float]] = []
    for _ in range(num_samples + 1):
        points.append((float(r[0]), float(r[1])))
        r, v = rk4_step(r, v, dt, cfg)

    return period, points


__all__ = [
    "PHYSICS_CFG",
    "PhysicsCfg",
    "accel",
    "atmosphere_depth_ratio",
    "clamp",
    "compute_orbit_prediction",
    "eccentricity",
    "energy_specific",
    "in_atmosphere",
    "rk4_step",
]
