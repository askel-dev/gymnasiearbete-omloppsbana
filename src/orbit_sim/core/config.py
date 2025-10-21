"""Configuration dataclasses for the orbit simulation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PhysicsCfg:
    gravitational_constant: float = 6.674e-11
    earth_mass: float = 5.972e24
    earth_radius: float = 6_371_000.0
    start_position: np.ndarray = field(
        default_factory=lambda: np.array([7_000_000.0, 0.0], dtype=float)
    )
    start_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 7_600.0], dtype=float)
    )
    dt: float = 0.25
    real_time_speed: float = 240.0
    max_substeps: int = 20
    log_every_steps: int = 20
    escape_radius_factor: float = 20.0
    orbit_prediction_interval: float = 1.0
    max_orbit_prediction_samples: int = 2_000
    max_rendered_orbit_points: int = 800
    atm_altitude: float = 120_000.0
    atm_drag_coeff: float = 1.4e-5
    atm_warning_duration: float = 2.0

    @property
    def mu(self) -> float:
        return self.gravitational_constant * self.earth_mass

    @property
    def atmosphere_boundary_radius(self) -> float:
        return self.earth_radius + self.atm_altitude


@dataclass(frozen=True)
class RenderCfg:
    width: int = 1000
    height: int = 800
    windowed_default_size: tuple[int, int] = (1000, 800)
    background_color: tuple[int, int, int] = (0, 34, 72)
    planet_color: tuple[int, int, int] = (255, 183, 77)
    satellite_color: tuple[int, int, int] = (255, 255, 255)
    satellite_pixel_radius: int = 6
    hud_text_color: tuple[int, int, int] = (234, 241, 255)
    hud_text_alpha_base: int = 255
    hud_shadow_color: tuple[int, int, int, int] = (10, 15, 30, 120)
    orbit_primary_color: tuple[int, int, int, int] = (255, 255, 255, 180)
    orbit_secondary_color: tuple[int, int, int, int] = (220, 236, 255, 140)
    orbit_line_width: int = 2
    velocity_arrow_color: tuple[int, int, int] = (255, 220, 180)
    velocity_arrow_scale: float = 0.004
    velocity_arrow_min_pixels: int = 0
    velocity_arrow_max_pixels: int = 90
    velocity_arrow_head_length: int = 10
    velocity_arrow_head_angle_deg: int = 26
    button_color: tuple[int, int, int, int] = (8, 32, 64, int(255 * 0.78))
    button_hover_color: tuple[int, int, int, int] = (18, 52, 94, int(255 * 0.88))
    button_text_color: tuple[int, int, int] = (234, 241, 255)
    button_border_color: tuple[int, int, int, int] = (88, 140, 255, int(255 * 0.55))
    button_hover_border_color: tuple[int, int, int, int] = (118, 180, 255, int(255 * 0.8))
    button_radius: int = 18
    menu_title_color: tuple[int, int, int] = (234, 241, 255)
    menu_subtitle_color: tuple[int, int, int] = (180, 198, 228)
    menu_button_color: tuple[int, int, int, int] = (9, 44, 92, 220)
    menu_button_hover_color: tuple[int, int, int, int] = (24, 74, 140, 235)
    menu_button_border_color: tuple[int, int, int, int] = (255, 255, 255, 50)
    menu_button_text_color: tuple[int, int, int] = (234, 241, 255)
    menu_button_radius: int = 20
    label_background_color: tuple[int, int, int, int] = (12, 18, 30, int(255 * 0.18))
    label_marker_color: tuple[int, int, int] = (46, 209, 195)
    label_text_color: tuple[int, int, int] = (234, 241, 255)
    label_marker_alpha: int = int(255 * 0.9)
    label_marker_hover_radius: int = 22
    label_marker_hover_alpha: int = 255
    label_marker_hover_radius_pixels: int = 6
    label_marker_pin_width: int = 10
    label_marker_pin_height: int = 16
    label_marker_pin_offset: int = 6
    label_marker_pinned_pin_color: tuple[int, int, int] = (248, 252, 255)
    label_marker_pinned_glow_color: tuple[int, int, int] = (255, 255, 255)
    label_marker_pinned_glow_alpha: int = 90
    label_marker_pinned_outline_alpha: int = 210
    label_marker_pinned_radius_pixels: int = 8
    label_marker_pinned_glow_radius: int = 14
    label_pinned_background_color: tuple[int, int, int, int] = (
        16,
        28,
        46,
        int(255 * 0.42),
    )
    label_pinned_badge_color: tuple[int, int, int, int] = (255, 255, 255, 220)
    label_pinned_badge_text_color: tuple[int, int, int] = (18, 36, 64)
    marker_pin_feedback_duration: float = 1.6
    fps_text_alpha: int = int(255 * 0.6)
    starfield_parallax: float = 0.12
    atm_warning_color: tuple[int, int, int] = (255, 176, 120)
    atm_glow_color: tuple[int, int, int] = (255, 120, 80)
    atm_glow_outer_alpha: int = 90
    atm_glow_inner_alpha: int = 180
    atm_glow_radius_factor: float = 2.6
    impact_freeze_delay: float = 0.7
    impact_hud_alpha_factor: float = 0.35
    impact_hud_fade_duration: float = 0.6
    impact_overlay_delay: float = 2.0
    impact_overlay_fade_duration: float = 1.0
    impact_overlay_color: tuple[int, int, int, int] = (12, 18, 30, int(255 * 0.75))
    impact_title_color: tuple[int, int, int] = (255, 214, 130)
    impact_text_color: tuple[int, int, int] = (234, 241, 255)
    shock_ring_color: tuple[int, int, int] = (255, 66, 66)
    shock_ring_duration: float = 2.5
    shock_ring_expansion_factor: float = 6.0
    shock_ring_width: int = 6
    grid_spacing_meters: float = 1_000_000.0
    grid_min_pixel_spacing: float = 42.0
    grid_line_color: tuple[int, int, int] = (200, 208, 220)
    grid_line_alpha: int = 40
    grid_label_color: tuple[int, int, int] = (208, 216, 228)
    grid_label_alpha: int = 180
    grid_axis_label_alpha: int = 160
    grid_label_margin: int = 10
    min_pixels_per_meter: float = 1e-7
    max_pixels_per_meter: float = 1e-2


PHYSICS_CFG = PhysicsCfg()
RENDER_CFG = RenderCfg()


__all__ = ["PHYSICS_CFG", "RENDER_CFG", "PhysicsCfg", "RenderCfg"]
