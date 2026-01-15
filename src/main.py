# src/main.py
"""
OrbitLab - Interactive Orbit Simulator
=======================================

An educational platform for exploring orbital mechanics through
numerical integration and real-time visualization.

Version: 1.5
"""

import math
import os
import random
import sys
import time

import numpy as np
import pygame
from pygame import gfxdraw
from pygame.locals import DOUBLEBUF, FULLSCREEN, NOFRAME, RESIZABLE
from collections import deque
from dataclasses import dataclass

from physics import G, rk4_step, euler_step
from logging_utils import RunLogger

from planet_generator import get_preset, get_planet_sprite, Planet

# =======================
#   CURRENT PLANET
# =======================
# The central body for the simulation - can be changed via presets or custom generation
PLANET_PRESETS = ["earth", "mars", "moon", "jupiter", "neptune", "venus", "saturn", "uranus", "io", "mercury"]
current_planet: Planet = get_preset("earth")
current_planet_index = 0

# =======================
#   PHYSICS CONSTANTS
# =======================

# =======================
#   SCENARIOS
# =======================
@dataclass(frozen=True)
class Scenario:
    """Preset initial conditions for different orbit types."""
    key: str
    name: str
    r0: float              # Starting distance from planet center (meters)
    velocity: float        # Starting velocity magnitude (m/s), positive = prograde
    description: str

    def position_vector(self) -> np.ndarray:
        """Returns starting position vector (on +X axis)."""
        return np.array([self.r0, 0.0], dtype=float)

    def velocity_vector(self) -> np.ndarray:
        """Returns starting velocity vector (along +Y axis for prograde)."""
        return np.array([0.0, self.velocity], dtype=float)


def generate_default_scenarios(planet: Planet) -> tuple[Scenario, ...]:
    """
    Generate default scenarios for a planet without custom definitions.
    
    Uses orbital mechanics formulas to calculate appropriate velocities:
    - Circular orbit: v = sqrt(mu / r)
    - Escape velocity: v = sqrt(2 * mu / r)
    - Elliptical orbit at periapsis: v = sqrt(mu * (1 + e) / r)
    """
    # Start 10% above the planet's surface
    r0 = planet.radius * 1.1
    
    v_circular = math.sqrt(planet.mu / r0)
    v_escape = math.sqrt(2 * planet.mu / r0)
    
    # Highly elliptical orbit (e ≈ 0.7) - velocity at periapsis
    e_target = 0.7
    v_elliptical = math.sqrt(planet.mu * (1 + e_target) / r0)
    
    return (
        Scenario(
            key="orbit",
            name="Circular Orbit",
            r0=r0,
            velocity=v_circular,
            description=f"Stable circular orbit (~{v_circular/1000:.1f} km/s prograde).",
        ),
        Scenario(
            key="suborbital",
            name="Suborbital",
            r0=r0,
            velocity=v_circular * 0.79,
            description=f"Too slow for orbit (~{v_circular*0.79/1000:.1f} km/s).",
        ),
        Scenario(
            key="escape",
            name="Escape",
            r0=r0,
            velocity=v_escape * 1.1,
            description=f"Exceeds escape velocity (~{v_escape*1.1/1000:.1f} km/s).",
        ),
        Scenario(
            key="parabolic",
            name="Parabolic",
            r0=r0,
            velocity=v_escape,
            description=f"Near escape threshold (~{v_escape/1000:.1f} km/s).",
        ),
        Scenario(
            key="retrograde",
            name="Retrograde",
            r0=r0,
            velocity=-v_circular,
            description=f"Circular orbit velocity but retrograde (~{v_circular/1000:.1f} km/s).",
        ),
        # New universal scenarios
        Scenario(
            key="elliptical",
            name="Highly Elliptical",
            r0=r0,
            velocity=v_elliptical,
            description=f"Molniya-style orbit with eccentricity ~0.7 (~{v_elliptical/1000:.1f} km/s).",
        ),
        Scenario(
            key="freefall",
            name="Free Fall",
            r0=r0,
            velocity=0,
            description="Pure radial drop toward planet center.",
        ),
    )


# Planet-specific scenario definitions with hand-tuned values
# Values calculated using: v_circular = sqrt(mu/r), v_escape = sqrt(2*mu/r)
# v_elliptical = sqrt(mu*(1+e)/r) at periapsis for target eccentricity e
# r_sync = (mu*T^2 / 4*pi^2)^(1/3) for synchronous orbit with period T
PLANET_SCENARIOS: dict[str, tuple[Scenario, ...]] = {
    "Earth": (
        Scenario("leo", "LEO", r0=7_000_000, velocity=7_546, description="Low Earth orbit (~7.5 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=7_000_000, velocity=6_000, description="Too slow for orbit (~6.0 km/s)."),
        Scenario("escape", "Escape", r0=7_000_000, velocity=11_500, description="Exceeds escape velocity (~11.5 km/s)."),
        Scenario("parabolic", "Parabolic", r0=7_000_000, velocity=10_670, description="Near escape threshold (~10.7 km/s)."),
        Scenario("retrograde", "Retrograde", r0=7_000_000, velocity=-7_546, description="LEO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=7_000_000, velocity=9_837, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=7_000_000, velocity=0, description="Pure radial drop toward Earth center."),
        Scenario("geo", "Geostationary", r0=42_164_000, velocity=3_075, description="24-hour orbital period at 35,786 km altitude."),
    ),
    "Mars": (
        Scenario("lmo", "Low Mars Orbit", r0=3_800_000, velocity=3_360, description="Low Mars orbit (~3.4 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=3_800_000, velocity=2_700, description="Too slow for orbit (~2.7 km/s)."),
        Scenario("escape", "Escape", r0=3_800_000, velocity=5_200, description="Exceeds escape velocity (~5.2 km/s)."),
        Scenario("parabolic", "Parabolic", r0=3_800_000, velocity=4_750, description="Near escape threshold (~4.8 km/s)."),
        Scenario("retrograde", "Retrograde", r0=3_800_000, velocity=-3_360, description="LMO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=3_800_000, velocity=4_380, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=3_800_000, velocity=0, description="Pure radial drop toward Mars center."),
        Scenario("areo", "Areosynchronous", r0=20_428_000, velocity=1_448, description="24h 37m orbital period (Mars day)."),
    ),
    "Moon": (
        Scenario("llo", "Low Lunar Orbit", r0=1_937_000, velocity=1_590, description="Low lunar orbit (~1.6 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=1_937_000, velocity=1_256, description="Too slow for orbit (~1.3 km/s)."),
        Scenario("escape", "Escape", r0=1_937_000, velocity=2_474, description="Exceeds escape velocity (~2.5 km/s)."),
        Scenario("parabolic", "Parabolic", r0=1_937_000, velocity=2_249, description="Near escape threshold (~2.2 km/s)."),
        Scenario("retrograde", "Retrograde", r0=1_937_000, velocity=-1_590, description="LLO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=1_937_000, velocity=2_074, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=1_937_000, velocity=0, description="Pure radial drop toward Moon center."),
    ),
    "Jupiter": (
        Scenario("ljo", "Low Jupiter Orbit", r0=75_000_000, velocity=41_070, description="Low Jupiter orbit (~41 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=75_000_000, velocity=32_000, description="Too slow for orbit (~32 km/s)."),
        Scenario("escape", "Escape", r0=75_000_000, velocity=64_000, description="Exceeds escape velocity (~64 km/s)."),
        Scenario("parabolic", "Parabolic", r0=75_000_000, velocity=58_080, description="Near escape threshold (~58 km/s)."),
        Scenario("retrograde", "Retrograde", r0=75_000_000, velocity=-41_070, description="LJO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=75_000_000, velocity=53_590, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=75_000_000, velocity=0, description="Pure radial drop toward Jupiter center."),
    ),
    "Neptune": (
        Scenario("lno", "Low Neptune Orbit", r0=27_000_000, velocity=15_909, description="Low Neptune orbit (~15.9 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=27_000_000, velocity=12_568, description="Too slow for orbit (~12.6 km/s)."),
        Scenario("escape", "Escape", r0=27_000_000, velocity=24_749, description="Exceeds escape velocity (~24.7 km/s)."),
        Scenario("parabolic", "Parabolic", r0=27_000_000, velocity=22_499, description="Near escape threshold (~22.5 km/s)."),
        Scenario("retrograde", "Retrograde", r0=27_000_000, velocity=-15_909, description="LNO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=27_000_000, velocity=20_760, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=27_000_000, velocity=0, description="Pure radial drop toward Neptune center."),
    ),
    "Venus": (
        Scenario("lvo", "Low Venus Orbit", r0=6_700_000, velocity=6_963, description="Low Venus orbit (~7.0 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=6_700_000, velocity=5_501, description="Too slow for orbit (~5.5 km/s)."),
        Scenario("escape", "Escape", r0=6_700_000, velocity=10_832, description="Exceeds escape velocity (~10.8 km/s)."),
        Scenario("parabolic", "Parabolic", r0=6_700_000, velocity=9_847, description="Near escape threshold (~9.8 km/s)."),
        Scenario("retrograde", "Retrograde", r0=6_700_000, velocity=-6_963, description="LVO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=6_700_000, velocity=9_082, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=6_700_000, velocity=0, description="Pure radial drop toward Venus center."),
    ),
    "Saturn": (
        Scenario("lso", "Low Saturn Orbit", r0=64_000_000, velocity=24_300, description="Low Saturn orbit (~24 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=64_000_000, velocity=19_000, description="Too slow for orbit (~19 km/s)."),
        Scenario("escape", "Escape", r0=64_000_000, velocity=38_000, description="Exceeds escape velocity (~38 km/s)."),
        Scenario("parabolic", "Parabolic", r0=64_000_000, velocity=34_360, description="Near escape threshold (~34 km/s)."),
        Scenario("retrograde", "Retrograde", r0=64_000_000, velocity=-24_300, description="LSO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=64_000_000, velocity=31_700, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=64_000_000, velocity=0, description="Pure radial drop toward Saturn center."),
    ),
    "Uranus": (
        Scenario("luo", "Low Uranus Orbit", r0=28_000_000, velocity=14_384, description="Low Uranus orbit (~14.4 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=28_000_000, velocity=11_363, description="Too slow for orbit (~11.4 km/s)."),
        Scenario("escape", "Escape", r0=28_000_000, velocity=22_378, description="Exceeds escape velocity (~22.4 km/s)."),
        Scenario("parabolic", "Parabolic", r0=28_000_000, velocity=20_343, description="Near escape threshold (~20.3 km/s)."),
        Scenario("retrograde", "Retrograde", r0=28_000_000, velocity=-14_384, description="LUO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=28_000_000, velocity=18_760, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=28_000_000, velocity=0, description="Pure radial drop toward Uranus center."),
    ),
    "Mercury": (
        Scenario("lmo", "Low Mercury Orbit", r0=2_700_000, velocity=2_860, description="Low Mercury orbit (~2.9 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=2_700_000, velocity=2_250, description="Too slow for orbit (~2.3 km/s)."),
        Scenario("escape", "Escape", r0=2_700_000, velocity=4_450, description="Exceeds escape velocity (~4.5 km/s)."),
        Scenario("parabolic", "Parabolic", r0=2_700_000, velocity=4_040, description="Near escape threshold (~4.0 km/s)."),
        Scenario("retrograde", "Retrograde", r0=2_700_000, velocity=-2_860, description="LMO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=2_700_000, velocity=3_730, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=2_700_000, velocity=0, description="Pure radial drop toward Mercury center."),
    ),
    "Io": (
        Scenario("lio", "Low Io Orbit", r0=2_000_000, velocity=1_728, description="Low Io orbit (~1.7 km/s prograde)."),
        Scenario("suborbital", "Suborbital", r0=2_000_000, velocity=1_360, description="Too slow for orbit (~1.4 km/s)."),
        Scenario("escape", "Escape", r0=2_000_000, velocity=2_700, description="Exceeds escape velocity (~2.7 km/s)."),
        Scenario("parabolic", "Parabolic", r0=2_000_000, velocity=2_440, description="Near escape threshold (~2.4 km/s)."),
        Scenario("retrograde", "Retrograde", r0=2_000_000, velocity=-1_728, description="LIO velocity but retrograde."),
        # New scenarios
        Scenario("elliptical", "Highly Elliptical", r0=2_000_000, velocity=2_254, description="Molniya-style orbit with eccentricity ~0.7."),
        Scenario("freefall", "Free Fall", r0=2_000_000, velocity=0, description="Pure radial drop toward Io center."),
    ),
}


def get_scenarios_for_planet(planet: Planet) -> tuple[Scenario, ...]:
    """Get the scenarios for a specific planet, generating defaults if needed."""
    if planet.name in PLANET_SCENARIOS:
        return PLANET_SCENARIOS[planet.name]
    return generate_default_scenarios(planet)


SCENARIO_FLASH_DURATION = 2.0


def get_default_r0() -> float:
    """Get default starting distance for current planet (used for scaling)."""
    scenarios = get_scenarios_for_planet(current_planet)
    if scenarios:
        return scenarios[0].r0
    return current_planet.radius * 1.1

# =======================
#   SIMULATOR SETTINGS
# =======================
DT_PHYS = 0.25                 # physics timestep (seconds)
REAL_TIME_SPEED = 240.0        # sim-seconds per real-second
MAX_SUBSTEPS = 20              # protection against too many physics steps/frame
LOG_EVERY_STEPS = 20           # log every 20th physics step
ESCAPE_RADIUS_FACTOR = 20.0
ORBIT_PREDICTION_INTERVAL = 1.0
MAX_ORBIT_PREDICTION_SAMPLES = 2_000
MAX_RENDERED_ORBIT_POINTS = 800
integrator = "RK4"              # "RK4" = Runge-Kutta 4, "Euler" = Euler's method

# =======================
#   DISPLAY SETTINGS
# =======================
WIDTH, HEIGHT = 1000, 800
WINDOWED_DEFAULT_SIZE = (WIDTH, HEIGHT)
FULLSCREEN_ENABLED = False      # Start in fullscreen by default

# Colors
BACKGROUND_COLOR = (0, 34, 72)
SATELLITE_COLOR = (255, 255, 255)
SATELLITE_PIXEL_RADIUS = 6
HUD_TEXT_COLOR = (234, 241, 255)
HUD_SHADOW_COLOR = (10, 15, 30, 120)
# HUD color coding for at-a-glance readability
HUD_LABEL_COLOR = (180, 185, 195)           # White/Gray for static labels
HUD_VALUE_COLOR = (100, 220, 255)           # Cyan/Light Blue for physics values
HUD_WARNING_COLOR = (255, 180, 60)          # Yellow/Orange for warnings/critical values
ORBIT_PRIMARY_COLOR = (255, 255, 255, 180)
ORBIT_LINE_WIDTH = 2
VEL_ARROW_COLOR = (255, 220, 180)
VEL_ARROW_SCALE = 0.004
VEL_ARROW_MIN_PIXELS = 0
VEL_ARROW_MAX_PIXELS = 90
VEL_ARROW_HEAD_LENGTH = 10
VEL_ARROW_HEAD_ANGLE_DEG = 26
# Menu color palette
MENU_BACKGROUND = (10, 15, 30)
BUTTON_IDLE = (30, 40, 60, 200)
BUTTON_HOVER = (50, 70, 100, 220)
BUTTON_SELECTED = (0, 120, 215, 255)
BUTTON_START = (40, 160, 80, 255)
BUTTON_START_HOVER = (60, 190, 100, 255)
BUTTON_OUTLINE = (70, 90, 120)
PREVIEW_PLACEHOLDER = (25, 35, 55, 200)
TEXT_WHITE = (255, 255, 255)
TEXT_GRAY = (180, 180, 180)
BUTTON_RADIUS = 12
MENU_SUBTITLE_COLOR = (180, 180, 180)
MENU_SECTION_TITLE_COLOR = (180, 180, 180)
MENU_PANEL_COLOR = (15, 20, 35, int(255 * 0.85))
MENU_COLUMN_DIVIDER_COLOR = (50, 60, 80, 120)
MENU_FOOTER_COLOR = (8, 12, 25, int(255 * 0.92))
MENU_BRIEFING_TEXT_COLOR = (200, 210, 230)
LABEL_BACKGROUND_COLOR = (12, 18, 30, int(255 * 0.18))
LABEL_MARKER_COLOR = (46, 209, 195)
LABEL_TEXT_COLOR = (234, 241, 255)
LABEL_MARKER_ALPHA = int(255 * 0.9)
LABEL_MARKER_HOVER_RADIUS = 22
LABEL_MARKER_HOVER_ALPHA = 255
LABEL_MARKER_HOVER_RADIUS_PIXELS = 6
LABEL_MARKER_PIN_WIDTH = 10
LABEL_MARKER_PIN_HEIGHT = 16
LABEL_MARKER_PIN_OFFSET = 6
FPS_TEXT_ALPHA = int(255 * 0.6)

# Impact settings
IMPACT_FREEZE_DELAY = 0.7
IMPACT_OVERLAY_DELAY = 2.0
IMPACT_OVERLAY_FADE_DURATION = 1.0
IMPACT_OVERLAY_COLOR = (12, 18, 30, int(255 * 0.75))
IMPACT_TITLE_COLOR = (255, 214, 130)
IMPACT_TEXT_COLOR = (234, 241, 255)
SHOCK_RING_COLOR = (255, 66, 66)
SHOCK_RING_DURATION = 2.5
SHOCK_RING_EXPANSION_FACTOR = 6.0
SHOCK_RING_WIDTH = 6

# Grid settings
GRID_SPACING_METERS = 1_000_000.0
GRID_MIN_PIXEL_SPACING = 42.0
GRID_LINE_COLOR = (200, 208, 220)
GRID_LINE_ALPHA = 40
GRID_LABEL_COLOR = (208, 216, 228)
GRID_LABEL_ALPHA = 180
GRID_AXIS_LABEL_ALPHA = 160
GRID_LABEL_MARGIN = 10

# =======================
#   HELPER FUNCTIONS
# =======================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_pixels_per_meter(width: int, height: int) -> float:
    """
    Calculate pixels per meter based on planet size and orbit distance.
    
    Adjusts zoom so that:
    - The planet appears at a reasonable size on screen (target: ~20-25% of screen height)
    - The orbit is visible and well-framed
    """
    r0 = get_default_r0()
    planet_radius = current_planet.radius
    
    # Calculate zoom based on planet radius (target: planet diameter = 20-45% of screen height)
    screen_size = min(width, height)
    # Use a target size that scales with planet size - larger planets get slightly more screen space
    target_planet_fraction = 0.20 + min(0.25, (planet_radius / 70_000_000) * 0.25)  # Up to 45% for very large planets
    target_planet_diameter_px = screen_size * target_planet_fraction
    planet_scale = target_planet_diameter_px / (2.0 * planet_radius)
    
    # Calculate zoom based on orbit distance (target: orbit fits well in view)
    orbit_scale = min(width, height) / (2.0 * r0) * 0.60
    
    # Use a weighted approach: prefer planet scale but ensure orbit is visible
    # If orbit scale is much smaller, use a compromise
    if orbit_scale < planet_scale * 0.5:
        # Orbit is much smaller - use a compromise to show both
        return (planet_scale + orbit_scale) / 2.0
    else:
        # Use the smaller scale to ensure both planet and orbit are visible
        return min(planet_scale, orbit_scale)


def update_display_metrics(width: int, height: int) -> None:
    global WIDTH, HEIGHT, PIXELS_PER_METER
    WIDTH = width
    HEIGHT = height
    PIXELS_PER_METER = compute_pixels_per_meter(width, height)


PIXELS_PER_METER = compute_pixels_per_meter(WIDTH, HEIGHT)
MIN_PPM = 1e-7
MAX_PPM = 1e-2


def world_to_screen(x, y, ppm, camera_center=(0.0, 0.0)):
    cx, cy = camera_center
    sx = WIDTH // 2 + int((x - cx) * ppm)
    sy = HEIGHT // 2 - int((y - cy) * ppm)
    return sx, sy


def energy_specific(r, v, mu: float) -> float:
    """Calculate specific orbital energy using the current planet's mu."""
    rmag = np.linalg.norm(r)
    vmag2 = v[0]*v[0] + v[1]*v[1]
    return 0.5*vmag2 - mu/rmag


def eccentricity(r, v, mu: float) -> float:
    """Calculate orbital eccentricity using the current planet's mu."""
    r3 = np.array([r[0], r[1], 0.0])
    v3 = np.array([v[0], v[1], 0.0])
    h = np.cross(r3, v3)
    e_vec = np.cross(v3, h)/mu - r3/np.linalg.norm(r3)
    return np.linalg.norm(e_vec[:2])


def compute_satellite_radius(_: float) -> int:
    return SATELLITE_PIXEL_RADIUS


def _format_megameters(value_m: float) -> str:
    value_mm = value_m / 1_000_000.0
    abs_mm = abs(value_mm)
    if abs_mm >= 1000:
        text = f"{value_mm:,.0f}"
    elif abs_mm >= 100:
        text = f"{value_mm:,.0f}"
    elif abs_mm >= 10:
        text = f"{value_mm:,.1f}"
    else:
        text = f"{value_mm:,.2f}"
    return text.rstrip("0").rstrip(".")


# =======================
#   DRAWING FUNCTIONS
# =======================
def draw_planet(surface: pygame.Surface, planet: Planet, position: tuple[int, int], radius_px: int) -> None:
    """Draw a planet at the given screen position using procedural sprite generation."""
    if radius_px <= 0:
        return
    diameter = radius_px * 2
    
    # Optimization: If planet is massive (zoomed in very close),
    # just draw a simple circle to avoid generating/blitting huge textures
    # which would cause lag or Out Of Memory errors.
    if diameter > 2048:
        base_color = planet.base_colors[0] if planet.base_colors else (128, 128, 128)
        pygame.draw.circle(surface, base_color, position, radius_px)
        return

    sprite = get_planet_sprite(planet, diameter)
    rect = sprite.get_rect(center=position)
    surface.blit(sprite, rect)


def draw_satellite(surface: pygame.Surface, position: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        return
    pygame.draw.circle(surface, SATELLITE_COLOR, position, radius)


def draw_velocity_arrow(
    surface: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    head_length: int,
    head_angle_deg: float,
) -> None:
    pygame.draw.line(surface, VEL_ARROW_COLOR, start, end, 2)
    angle = math.atan2(start[1] - end[1], end[0] - start[0])
    head_angle = math.radians(head_angle_deg)
    left = (
        int(end[0] - head_length * math.cos(angle - head_angle)),
        int(end[1] + head_length * math.sin(angle - head_angle)),
    )
    right = (
        int(end[0] - head_length * math.cos(angle + head_angle)),
        int(end[1] + head_length * math.sin(angle + head_angle)),
    )
    pygame.draw.polygon(surface, VEL_ARROW_COLOR, [end, left, right])


def draw_marker_pin(
    surface: pygame.Surface,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int],
    alpha: int,
    orientation: str,
) -> None:
    if alpha <= 0:
        return
    half_width = LABEL_MARKER_PIN_WIDTH // 2
    offset = LABEL_MARKER_PIN_OFFSET
    height = LABEL_MARKER_PIN_HEIGHT

    if orientation == "pericenter":
        base_y = position[1] + offset
        tip = (position[0], position[1] + height)
    else:
        base_y = position[1] - offset
        tip = (position[0], position[1] - height)

    base_left = (position[0] - half_width, base_y)
    base_right = (position[0] + half_width, base_y)

    pygame.draw.polygon(surface, (*color, alpha), [base_left, base_right, tip])
    pygame.draw.line(surface, (*color, min(255, int(alpha * 0.95))), position, tip, 2)


def draw_orbit_line(
    surface: pygame.Surface,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    points: list[tuple[int, int]],
    width: int,
) -> None:
    if len(points) < 2:
        return
    if width <= 1:
        pygame.draw.aalines(surface, color, False, points)
    else:
        pygame.draw.lines(surface, color, False, points, width)
        pygame.draw.aalines(surface, color, False, points)


def downsample_points(
    points: list[tuple[float, float]], max_points: int
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def draw_coordinate_grid(
    surface: pygame.Surface,
    ppm: float,
    camera_center: tuple[float, float],
    *,
    tick_font: pygame.font.Font,
    axis_font: pygame.font.Font,
) -> None:
    if ppm <= 0.0:
        return

    spacing = GRID_SPACING_METERS
    spacing_px = spacing * ppm
    if spacing_px <= 0.0:
        return

    if spacing_px < GRID_MIN_PIXEL_SPACING:
        multiplier = max(1, math.ceil(GRID_MIN_PIXEL_SPACING / spacing_px))
        spacing *= multiplier
        spacing_px = spacing * ppm
        if spacing_px < GRID_MIN_PIXEL_SPACING:
            return

    width, height = surface.get_size()
    if width <= 0 or height <= 0:
        return

    cx, cy = camera_center
    half_width_world = width / (2.0 * ppm)
    half_height_world = height / (2.0 * ppm)
    world_left = cx - half_width_world
    world_right = cx + half_width_world
    world_bottom = cy - half_height_world
    world_top = cy + half_height_world

    line_color = (*GRID_LINE_COLOR, GRID_LINE_ALPHA)
    origin_line_color = (*GRID_LINE_COLOR, min(255, GRID_LINE_ALPHA + 150))
    origin_line_width = 3

    start_x = math.floor(world_left / spacing) * spacing
    max_vertical = int(math.ceil((world_right - world_left) / spacing)) + 3
    for i in range(max_vertical):
        x_world = start_x + i * spacing
        if x_world > world_right + spacing:
            break
        sx, _ = world_to_screen(x_world, cy, ppm, camera_center)
        if 0 <= sx <= width:
            draw_x = int(clamp(sx, 0, width))
            is_origin_line = abs(x_world) < spacing * 0.5
            color = origin_line_color if is_origin_line else line_color
            width_px = origin_line_width if is_origin_line else 1
            pygame.draw.line(surface, color, (draw_x, 0), (draw_x, height), width_px)
            if 0 <= sx <= width and GRID_LABEL_MARGIN < sx < width - 140:
                label_text = _format_megameters(x_world)
                if label_text:
                    label_surf = tick_font.render(label_text, True, GRID_LABEL_COLOR)
                    if GRID_LABEL_ALPHA < 255:
                        label_surf = label_surf.copy()
                        label_surf.set_alpha(GRID_LABEL_ALPHA)
                    rect = label_surf.get_rect()
                    rect.midtop = (sx, GRID_LABEL_MARGIN)
                    if rect.bottom <= height:
                        surface.blit(label_surf, rect)

    start_y = math.floor(world_bottom / spacing) * spacing
    max_horizontal = int(math.ceil((world_top - world_bottom) / spacing)) + 3
    for i in range(max_horizontal):
        y_world = start_y + i * spacing
        if y_world > world_top + spacing:
            break
        _, sy = world_to_screen(cx, y_world, ppm, camera_center)
        if 0 <= sy <= height:
            draw_y = int(clamp(sy, 0, height))
            is_origin_line = abs(y_world) < spacing * 0.5
            color = origin_line_color if is_origin_line else line_color
            width_px = origin_line_width if is_origin_line else 1
            pygame.draw.line(surface, color, (0, draw_y), (width, draw_y), width_px)
            if 0 <= sy <= height and GRID_LABEL_MARGIN * 2 < sy < height - GRID_LABEL_MARGIN:
                label_text = _format_megameters(y_world)
                if label_text:
                    label_surf = tick_font.render(label_text, True, GRID_LABEL_COLOR)
                    if GRID_LABEL_ALPHA < 255:
                        label_surf = label_surf.copy()
                        label_surf.set_alpha(GRID_LABEL_ALPHA)
                    rect = label_surf.get_rect()
                    rect.midright = (width - GRID_LABEL_MARGIN, sy)
                    if rect.left >= 0:
                        surface.blit(label_surf, rect)

    axis_color = GRID_LABEL_COLOR
    axis_label = axis_font.render("Y [Mm]", True, axis_color)
    if GRID_AXIS_LABEL_ALPHA < 255:
        axis_label = axis_label.copy()
        axis_label.set_alpha(GRID_AXIS_LABEL_ALPHA)
    axis_rect = axis_label.get_rect()
    axis_rect.bottomright = (width - GRID_LABEL_MARGIN, height - GRID_LABEL_MARGIN)
    surface.blit(axis_label, axis_rect)

    y_axis_label = axis_font.render("X [Mm]", True, axis_color)
    if GRID_AXIS_LABEL_ALPHA < 255:
        y_axis_label = y_axis_label.copy()
        y_axis_label.set_alpha(GRID_AXIS_LABEL_ALPHA)
    y_axis_rect = y_axis_label.get_rect()
    y_axis_rect.topright = (width - GRID_LABEL_MARGIN, GRID_LABEL_MARGIN)
    surface.blit(y_axis_label, y_axis_rect)


def compute_orbit_prediction(r_init: np.ndarray, v_init: np.ndarray, mu: float) -> tuple[float | None, list[tuple[float, float]]]:
    """Compute predicted orbit trajectory using the planet's gravitational parameter."""
    eps = energy_specific(r_init, v_init, mu)
    
    r = r_init.copy()
    v = v_init.copy()
    points: list[tuple[float, float]] = []
    
    if eps < 0.0:
        # BOUND ORBIT (elliptical): compute one full period
        a = -mu / (2.0 * eps)
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        estimated_samples = max(360, int(period / ORBIT_PREDICTION_INTERVAL))
        num_samples = max(2, min(MAX_ORBIT_PREDICTION_SAMPLES, estimated_samples))
        dt = period / num_samples
        
        for _ in range(num_samples + 1):
            points.append((float(r[0]), float(r[1])))
            if integrator == "RK4":
                r, v = rk4_step(r, v, dt, mu)
            elif integrator == "Euler":
                r, v = euler_step(r, v, dt, mu)
        
        return period, points
    else:
        # OPEN TRAJECTORY (parabolic/hyperbolic): propagate until far away
        r0_mag = np.linalg.norm(r_init)
        escape_limit = ESCAPE_RADIUS_FACTOR * r0_mag  # Use existing factor
        v_mag = np.linalg.norm(v_init)
        # Estimate time to reach escape limit (rough approximation)
        estimated_time = escape_limit / max(v_mag, 1.0) * 2.0
        dt = estimated_time / MAX_ORBIT_PREDICTION_SAMPLES
        
        for _ in range(MAX_ORBIT_PREDICTION_SAMPLES):
            points.append((float(r[0]), float(r[1])))
            if np.linalg.norm(r) > escape_limit:
                break
            if integrator == "RK4":
                r, v = rk4_step(r, v, dt, mu)
            elif integrator == "Euler":
                r, v = euler_step(r, v, dt, mu)
        
        return None, points  # No period for open trajectories


# =======================
#   STARFIELD CLASS
# =======================
@dataclass
class Star:
    """Represents a single star in the infinite parallax starfield."""
    x: float           # Base x position (0 to screen width)
    y: float           # Base y position (0 to screen height)
    size: int          # Radius in pixels (1-3)
    brightness: int    # Gray value (0-255), linked to depth
    depth: float       # Parallax depth factor (0.1 = far, 0.5 = close)


class Starfield:
    """
    Generates and renders an infinite parallax starfield background.
    
    Stars have varying depths that affect both their parallax speed and brightness:
    - Far stars (depth ~0.1): Move slowly, appear dimmer (dark gray)
    - Close stars (depth ~0.5): Move faster, appear brighter (white)
    """
    
    def __init__(self, num_stars: int = 200, seed: int = 42, base_width: int = 2000, base_height: int = 1500):
        """
        Generate starfield with consistent positions using a fixed seed.
        
        Args:
            num_stars: Number of stars to generate (default 200)
            seed: Random seed for reproducible star placement
            base_width: Base width for star positioning (stars will wrap to actual screen size)
            base_height: Base height for star positioning
        """
        rng = random.Random(seed)
        self.base_width = base_width
        self.base_height = base_height
        self.stars: list[Star] = []
        
        for _ in range(num_stars):
            # Random position within base dimensions
            x = rng.uniform(0, base_width)
            y = rng.uniform(0, base_height)
            
            # Random size (1-3 pixels radius)
            size = rng.randint(1, 3)
            
            # Random depth: 0.1 (very far) to 0.5 (closer)
            depth = rng.uniform(0.1, 0.5)
            
            # Brightness linked to depth: far stars are dim, close stars are bright
            # Map depth [0.1, 0.5] to brightness [80, 255]
            # Linear interpolation: brightness = 80 + (depth - 0.1) / 0.4 * 175
            brightness = int(80 + (depth - 0.1) / 0.4 * 175)
            brightness = max(80, min(255, brightness))  # Clamp to valid range
            
            self.stars.append(Star(x, y, size, brightness, depth))
    
    def draw(self, surface: pygame.Surface, camera_x: float = 0.0, camera_y: float = 0.0) -> None:
        """
        Draw the starfield with infinite parallax scrolling.
        
        Stars shift based on camera position and their individual depth values,
        creating a layered 3D effect. Stars wrap around screen edges for infinite scrolling.
        
        Args:
            surface: Pygame surface to draw on
            camera_x: Camera x position in world coordinates
            camera_y: Camera y position in world coordinates
        """
        width, height = surface.get_size()
        
        for star in self.stars:
            # Calculate parallax offset - stars move opposite to camera, scaled by depth
            # Deeper stars (higher depth) move more, creating parallax effect
            # Use modulo for infinite wrapping
            parallax_x = (star.x - camera_x * star.depth * 0.00001) % width
            parallax_y = (star.y + camera_y * star.depth * 0.00001) % height  # +y because screen Y is inverted
            
            # Create grayscale color from brightness
            star_color = (star.brightness, star.brightness, star.brightness)
            
            # Draw star - use set_at for 1px stars (faster), circle for larger
            screen_x = int(parallax_x)
            screen_y = int(parallax_y)
            
            if star.size == 1:
                if 0 <= screen_x < width and 0 <= screen_y < height:
                    surface.set_at((screen_x, screen_y), star_color)
            else:
                pygame.draw.circle(surface, star_color, (screen_x, screen_y), star.size)


# =======================
#   PLANET VISUAL EFFECTS
# =======================
def draw_planet_atmosphere(
    surface: pygame.Surface,
    position: tuple[int, int],
    planet_radius_px: int,
    atmosphere_color: tuple[int, int, int] = (135, 206, 250),  # Light sky blue
) -> None:
    """
    Draw atmospheric glow effect behind the planet.
    Renders 3 concentric circles with decreasing alpha for a soft glow.
    """
    if planet_radius_px <= 0:
        return
    
    # Create a surface for the atmosphere (needs alpha)
    glow_layers = [
        (planet_radius_px + int(planet_radius_px * 0.25), 15),  # Outer glow: 25% larger, alpha 15
        (planet_radius_px + int(planet_radius_px * 0.15), 20),  # Middle glow: 15% larger, alpha 20
        (planet_radius_px + int(planet_radius_px * 0.08), 25),  # Inner glow: 8% larger, alpha 25
    ]
    
    for glow_radius, alpha in glow_layers:
        if glow_radius > 0:
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface,
                (*atmosphere_color, alpha),
                (glow_radius, glow_radius),
                glow_radius,
            )
            glow_rect = glow_surface.get_rect(center=position)
            surface.blit(glow_surface, glow_rect)


def draw_planet_night_side(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius_px: int,
    shadow_side: str = "left",
    shadow_alpha: int = 100,
) -> None:
    """
    Draw a semi-transparent shadow on one half of the planet to simulate 3D shading.
    
    Args:
        surface: The surface to draw on
        position: Center position of the planet
        radius_px: Planet radius in pixels
        shadow_side: "left" or "right" - which side is in shadow
        shadow_alpha: Transparency of the shadow (0-255)
    """
    if radius_px <= 0:
        return
    
    diameter = radius_px * 2
    
    # Create a circular mask surface
    shadow_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Draw the full circle mask
    pygame.draw.circle(shadow_surface, (0, 0, 0, shadow_alpha), (radius_px, radius_px), radius_px)
    
    # Clear the lit half by drawing a transparent rectangle over it
    if shadow_side == "left":
        # Shadow on left, so clear the right half
        pygame.draw.rect(shadow_surface, (0, 0, 0, 0), (radius_px, 0, radius_px, diameter))
    else:
        # Shadow on right, so clear the left half
        pygame.draw.rect(shadow_surface, (0, 0, 0, 0), (0, 0, radius_px, diameter))
    
    shadow_rect = shadow_surface.get_rect(center=position)
    surface.blit(shadow_surface, shadow_rect)


# =======================
#   BUTTON CLASS
# =======================
class Button:
    """Simple rectangular button with hover feedback and callbacks."""

    def __init__(self, rect, text, callback, text_getter=None):
        self.rect = pygame.Rect(rect)
        self._text = text
        self._callback = callback
        self._text_getter = text_getter

    def get_text(self):
        if self._text_getter is not None:
            return self._text_getter()
        return self._text

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        if hovered:
            pygame.draw.rect(button_surface, BUTTON_HOVER, button_surface.get_rect(), border_radius=BUTTON_RADIUS)
        else:
            pygame.draw.rect(button_surface, BUTTON_IDLE, button_surface.get_rect(), border_radius=BUTTON_RADIUS)
            pygame.draw.rect(button_surface, BUTTON_OUTLINE, button_surface.get_rect(), width=1, border_radius=BUTTON_RADIUS)
        surface.blit(button_surface, self.rect.topleft)
        text_value = self.get_text()
        text_surf = font.render(text_value, True, TEXT_WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._callback()    # Call the button's callback function


# =======================
#   DISPLAY SETUP
# =======================
def _set_display_mode_with_vsync(size: tuple[int, int], flags: int = 0) -> pygame.Surface:
    """Create the display surface with double buffering and vsync when available."""
    flags |= DOUBLEBUF
    try:
        return pygame.display.set_mode(size, flags, vsync=1)
    except TypeError:
        return pygame.display.set_mode(size, flags)
    except pygame.error as err:
        try:
            return pygame.display.set_mode(size, flags)
        except pygame.error:
            raise err


# =======================
#   MAIN MENU
# =======================
def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines


def run_main_menu(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    initial_planet: Planet,
    initial_scenario: Scenario | None = None,
) -> tuple[Planet, Scenario] | None:
    """
    Display main menu with planet and scenario selection.
    Returns (selected_planet, selected_scenario) when user clicks Start.
    Returns None if user wants to quit.
    """
    global integrator, WIDTH, HEIGHT, PIXELS_PER_METER
    
    # Fonts
    font = pygame.font.SysFont("consolas", 16)
    font_small = pygame.font.SysFont("consolas", 14)
    font_fps = pygame.font.SysFont("consolas", 14)
    title_font = pygame.font.SysFont("arial", 52, bold=True)
    section_font = pygame.font.SysFont("consolas", 15, bold=True)
    briefing_title_font = pygame.font.SysFont("arial", 22, bold=True)
    briefing_font = pygame.font.SysFont("consolas", 15)
    
    # Create starfield for background with animated drift
    starfield = Starfield(num_stars=200, seed=42)
    menu_start_time = time.perf_counter()
    
    # State
    selected_planet = initial_planet
    scenarios = list(get_scenarios_for_planet(selected_planet))
    if initial_scenario and initial_scenario in scenarios:
        selected_scenario = initial_scenario
    else:
        selected_scenario = scenarios[0] if scenarios else None
    
    # Button dimensions
    planet_btn_height = 32
    scenario_btn_height = 32
    start_btn_width = 220
    start_btn_height = 46
    btn_gap = 6
    
    def update_scenarios():
        nonlocal scenarios, selected_scenario
        scenarios = list(get_scenarios_for_planet(selected_planet))
        selected_scenario = scenarios[0] if scenarios else None
    
    while True:
        current_size = screen.get_size()
        WIDTH, HEIGHT = current_size
        
        mouse_pos = pygame.mouse.get_pos()
        
        # =====================
        # Layout Calculations
        # =====================
        panel_margin = 30
        panel_top = 110
        footer_height = 75
        panel_bottom = HEIGHT - footer_height - 15
        
        panel_rect = pygame.Rect(
            panel_margin,
            panel_top,
            WIDTH - 2 * panel_margin,
            panel_bottom - panel_top
        )
        
        # Column widths (25%, 35%, 40%)
        col1_width = int(panel_rect.width * 0.25)
        col2_width = int(panel_rect.width * 0.35)
        col3_width = panel_rect.width - col1_width - col2_width
        
        col1_x = panel_rect.left
        col2_x = col1_x + col1_width
        col3_x = col2_x + col2_width
        content_top = panel_rect.top + 20
        col_padding = 15
        
        # Button widths based on column widths
        planet_btn_width = col1_width - 2 * col_padding
        scenario_btn_width = col2_width - 2 * col_padding
        
        # =====================
        # Event Handling
        # =====================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    if selected_scenario:
                        return (selected_planet, selected_scenario)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check planet buttons (Column 1)
                for i, preset_name in enumerate(PLANET_PRESETS):
                    btn_x = col1_x + col_padding
                    btn_y = content_top + 35 + i * (planet_btn_height + btn_gap)
                    btn_rect = pygame.Rect(btn_x, btn_y, planet_btn_width, planet_btn_height)
                    if btn_rect.collidepoint(event.pos):
                        selected_planet = get_preset(preset_name)
                        update_scenarios()
                        break
                
                # Check scenario buttons (Column 2)
                for i, scenario in enumerate(scenarios):
                    btn_x = col2_x + col_padding
                    btn_y = content_top + 35 + i * (scenario_btn_height + btn_gap)
                    btn_rect = pygame.Rect(btn_x, btn_y, scenario_btn_width, scenario_btn_height)
                    if btn_rect.collidepoint(event.pos):
                        selected_scenario = scenario
                        break
                
                # Check start button (bottom of right column)
                start_btn_x = col3_x + col_padding
                start_btn_y = panel_rect.bottom - start_btn_height - 15
                start_rect = pygame.Rect(start_btn_x, start_btn_y, col3_width - 2 * col_padding, start_btn_height)
                if start_rect.collidepoint(event.pos) and selected_scenario:
                    return (selected_planet, selected_scenario)
                
                # Check integrator button (Footer)
                integrator_rect = pygame.Rect(panel_margin + 10, HEIGHT - footer_height + 20, 140, 32)
                if integrator_rect.collidepoint(event.pos):
                    integrator = "Euler" if integrator == "RK4" else "RK4"
        
        # =====================
        # Render
           # =====================
        screen.fill(BACKGROUND_COLOR)
        
        # Draw starfield with gentle animated drift
        elapsed = time.perf_counter() - menu_start_time
        # Slow diagonal drift - creates a peaceful space ambiance
        drift_x = elapsed * 5000000  # Horizontal drift speed
        drift_y = elapsed * 3000000  # Vertical drift speed
        starfield.draw(screen, drift_x, drift_y)
        
        # Title with shadow
        title_text = "ORBITLAB"
        # Shadow (offset by 2 pixels)
        shadow_surf = title_font.render(title_text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 2, 50 + 2))
        screen.blit(shadow_surf, shadow_rect)
        # Main title
        title_surf = title_font.render(title_text, True, TEXT_WHITE)
        title_rect = title_surf.get_rect(center=(WIDTH // 2, 50))
        screen.blit(title_surf, title_rect)
        
        # Subtitle
        subtitle_text = "av Axel Jönsson"
        subtitle_surf = font_small.render(subtitle_text, True, MENU_SUBTITLE_COLOR)
        subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, 90))
        screen.blit(subtitle_surf, subtitle_rect)
        
        # =====================
        # Main Container Panel
        # =====================
        panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, MENU_PANEL_COLOR, panel_surface.get_rect(), border_radius=16)
        screen.blit(panel_surface, panel_rect.topleft)
        
        # Column dividers
        divider1_x = col2_x
        divider2_x = col3_x
        pygame.draw.line(screen, MENU_COLUMN_DIVIDER_COLOR,
                         (divider1_x, panel_rect.top + 15),
                         (divider1_x, panel_rect.bottom - 15), 1)
        pygame.draw.line(screen, MENU_COLUMN_DIVIDER_COLOR,
                         (divider2_x, panel_rect.top + 15),
                         (divider2_x, panel_rect.bottom - 15), 1)
        
        # =====================
        # Column 1: Celestial Body Selection
        # =====================
        planet_title = section_font.render("Celestial Body", True, TEXT_GRAY)
        screen.blit(planet_title, (col1_x + col_padding, content_top))
        
        for i, preset_name in enumerate(PLANET_PRESETS):
            btn_x = col1_x + col_padding
            btn_y = content_top + 35 + i * (planet_btn_height + btn_gap)
            btn_rect = pygame.Rect(btn_x, btn_y, planet_btn_width, planet_btn_height)
            
            is_selected = selected_planet.name.lower() == preset_name.lower()
            is_hovered = btn_rect.collidepoint(mouse_pos)
            
            btn_surface = pygame.Surface(btn_rect.size, pygame.SRCALPHA)
            if is_selected:
                # Solid fill for selected button
                pygame.draw.rect(btn_surface, BUTTON_SELECTED, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
            elif is_hovered:
                # Brighter fill on hover
                pygame.draw.rect(btn_surface, BUTTON_HOVER, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
            else:
                # Semi-transparent fill with outline for idle
                pygame.draw.rect(btn_surface, BUTTON_IDLE, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
                pygame.draw.rect(btn_surface, BUTTON_OUTLINE, btn_surface.get_rect(), width=1, border_radius=BUTTON_RADIUS)
            screen.blit(btn_surface, btn_rect.topleft)
            
            text_surf = font.render(preset_name.capitalize(), True, TEXT_WHITE)
            text_rect = text_surf.get_rect(center=btn_rect.center)
            screen.blit(text_surf, text_rect)
        
        # =====================
        # Column 2: Scenario Parameters
        # =====================
        scenario_title = section_font.render("Scenario", True, TEXT_GRAY)
        screen.blit(scenario_title, (col2_x + col_padding, content_top))
        
        for i, scenario in enumerate(scenarios):
            btn_x = col2_x + col_padding
            btn_y = content_top + 35 + i * (scenario_btn_height + btn_gap)
            btn_rect = pygame.Rect(btn_x, btn_y, scenario_btn_width, scenario_btn_height)
            
            is_selected = selected_scenario and scenario.key == selected_scenario.key
            is_hovered = btn_rect.collidepoint(mouse_pos)
            
            btn_surface = pygame.Surface(btn_rect.size, pygame.SRCALPHA)
            if is_selected:
                # Solid fill for selected button
                pygame.draw.rect(btn_surface, BUTTON_SELECTED, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
            elif is_hovered:
                # Brighter fill on hover
                pygame.draw.rect(btn_surface, BUTTON_HOVER, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
            else:
                # Semi-transparent fill with outline for idle
                pygame.draw.rect(btn_surface, BUTTON_IDLE, btn_surface.get_rect(), border_radius=BUTTON_RADIUS)
                pygame.draw.rect(btn_surface, BUTTON_OUTLINE, btn_surface.get_rect(), width=1, border_radius=BUTTON_RADIUS)
            screen.blit(btn_surface, btn_rect.topleft)
            
            text_surf = font.render(scenario.name, True, TEXT_WHITE)
            text_rect = text_surf.get_rect(center=btn_rect.center)
            screen.blit(text_surf, text_rect)
        
        # =====================
        # Column 3: Mission Briefing
        # =====================
        briefing_title = section_font.render("Mission Briefing", True, TEXT_GRAY)
        screen.blit(briefing_title, (col3_x + col_padding, content_top))
        
        if selected_scenario:
            briefing_y = content_top + 40
            
            # Scenario name (larger)
            name_surf = briefing_title_font.render(selected_scenario.name, True, TEXT_WHITE)
            screen.blit(name_surf, (col3_x + col_padding, briefing_y))
            briefing_y += 35
            
            # Description (word-wrapped)
            wrap_width = col3_width - 2 * col_padding
            desc_lines = wrap_text(selected_scenario.description, briefing_font, wrap_width)
            for line in desc_lines:
                line_surf = briefing_font.render(line, True, MENU_BRIEFING_TEXT_COLOR)
                screen.blit(line_surf, (col3_x + col_padding, briefing_y))
                briefing_y += briefing_font.get_linesize() + 2
            
            briefing_y += 15
            
            # Parameters section
            params_title = section_font.render("Parameters", True, TEXT_GRAY)
            screen.blit(params_title, (col3_x + col_padding, briefing_y))
            briefing_y += 25
            
            # Calculate altitude
            altitude_km = (selected_scenario.r0 - selected_planet.radius) / 1000
            velocity_ms = abs(selected_scenario.velocity)
            direction = "Prograde" if selected_scenario.velocity > 0 else "Retrograde"
            
            # Parameters with gray labels and white values (two-column layout)
            params = [
                ("Planet:", selected_planet.name),
                ("Starting Altitude:", f"{altitude_km:,.0f} km"),
                ("Initial Velocity:", f"{velocity_ms:,.0f} m/s"),
                ("Direction:", direction),
            ]
            
            # Fixed offset for value column alignment (ensures numbers line up)
            value_x_offset = 145
            
            for label, value in params:
                # Column 1: Label (left-aligned)
                label_surf = briefing_font.render(label, True, TEXT_GRAY)
                screen.blit(label_surf, (col3_x + col_padding, briefing_y))
                # Column 2: Value (at fixed offset)
                value_surf = briefing_font.render(value, True, TEXT_WHITE)
                screen.blit(value_surf, (col3_x + col_padding + value_x_offset, briefing_y))
                briefing_y += briefing_font.get_linesize() + 6
            
            briefing_y += 15
            
            # Planet Preview Area
            preview_height = min(350, panel_rect.bottom - briefing_y - start_btn_height - 50)
            preview_height = max(150, preview_height)  # Minimum size
            preview_x = col3_x + col_padding
            preview_y = briefing_y
            preview_rect = pygame.Rect(preview_x, preview_y, col3_width - 2 * col_padding, preview_height)
            
            # Draw preview background (no outline - cleaner look)
            preview_surface = pygame.Surface(preview_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(preview_surface, PREVIEW_PLACEHOLDER, preview_surface.get_rect(), border_radius=8)
            screen.blit(preview_surface, preview_rect.topleft)
            
            # Generate and draw rotating planet sprite
            planet_sprite_size = min(preview_rect.width, preview_rect.height) - 50
            planet_sprite = get_planet_sprite(selected_planet, planet_sprite_size)
            
            # Slow rotation animation
            rotation_angle = (pygame.time.get_ticks() / 80) % 360
            rotated_sprite = pygame.transform.rotate(planet_sprite, rotation_angle)
            
            # Center the rotated sprite (slightly above center to leave room for label)
            sprite_center_y = preview_rect.centery - 10
            sprite_rect = rotated_sprite.get_rect(center=(preview_rect.centerx, sprite_center_y))
            screen.blit(rotated_sprite, sprite_rect)
            
            # Planet name label at fixed position (below planet area)
            planet_label = font_small.render(selected_planet.name, True, TEXT_GRAY)
            label_y = sprite_center_y + planet_sprite_size // 2 + 12  # Fixed position based on original sprite size
            planet_label_rect = planet_label.get_rect(centerx=preview_rect.centerx, top=label_y)
            screen.blit(planet_label, planet_label_rect)
        
        # Start button (bottom of right column)
        start_btn_x = col3_x + col_padding
        start_btn_y = panel_rect.bottom - start_btn_height - 15
        start_rect = pygame.Rect(start_btn_x, start_btn_y, col3_width - 2 * col_padding, start_btn_height)
        is_hovered = start_rect.collidepoint(mouse_pos)
        
        start_surface = pygame.Surface(start_rect.size, pygame.SRCALPHA)
        if is_hovered:
            pygame.draw.rect(start_surface, BUTTON_START_HOVER, start_surface.get_rect(), border_radius=BUTTON_RADIUS)
        else:
            pygame.draw.rect(start_surface, BUTTON_START, start_surface.get_rect(), border_radius=BUTTON_RADIUS)
        screen.blit(start_surface, start_rect.topleft)
        
        start_text = font.render("Start Simulation", True, TEXT_WHITE)
        start_text_rect = start_text.get_rect(center=start_rect.center)
        screen.blit(start_text, start_text_rect)
        
        # =====================
        # Footer Bar
        # =====================
        footer_rect = pygame.Rect(0, HEIGHT - footer_height, WIDTH, footer_height)
        footer_surface = pygame.Surface(footer_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(footer_surface, MENU_FOOTER_COLOR, footer_surface.get_rect())
        screen.blit(footer_surface, footer_rect.topleft)
        
        # Integrator toggle button (footer left)
        integrator_rect = pygame.Rect(panel_margin + 10, HEIGHT - footer_height + 20, 140, 32)
        is_hovered = integrator_rect.collidepoint(mouse_pos)
        
        int_surface = pygame.Surface(integrator_rect.size, pygame.SRCALPHA)
        if is_hovered:
            pygame.draw.rect(int_surface, BUTTON_HOVER, int_surface.get_rect(), border_radius=BUTTON_RADIUS)
        else:
            pygame.draw.rect(int_surface, BUTTON_IDLE, int_surface.get_rect(), border_radius=BUTTON_RADIUS)
            pygame.draw.rect(int_surface, BUTTON_OUTLINE, int_surface.get_rect(), width=1, border_radius=BUTTON_RADIUS)
        screen.blit(int_surface, integrator_rect.topleft)
        
        int_text = font_small.render(f"Integrator: {integrator}", True, TEXT_WHITE)
        int_text_rect = int_text.get_rect(center=integrator_rect.center)
        screen.blit(int_text, int_text_rect)
        
        # Instructions (centered in footer)
        inst_text = "Press SPACE or ENTER to start  |  ESC to quit"
        inst_surf = font_small.render(inst_text, True, TEXT_GRAY)
        inst_rect = inst_surf.get_rect(center=(WIDTH // 2, HEIGHT - footer_height // 2))
        screen.blit(inst_surf, inst_rect)
        
        # FPS (bottom right corner)
        fps_value = clock.get_fps()
        fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, (140, 180, 220))
        fps_rect = fps_text.get_rect(bottomright=(WIDTH - 10, HEIGHT - 3))
        screen.blit(fps_text, fps_rect)
        
        pygame.display.flip()
        clock.tick(60)


# =======================
#   MAIN FUNCTION
# =======================
def main():
    pygame.init()
    pygame.display.set_caption("OrbitLab – Interactive Orbit Simulator")
    
    font_fps = pygame.font.SysFont("consolas", 14)

    fullscreen_flags = FULLSCREEN | DOUBLEBUF
    borderless_flags = NOFRAME | DOUBLEBUF
    windowed_flags = RESIZABLE | DOUBLEBUF

    def create_fullscreen_surface() -> pygame.Surface:
        info = pygame.display.Info()
        fallback_resolution = (info.current_w or WIDTH, info.current_h or HEIGHT)
        screen_fs: pygame.Surface | None = None
        if info.current_w and info.current_h:
            try:
                os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
                screen_fs = _set_display_mode_with_vsync(
                    (info.current_w, info.current_h),
                    borderless_flags,
                )
            except pygame.error:
                screen_fs = None
        if screen_fs is None:
            try:
                screen_fs = _set_display_mode_with_vsync((0, 0), fullscreen_flags)
            except pygame.error:
                screen_fs = _set_display_mode_with_vsync(fallback_resolution, DOUBLEBUF)
        return screen_fs

    def create_windowed_surface() -> pygame.Surface:
        return _set_display_mode_with_vsync(WINDOWED_DEFAULT_SIZE, windowed_flags)

    fullscreen_enabled = FULLSCREEN_ENABLED
    try:
        if fullscreen_enabled:
            screen = create_fullscreen_surface()
        else:
            screen = create_windowed_surface()
            fullscreen_enabled = False
    except pygame.error:
        screen = _set_display_mode_with_vsync(WINDOWED_DEFAULT_SIZE, windowed_flags)
        fullscreen_enabled = False

    current_size = screen.get_size()
    update_display_metrics(*current_size)

    overlay_size = current_size
    orbit_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
    label_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
    grid_surface = pygame.Surface(overlay_size, pygame.SRCALPHA)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    scenario_font = pygame.font.SysFont("consolas", 16)
    grid_label_font = pygame.font.SysFont("consolas", 14)
    grid_axis_font = pygame.font.SysFont("consolas", 16)
    title_font = pygame.font.SysFont("arial", 48, bold=True)
    
    # Create starfield for game loop background (200 stars with parallax effect)
    game_starfield = Starfield(num_stars=200, seed=42)

    # Mutable state containers that will be updated from menu
    scenarios_list: list[Scenario] = []
    scenarios_dict: dict[str, Scenario] = {}
    scenario_order: list[str] = []
    active_scenario_key: str = ""
    
    scenario_flash_text: str | None = None
    scenario_flash_time = 0.0

    def get_current_scenario() -> Scenario:
        return scenarios_dict[active_scenario_key]

    def get_current_r0() -> np.ndarray:
        """Get starting position vector from current scenario."""
        return get_current_scenario().position_vector()

    def get_current_velocity() -> np.ndarray:
        """Get starting velocity vector from current scenario."""
        return get_current_scenario().velocity_vector()

    # Simulation state - will be reset when entering simulation
    r = np.array([0.0, 0.0], dtype=float)
    v = np.array([0.0, 0.0], dtype=float)
    t_sim = 0.0
    paused = False
    ppm = PIXELS_PER_METER
    ppm_target = ppm
    real_time_speed = REAL_TIME_SPEED
    grid_overlay_enabled = False
    show_velocity_arrow = True
    camera_mode = "earth"
    camera_center = np.array([0.0, 0.0], dtype=float)
    camera_target = np.array([0.0, 0.0], dtype=float)
    is_dragging_camera = False
    drag_last_pos = (0, 0)

    orbit_prediction_period: float | None = None
    orbit_prediction_points: list[tuple[float, float]] = []

    orbit_markers: deque[tuple[str, float, float, float]] = deque(maxlen=20)
    impact_info: dict[str, float] | None = None
    shock_ring_start: float | None = None
    impact_freeze_time: float | None = None
    impact_overlay_reveal_time: float | None = None
    impact_overlay_visible_since: float | None = None

    accumulator = 0.0
    last_time = time.perf_counter()

    # Logging state
    logger: RunLogger | None = None
    log_step_counter = 0
    prev_r: float | None = None
    prev_dr: float | None = None
    impact_logged = False
    escape_logged = False

    def close_logger():
        nonlocal logger
        if logger is not None:
            logger.close()
            logger = None

    def reset():
        nonlocal r, v, t_sim, paused, ppm, real_time_speed
        nonlocal accumulator, last_time, log_step_counter, prev_r, prev_dr
        nonlocal impact_logged, escape_logged, ppm_target
        nonlocal camera_center, orbit_markers, camera_target
        nonlocal camera_mode, is_dragging_camera
        nonlocal orbit_prediction_period, orbit_prediction_points
        nonlocal impact_info, shock_ring_start, impact_freeze_time
        nonlocal impact_overlay_reveal_time, impact_overlay_visible_since
        nonlocal show_velocity_arrow
        close_logger()
        r = get_current_r0()
        v = get_current_velocity()
        t_sim = 0.0
        paused = False
        ppm_target = clamp(ppm_target, MIN_PPM, MAX_PPM)
        ppm = clamp(ppm_target, MIN_PPM, MAX_PPM)
        real_time_speed = clamp(real_time_speed, 0.1, 10_000.0)
        accumulator = 0.0
        last_time = time.perf_counter()
        log_step_counter = 0
        prev_r = None
        prev_dr = None
        impact_logged = False
        escape_logged = False
        orbit_markers.clear()
        camera_target[:] = camera_center
        is_dragging_camera = False
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(r, v, current_planet.mu)
        impact_info = None
        shock_ring_start = None
        impact_freeze_time = None
        impact_overlay_reveal_time = None
        impact_overlay_visible_since = None

    

    state = "running"  # "running", "impact" (menu is now handled separately)
    return_to_menu = False  # Flag to signal return to main menu
    escape_radius_limit = ESCAPE_RADIUS_FACTOR * get_default_r0()
    
    def go_to_menu() -> None:
        """Signal that we want to return to the main menu."""
        nonlocal return_to_menu
        return_to_menu = True

    def marker_display_name(marker_type: str) -> str:
        return "Periapsis" if marker_type == "pericenter" else "Apoapsis"

    def get_latest_marker(marker_type: str) -> tuple[float, float, float] | None:
        for m_type, mx, my, mr in reversed(orbit_markers):
            if m_type == marker_type:
                return (mx, my, mr)
        return None

    def marker_hit_test(mouse_pos: tuple[int, int]) -> str | None:
        if not orbit_markers:
            return None
        camera_tuple = (float(camera_center[0]), float(camera_center[1]))
        closest_type: str | None = None
        closest_distance = float("inf")
        for marker_type, mx, my, _ in reversed(orbit_markers):
            marker_pos = world_to_screen(mx, my, ppm, camera_tuple)
            distance = math.hypot(marker_pos[0] - mouse_pos[0], marker_pos[1] - mouse_pos[1])
            if distance <= LABEL_MARKER_HOVER_RADIUS and distance < closest_distance:
                closest_distance = distance
                closest_type = marker_type
        return closest_type

    def log_state(dt_eff: float) -> None:
        if logger is None:
            return
        rmag = float(np.linalg.norm(r))
        vmag = float(np.linalg.norm(v))
        eps = float(energy_specific(r, v, current_planet.mu))
        h_vec = np.cross(np.array([r[0], r[1], 0.0]), np.array([v[0], v[1], 0.0]))
        h_mag = float(np.linalg.norm(h_vec))
        e_val = float(eccentricity(r, v, current_planet.mu))
        logger.log_ts(
            [
                float(t_sim),
                float(r[0]),
                float(r[1]),
                float(v[0]),
                float(v[1]),
                rmag,
                vmag,
                eps,
                h_mag,
                e_val,
                float(dt_eff),
            ]
        )

    def init_run_logging() -> None:
        nonlocal logger, log_step_counter, prev_r, prev_dr, impact_logged, escape_logged
        close_logger()
        logger = RunLogger()
        scenario = get_current_scenario()
        r0_vector = scenario.position_vector()
        v0_vector = scenario.velocity_vector()
        meta = {
            "scenario_key": scenario.key,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "planet_name": current_planet.name,
            "planet_radius": current_planet.radius,
            "R0": r0_vector.tolist(),
            "V0": v0_vector.tolist(),
            "v0": float(np.linalg.norm(v0_vector)),
            "G": G,
            "M": current_planet.mass,
            "mu": current_planet.mu,
            "integrator": "RK4",
            "dt_phys": DT_PHYS,
            "start_speed": REAL_TIME_SPEED,
            "log_strategy": "every_20_steps",
            "code_version": "OrbitLab v1.5",
        }
        logger.write_meta(meta)
        log_step_counter = 0
        prev_r = float(np.linalg.norm(r))
        prev_dr = None
        impact_logged = False
        escape_logged = False
        log_state(0.0)

    def quit_app():
        close_logger()
        pygame.quit()
        sys.exit()

    def render_marker_label(
        surface: pygame.Surface,
        marker_type: str,
        marker_pos: tuple[int, int],
        radius_world: float,
    ) -> None:
        label = marker_display_name(marker_type)
        altitude_km = (radius_world - current_planet.radius) / 1_000.0
        text = f"{label}: {altitude_km:,.1f} km"
        text_surf = font.render(text, True, LABEL_TEXT_COLOR)

        padding = 6
        bg_width = text_surf.get_width() + padding * 2
        bg_height = text_surf.get_height() + padding * 2

        connector_alpha = int(LABEL_MARKER_ALPHA * 0.6)
        connector_color = LABEL_MARKER_COLOR
        line_length = 18
        direction = -1 if marker_type == "pericenter" else 1
        anchor_y = marker_pos[1] + direction * line_length

        bg_rect = pygame.Rect(0, 0, bg_width, bg_height)
        bg_rect.center = (
            marker_pos[0],
            int(anchor_y + direction * (bg_rect.height / 2 + 6)),
        )

        pygame.draw.line(
            surface,
            (*connector_color, connector_alpha),
            marker_pos,
            (marker_pos[0], anchor_y),
            2,
        )

        pygame.draw.rect(
            surface,
            LABEL_BACKGROUND_COLOR,
            bg_rect,
            border_radius=10,
        )

        surface.blit(text_surf, (bg_rect.left + padding, bg_rect.top + padding))

    button_width = 140
    button_height = 40
    button_gap = 10

    def toggle_pause():
        nonlocal paused
        paused = not paused

    def slow_down():
        nonlocal real_time_speed
        real_time_speed = max(real_time_speed / 1.5, 0.1)

    def speed_up():
        nonlocal real_time_speed
        real_time_speed = min(real_time_speed * 1.5, 10_000.0)

    def toggle_camera():
        nonlocal camera_mode
        if camera_mode == "earth":
            camera_mode = "satellite"
        elif camera_mode == "satellite":
            camera_mode = "manual"
            # Sync target to prevent drift when switching to manual
            camera_target[:] = camera_center
        else:
            camera_mode = "earth"

    def reset_and_continue():
        nonlocal state, paused, impact_info, shock_ring_start
        reset()
        state = "running"
        paused = False
        impact_info = None
        shock_ring_start = None
        init_run_logging()

    sim_buttons: list[Button] = []

    def update_sim_button_layout() -> None:
        if not sim_buttons:
            if state == "running":
                for btn in sim_buttons:
                    btn.handle_event(event)
            return
        total_buttons = len(sim_buttons)
        total_height = total_buttons * button_height + (total_buttons - 1) * button_gap
        start_y = int((HEIGHT - total_height) / 2)
        x_pos = WIDTH - button_width - 20
        for idx, btn in enumerate(sim_buttons):
            top = start_y + idx * (button_height + button_gap)
            btn.rect.update(x_pos, top, button_width, button_height)

    sim_buttons = [
        Button((20, 20, button_width, button_height), "Pause", toggle_pause, lambda: "Resume" if paused else "Pause"),
        Button((20, 20 + (button_height + button_gap), button_width, button_height), "Reset", reset_and_continue),
        Button((20, 20 + 2 * (button_height + button_gap), button_width, button_height), "Slower", slow_down),
        Button((20, 20 + 3 * (button_height + button_gap), button_width, button_height), "Faster", speed_up),
        Button(
            (20, 20 + 4 * (button_height + button_gap), button_width, button_height),
            "Camera",
            toggle_camera,
            lambda: (
                "Camera: Earth"
                if camera_mode == "earth"
                else "Camera: Sat" if camera_mode == "satellite" else "Camera: Free"
            ),
        ),
        Button((20, 20 + 5 * (button_height + button_gap), button_width, button_height), "Menu", go_to_menu),
    ]

    update_sim_button_layout()

    def toggle_fullscreen_mode() -> None:
        nonlocal screen, fullscreen_enabled, overlay_size
        target_fullscreen = not fullscreen_enabled
        try:
            screen = (
                create_fullscreen_surface()
                if target_fullscreen
                else create_windowed_surface()
            )
        except pygame.error:
            return
        fullscreen_enabled = target_fullscreen
        overlay_size = (-1, -1)

    def is_over_button(pos: tuple[int, int]) -> bool:
        if state == "running":
            return any(btn.rect.collidepoint(pos) for btn in sim_buttons)
        return False

    def setup_simulation_from_selection(selected_planet: Planet, selected_scenario: Scenario) -> None:
        """Initialize simulation state based on menu selection."""
        global current_planet, current_planet_index, PIXELS_PER_METER
        nonlocal scenarios_list, scenarios_dict, scenario_order, active_scenario_key
        nonlocal r, v, t_sim, paused, ppm, ppm_target, real_time_speed
        nonlocal grid_overlay_enabled, show_velocity_arrow, camera_mode
        nonlocal camera_center, camera_target, is_dragging_camera
        nonlocal orbit_prediction_period, orbit_prediction_points
        nonlocal orbit_markers, impact_info, shock_ring_start
        nonlocal impact_freeze_time, impact_overlay_reveal_time, impact_overlay_visible_since
        nonlocal accumulator, last_time, state, return_to_menu, escape_radius_limit
        
        # Update planet
        current_planet = selected_planet
        try:
            current_planet_index = PLANET_PRESETS.index(selected_planet.name.lower())
        except ValueError:
            current_planet_index = 0
        
        # Update scenarios
        scenarios_list = list(get_scenarios_for_planet(current_planet))
        scenarios_dict = {s.key: s for s in scenarios_list}
        scenario_order = [s.key for s in scenarios_list]
        active_scenario_key = selected_scenario.key
        
        # Update display metrics
        update_display_metrics(*screen.get_size())
        PIXELS_PER_METER = compute_pixels_per_meter(WIDTH, HEIGHT)
        ppm = PIXELS_PER_METER
        ppm_target = ppm
        
        # Reset simulation state
        r = get_current_r0()
        v = get_current_velocity()
        t_sim = 0.0
        paused = False
        real_time_speed = REAL_TIME_SPEED
        grid_overlay_enabled = False
        show_velocity_arrow = True
        camera_mode = "earth"
        camera_center = np.array([0.0, 0.0], dtype=float)
        camera_target = np.array([0.0, 0.0], dtype=float)
        is_dragging_camera = False
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(r, v, current_planet.mu)
        orbit_markers.clear()
        impact_info = None
        shock_ring_start = None
        impact_freeze_time = None
        impact_overlay_reveal_time = None
        impact_overlay_visible_since = None
        accumulator = 0.0
        last_time = time.perf_counter()
        state = "running"
        return_to_menu = False
        escape_radius_limit = ESCAPE_RADIUS_FACTOR * get_default_r0()

    # ========= OUTER LOOP: MENU -> SIMULATION -> MENU =========
    while True:
        # Show main menu and get selection
        menu_result = run_main_menu(screen, clock, current_planet)
        
        if menu_result is None:
            # User quit from menu
            pygame.quit()
            sys.exit()
        
        selected_planet, selected_scenario = menu_result
        
        # Setup simulation based on selection
        setup_simulation_from_selection(selected_planet, selected_scenario)
        
        # Initialize logging for the new run
        init_run_logging()

        # ========= SIMULATION LOOP =========
        while True:
            current_size = screen.get_size()
            if current_size != overlay_size:
                overlay_size = current_size
                update_display_metrics(*current_size)
                update_sim_button_layout()
                orbit_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
                label_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
                grid_surface = pygame.Surface(overlay_size, pygame.SRCALPHA)

            orbit_layer.fill((0, 0, 0, 0))
            label_layer.fill((0, 0, 0, 0))
            if grid_overlay_enabled:
                grid_surface.fill((0, 0, 0, 0))

            # --- Input ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_app()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        go_to_menu()
                        continue
                    if event.key == pygame.K_F11:
                        toggle_fullscreen_mode()
                        continue
                    if event.key == pygame.K_g:
                        grid_overlay_enabled = not grid_overlay_enabled
                        continue
                        
                    if state == "impact":
                        if event.key == pygame.K_r:
                            reset_and_continue()
                        continue
                        
                    if state == "running":
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            reset_and_continue()
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                            ppm_target = clamp(ppm_target * 1.2, MIN_PPM, MAX_PPM)
                        elif event.key == pygame.K_MINUS:
                            ppm_target = clamp(ppm_target / 1.2, MIN_PPM, MAX_PPM)
                        elif event.key == pygame.K_RIGHT:
                            real_time_speed = min(real_time_speed * 1.5, 10_000.0)
                        elif event.key == pygame.K_LEFT:
                            real_time_speed = max(real_time_speed / 1.5, 0.1)
                        elif event.key == pygame.K_UP:
                            real_time_speed = min(real_time_speed * 2.0, 10_000.0)
                        elif event.key == pygame.K_DOWN:
                            real_time_speed = max(real_time_speed / 2.0, 0.1)
                        elif event.key == pygame.K_c:
                            toggle_camera()
                        elif event.key == pygame.K_v:
                            show_velocity_arrow = not show_velocity_arrow
                            
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and state == "running":
                        if not is_over_button(event.pos):
                            is_dragging_camera = True
                            drag_last_pos = event.pos
                            camera_mode = "manual"
                            camera_target[:] = camera_center
                    elif event.button == 4 and state == "running":
                        ppm_target = clamp(ppm_target * 1.1, MIN_PPM, MAX_PPM)
                    elif event.button == 5 and state == "running":
                        ppm_target = clamp(ppm_target / 1.1, MIN_PPM, MAX_PPM)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        is_dragging_camera = False
                elif event.type == pygame.MOUSEMOTION:
                    if is_dragging_camera and state == "running":
                        dx = event.pos[0] - drag_last_pos[0]
                        dy = event.pos[1] - drag_last_pos[1]
                        if dx != 0 or dy != 0:
                            camera_center[0] -= dx / ppm
                            camera_center[1] += dy / ppm
                            camera_target[:] = camera_center
                            drag_last_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    if state == "running" and event.y != 0:
                        zoom_factor = 1.1 ** event.y
                        new_ppm_target = clamp(ppm_target * zoom_factor, MIN_PPM, MAX_PPM)

                        if camera_mode == "manual":
                            # Mouse-centered zoom logic
                            mx, my = pygame.mouse.get_pos()
                            w_half = WIDTH / 2.0
                            h_half = HEIGHT / 2.0
                            
                            # 1. Calculate World position under mouse using CURRENT ppm and center
                            # Formula derived from world_to_screen inversion
                            m_wx = camera_center[0] + (mx - w_half) / ppm
                            m_wy = camera_center[1] + (h_half - my) / ppm

                            # 2. Calculate where the camera target MUST be for that World position 
                            # to project back to the same mouse screen coordinates at the NEW ppm scale.
                            # mx = w_half + (m_wx - target_cx) * new_ppm_target
                            target_cx = m_wx - (mx - w_half) / new_ppm_target
                            
                            # my = h_half - (m_wy - target_cy) * new_ppm_target (inverted Y)
                            target_cy = m_wy - (h_half - my) / new_ppm_target

                            camera_target[:] = (target_cx, target_cy)
                        
                        ppm_target = new_ppm_target
                
                if state == "running":
                    for btn in sim_buttons:
                        btn.handle_event(event)

            # Check if we should return to menu
            if return_to_menu:
                close_logger()
                break

            # === PHYSICS UPDATE ===
            now = time.perf_counter()
            frame_dt_real = now - last_time
            last_time = now
            sim_dt_target = frame_dt_real * real_time_speed
            accumulator += sim_dt_target

            if paused:
                accumulator = 0.0

            time_to_simulate = accumulator if not paused else 0.0
            if time_to_simulate > 0.0:
                steps_needed = max(1, math.ceil(time_to_simulate / DT_PHYS))
                steps_to_run = min(steps_needed, MAX_SUBSTEPS)
                dt_step = time_to_simulate / steps_to_run

                for _ in range(steps_to_run):
                    if integrator == "RK4":
                        r, v = rk4_step(r, v, dt_step, current_planet.mu)
                    elif integrator == "Euler":
                        r, v = euler_step(r, v, dt_step, current_planet.mu)
                    t_sim += dt_step
                    rmag = float(np.linalg.norm(r))
                    
                    event_type: str | None = None
                    prev_radius = prev_r
                    dr = None
                    if prev_radius is not None:
                        dr = rmag - prev_radius
                        if prev_dr is not None:
                            if prev_dr < 0.0 and dr >= 0.0:
                                event_type = "pericenter"
                            elif prev_dr > 0.0 and dr <= 0.0:
                                event_type = "apocenter"
                    prev_dr = dr
                    prev_r = rmag

                    if event_type is not None:
                        orbit_markers.append((event_type, float(r[0]), float(r[1]), rmag))

                    vmag = float(np.linalg.norm(v))
                    eps = float(energy_specific(r, v, current_planet.mu))
                    e_val = float(eccentricity(r, v, current_planet.mu))
                    impact_triggered = not impact_logged and rmag <= current_planet.radius

                    if logger is not None:
                        log_step_counter += 1

                        if event_type is not None:
                            import json
                            logger.log_event([
                                float(t_sim),
                                event_type,
                                rmag,
                                vmag,
                                json.dumps({"ecc": e_val, "energy": eps}),
                            ])

                        if impact_triggered:
                            import json
                            logger.log_event([
                                float(t_sim),
                                "impact",
                                rmag,
                                vmag,
                                json.dumps({"penetration": current_planet.radius - rmag, "energy": eps}),
                            ])

                        if not escape_logged and eps > 0.0 and rmag > escape_radius_limit:
                            import json
                            logger.log_event([
                                float(t_sim),
                                "escape",
                                rmag,
                                vmag,
                                json.dumps({"energy": eps, "ecc": e_val}),
                            ])
                            escape_logged = True

                        if log_step_counter >= LOG_EVERY_STEPS or event_type is not None or impact_triggered:
                            log_state(dt_step)
                            log_step_counter = 0

                    if impact_triggered:
                        impact_logged = True
                        if impact_info is None:
                            normal = r / max(rmag, 1e-9)
                            tangent = np.array([-normal[1], normal[0]])
                            tangential_speed = float(np.dot(v, tangent))
                            radial_speed = float(np.dot(v, normal))
                            angle_rad = math.atan2(abs(radial_speed), abs(tangential_speed))
                            impact_info = {
                                "time": float(t_sim),
                                "speed": vmag,
                                "angle": math.degrees(angle_rad),
                                "radial_speed": radial_speed,
                                "tangential_speed": tangential_speed,
                                "position": (float(r[0]), float(r[1])),
                            }
                            shock_ring_start = time.perf_counter()
                            impact_freeze_time = shock_ring_start + IMPACT_FREEZE_DELAY
                            impact_overlay_reveal_time = shock_ring_start + IMPACT_OVERLAY_DELAY
                            impact_overlay_visible_since = None
                            paused = True
                            state = "impact"
                            accumulator = 0.0
                            break

                accumulator = 0.0

            now_time = time.perf_counter()
            
            # Smooth zoom
            ppm += (ppm_target - ppm) * 0.1
            ppm = clamp(ppm, MIN_PPM, MAX_PPM)

            # Camera targets
            if camera_mode == "earth":
                view_offset = (HEIGHT * 0.08) / max(ppm, 1e-6)
                camera_target[:] = (0.0, view_offset)
            elif camera_mode == "satellite":
                camera_target[:] = (r[0], r[1])
            else:
                # Manual mode
                # Only lock target to center if dragging.
                # Otherwise, allow camera_target to differ (e.g. set by zoom logic)
                if is_dragging_camera:
                    camera_target[:] = camera_center
            
            camera_center += (camera_target - camera_center) * 0.1

            # === RENDER ===
            screen.fill(BACKGROUND_COLOR)
            
            # Calculate camera position tuple for rendering
            camera_center_tuple = (float(camera_center[0]), float(camera_center[1]))
            
            # Draw starfield background with parallax effect
            game_starfield.draw(screen, camera_center_tuple[0], camera_center_tuple[1])
            if grid_overlay_enabled:
                draw_coordinate_grid(
                    grid_surface,
                    ppm,
                    camera_center_tuple,
                    tick_font=grid_label_font,
                    axis_font=grid_axis_font,
                )
                screen.blit(grid_surface, (0, 0))

            mouse_pos = pygame.mouse.get_pos()
            rmag = float(np.linalg.norm(r))
            planet_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)

            # Draw predicted orbit
            if orbit_prediction_points:
                if orbit_prediction_period is None or orbit_prediction_period <= 0.0:
                    reveal_fraction = 1.0
                else:
                    reveal_fraction = clamp(t_sim / orbit_prediction_period, 0.0, 1.0)
                total_points = len(orbit_prediction_points)
                if reveal_fraction >= 1.0:
                    subset = orbit_prediction_points
                else:
                    max_index = max(2, int(total_points * reveal_fraction))
                    subset = orbit_prediction_points[:max_index]
                sampled_points = downsample_points(subset, MAX_RENDERED_ORBIT_POINTS)
                screen_points = [
                    world_to_screen(px, py, ppm, camera_center_tuple)
                    for px, py in sampled_points
                ]
                if len(screen_points) >= 2:
                    draw_orbit_line(orbit_layer, ORBIT_PRIMARY_COLOR, screen_points, ORBIT_LINE_WIDTH)
                    screen.blit(orbit_layer, (0, 0))

            planet_radius_px = max(1, int(current_planet.radius * ppm))
            
            # Draw atmosphere glow behind the planet
            draw_planet_atmosphere(screen, planet_screen_pos, planet_radius_px)
            
            # Draw the planet
            draw_planet(screen, current_planet, planet_screen_pos, planet_radius_px)
            
            # Draw night-side shadow for 3D effect (shadow on left side)
            draw_planet_night_side(screen, planet_screen_pos, planet_radius_px, shadow_side="left")

            # Shock ring on impact
            if impact_info is not None and shock_ring_start is not None:
                elapsed_ring = now_time - shock_ring_start
                if elapsed_ring <= SHOCK_RING_DURATION:
                    progress = clamp(elapsed_ring / SHOCK_RING_DURATION, 0.0, 1.0)
                    impact_position = impact_info.get("position")
                    if impact_position is not None:
                        ring_center = world_to_screen(
                            impact_position[0], impact_position[1], ppm, camera_center_tuple
                        )
                    else:
                        ring_center = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
                    base_ring_radius = max(4, compute_satellite_radius(current_planet.radius) * 2)
                    ring_radius_px = max(
                        base_ring_radius,
                        int(base_ring_radius + planet_radius_px * SHOCK_RING_EXPANSION_FACTOR * progress),
                    )
                    ring_alpha = int(255 * (1.0 - progress))
                    if ring_alpha > 0 and ring_radius_px > 0:
                        ring_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                        ring_width = max(1, min(SHOCK_RING_WIDTH, ring_radius_px))
                        pygame.draw.circle(
                            ring_surface,
                            (*SHOCK_RING_COLOR, ring_alpha),
                            ring_center,
                            ring_radius_px,
                            ring_width,
                        )
                        screen.blit(ring_surface, (0, 0))

            # Draw satellite
            sat_pos = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
            sat_radius_px = compute_satellite_radius(rmag)
            draw_satellite(screen, sat_pos, sat_radius_px)

            # Velocity arrow
            if show_velocity_arrow:
                vx, vy = float(v[0]), float(v[1])
                vmag = math.hypot(vx, vy)
                if vmag > 1e-6:
                    arrow_length = clamp(
                        vmag * VEL_ARROW_SCALE,
                        VEL_ARROW_MIN_PIXELS,
                        VEL_ARROW_MAX_PIXELS,
                    )
                    if arrow_length > 4.0:
                        dir_x = vx / vmag
                        dir_y = vy / vmag
                        arrow_end = (
                            int(round(sat_pos[0] + dir_x * arrow_length)),
                            int(round(sat_pos[1] - dir_y * arrow_length)),
                        )
                        head_length = int(max(4, min(VEL_ARROW_HEAD_LENGTH, arrow_length * 0.6)))
                        draw_velocity_arrow(screen, sat_pos, arrow_end, head_length, VEL_ARROW_HEAD_ANGLE_DEG)

            # Draw orbit markers
            hovered_marker: tuple[str, tuple[int, int], float] | None = None
            if orbit_markers:
                markers_snapshot = list(orbit_markers)
                latest_index: dict[str, int] = {}
                for idx, (marker_type, _, _, _) in enumerate(markers_snapshot):
                    latest_index[marker_type] = idx
                for idx, (marker_type, mx, my, mr) in enumerate(markers_snapshot):
                    marker_pos = world_to_screen(mx, my, ppm, camera_center_tuple)
                    dist_to_mouse = math.hypot(marker_pos[0] - mouse_pos[0], marker_pos[1] - mouse_pos[1])
                    hovered = dist_to_mouse <= LABEL_MARKER_HOVER_RADIUS
                    alpha = LABEL_MARKER_HOVER_ALPHA if hovered else LABEL_MARKER_ALPHA
                    radius = LABEL_MARKER_HOVER_RADIUS_PIXELS if hovered else 4
                    draw_marker_pin(
                        label_layer,
                        marker_pos,
                        color=LABEL_MARKER_COLOR,
                        alpha=alpha,
                        orientation=marker_type,
                    )
                    pygame.draw.circle(label_layer, (*LABEL_MARKER_COLOR, alpha), marker_pos, radius)
                    if hovered:
                        hovered_marker = (marker_type, marker_pos, mr)
                        pygame.draw.circle(
                            label_layer, (*LABEL_MARKER_COLOR, int(alpha * 0.4)), marker_pos, radius + 4, 1
                        )
                screen.blit(label_layer, (0, 0))

            if hovered_marker is not None:
                render_marker_label(screen, hovered_marker[0], hovered_marker[1], hovered_marker[2])

            # HUD
            vmag = float(np.linalg.norm(v))
            eps = energy_specific(r, v, current_planet.mu)
            e = eccentricity(r, v, current_planet.mu)
            altitude_km = (rmag - current_planet.radius) / 1_000.0
            scenario = get_current_scenario()
            
            # Determine colors for physics values based on warning conditions
            altitude_value_color = HUD_WARNING_COLOR if altitude_km < 0 else HUD_VALUE_COLOR
            # Warning color for very high eccentricity (near escape) or negative energy becoming positive
            energy_value_color = HUD_WARNING_COLOR if eps > 0 else HUD_VALUE_COLOR
            
            # HUD entries: tuple of (label, value, value_color) or None for separator
            # Label uses HUD_LABEL_COLOR, value uses specified color
            hud_entries: list[tuple[str, str, tuple[int, int, int]] | None] = [
                # Simulation Info (static values use label color for both)
                ("Planet:", current_planet.name, HUD_LABEL_COLOR),
                ("Scenario:", scenario.name, HUD_LABEL_COLOR),
                ("Integrator:", integrator, HUD_LABEL_COLOR),
                ("Time:", f"{t_sim:,.0f} s", HUD_VALUE_COLOR),
                ("Time Warp:", f"{real_time_speed:.0f}x", HUD_VALUE_COLOR),
                None,  # Separator
                # Orbital Data (physics values use cyan, warnings use orange)
                ("Altitude:", f"{altitude_km:,.1f} km", altitude_value_color),
                ("Velocity:", f"{vmag:,.1f} m/s", HUD_VALUE_COLOR),
                ("Eccentricity:", f"{e:.4f}", HUD_VALUE_COLOR),
                ("Energy:", f"{eps:.2e} J/kg", energy_value_color),
            ]
            
            padding_x = 16
            padding_y = 14
            line_height = font.get_linesize()
            separator_height = 8  # Extra space for separator
            
            # Calculate width (excluding None separators)
            text_entries = [(label, value) for entry in hud_entries if entry is not None for label, value, _ in [entry]]
            hud_width = max(font.size(label + " " + value)[0] for label, value in text_entries) + padding_x * 2
            
            # Calculate height (separators add less height than text lines)
            num_separators = sum(1 for line in hud_entries if line is None)
            num_text_lines = len(hud_entries) - num_separators
            hud_height = line_height * num_text_lines + separator_height * num_separators + padding_y
            
            hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
            pygame.draw.rect(hud_surface, LABEL_BACKGROUND_COLOR, hud_surface.get_rect(), border_radius=14)
            
            y_offset = int(padding_y / 2)
            for entry in hud_entries:
                if entry is None:
                    # Draw separator line
                    sep_y = y_offset + separator_height // 2
                    pygame.draw.line(
                        hud_surface,
                        (*HUD_LABEL_COLOR[:3], 60),  # Dim separator
                        (padding_x, sep_y),
                        (hud_width - padding_x, sep_y),
                        1
                    )
                    y_offset += separator_height
                else:
                    label, value, value_color = entry
                    # Render label in gray
                    label_surf = font.render(label, True, HUD_LABEL_COLOR)
                    hud_surface.blit(label_surf, (padding_x, y_offset))
                    # Render value in its designated color (cyan for physics, orange for warnings)
                    value_surf = font.render(" " + value, True, value_color)
                    hud_surface.blit(value_surf, (padding_x + label_surf.get_width(), y_offset))
                    y_offset += line_height
            
            screen.blit(hud_surface, (20, 20))

            # Sim buttons
            for btn in sim_buttons:
                btn.draw(screen, font, mouse_pos)

            # Impact overlay
            overlay_alpha_factor = 0.0
            overlay_should_draw = False
            if impact_info is not None and state == "impact":
                if impact_overlay_reveal_time is None or now_time >= impact_overlay_reveal_time:
                    if impact_overlay_visible_since is None:
                        impact_overlay_visible_since = now_time
                    fade_elapsed = now_time - impact_overlay_visible_since
                    fade_duration = max(IMPACT_OVERLAY_FADE_DURATION, 1e-6)
                    overlay_alpha_factor = clamp(fade_elapsed / fade_duration, 0.0, 1.0)
                    overlay_should_draw = True

            if overlay_should_draw and impact_info is not None:
                impact_time_value = impact_info.get("time", 0.0)
                impact_speed_value = impact_info.get("speed", 0.0)
                impact_angle_value = impact_info.get("angle", 0.0)
                overlay_lines = [
                    ("IMPACT DETECTED", IMPACT_TITLE_COLOR),
                    (f"t = {impact_time_value:,.2f} s", IMPACT_TEXT_COLOR),
                    (f"|v| = {impact_speed_value:,.1f} m/s", IMPACT_TEXT_COLOR),
                    (f"Impact angle = {impact_angle_value:.1f} deg from horizon", IMPACT_TEXT_COLOR),
                    ("", IMPACT_TEXT_COLOR),
                    ("Press R to reset scenario", IMPACT_TEXT_COLOR),
                    ("Press ESC for main menu", IMPACT_TEXT_COLOR),
                ]
                overlay_padding_x = 24
                overlay_padding_y = 18
                line_height = font.get_linesize()
                overlay_width = max(font.size(text)[0] for text, _ in overlay_lines) + overlay_padding_x * 2
                overlay_height = line_height * len(overlay_lines) + overlay_padding_y * 2
                overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
                pygame.draw.rect(overlay_surface, IMPACT_OVERLAY_COLOR, overlay_surface.get_rect(), border_radius=18)
                for idx, (text, color) in enumerate(overlay_lines):
                    if not text:
                        continue
                    text_surf = font.render(text, True, color)
                    overlay_surface.blit(text_surf, (overlay_padding_x, overlay_padding_y + idx * line_height))
                if overlay_alpha_factor < 1.0:
                    overlay_surface.set_alpha(int(255 * overlay_alpha_factor))
                overlay_rect = overlay_surface.get_rect()
                overlay_rect.center = (WIDTH // 2, HEIGHT // 2)
                screen.blit(overlay_surface, overlay_rect)

            # Scenario flash
            if scenario_flash_text and now_time - scenario_flash_time < SCENARIO_FLASH_DURATION:
                flash_surf = scenario_font.render(scenario_flash_text, True, HUD_TEXT_COLOR)
                flash_rect = flash_surf.get_rect(center=(WIDTH // 2, 40))
                screen.blit(flash_surf, flash_rect)

            # FPS
            fps_value = clock.get_fps()
            fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, HUD_TEXT_COLOR)
            fps_text.set_alpha(FPS_TEXT_ALPHA)
            fps_rect = fps_text.get_rect(bottomright=(WIDTH - 16, HEIGHT - 16))
            screen.blit(fps_text, fps_rect)

            pygame.display.flip()
            clock.tick()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()