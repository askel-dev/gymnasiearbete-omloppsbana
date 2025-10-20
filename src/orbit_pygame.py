# src/orbit_pygame.py
import json
import math
import os
import random
import time
import pygame
from pygame.locals import *  # noqa: F401,F403 - required for constants such as FULLSCREEN
import sys
import numpy as np
from collections import deque
from dataclasses import dataclass
from functools import lru_cache


_TEXT_SURFACE_CACHE: dict[tuple[int, str, tuple[int, int, int] | tuple[int, int, int, int]], pygame.Surface] = {}


def get_text_surface(
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int] | tuple[int, int, int, int],
) -> pygame.Surface:
    """Return a cached rendered surface for the given font, text and color.

    Surfaces retrieved from the cache must be treated as immutable by callers.
    """

    key = (id(font), text, color)
    cached = _TEXT_SURFACE_CACHE.get(key)
    if cached is None:
        cached = font.render(text, True, color)
        _TEXT_SURFACE_CACHE[key] = cached
    return cached

from logging_utils import RunLogger


def draw_text_with_shadow(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    *,
    shadow: bool = False,
) -> None:
    text_surf = get_text_surface(font, text, HUD_TEXT_COLOR)
    if CURRENT_HUD_ALPHA < 255:
        text_surf = text_surf.copy()
        text_surf.set_alpha(int(CURRENT_HUD_ALPHA))
    if shadow:
        shadow_color = HUD_SHADOW_COLOR[:3]
        shadow_surf = get_text_surface(font, text, shadow_color)
        if len(HUD_SHADOW_COLOR) == 4:
            shadow_surf.set_alpha(HUD_SHADOW_COLOR[3])
        x, y = position
        surface.blit(shadow_surf, (x + 1, y + 1))
    surface.blit(text_surf, position)


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


_EARTH_SPRITE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "earth_sprite_2.png"
)
_EARTH_SPRITE_BASE: pygame.Surface | None = None
_EARTH_SPRITE_CACHE: dict[int, pygame.Surface] = {}


def _load_earth_sprite_base() -> pygame.Surface:
    global _EARTH_SPRITE_BASE
    if _EARTH_SPRITE_BASE is None:
        _EARTH_SPRITE_BASE = pygame.image.load(_EARTH_SPRITE_PATH).convert_alpha()
    return _EARTH_SPRITE_BASE


def _get_scaled_earth_sprite(diameter: int) -> pygame.Surface:
    if diameter <= 0:
        raise ValueError("Earth sprite diameter must be positive")

    cached = _EARTH_SPRITE_CACHE.get(diameter)
    if cached is not None:
        return cached

    base_sprite = _load_earth_sprite_base()
    scaled = pygame.transform.smoothscale(base_sprite, (diameter, diameter))
    _EARTH_SPRITE_CACHE[diameter] = scaled
    return scaled


def draw_earth(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
) -> None:
    if radius <= 0:
        return

    diameter = radius * 2
    sprite = _get_scaled_earth_sprite(diameter)
    rect = sprite.get_rect(center=position)
    surface.blit(sprite, rect)


@lru_cache(maxsize=64)
def _satellite_surface(radius: int) -> pygame.Surface:
    size = radius * 4
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2
    light_dir = SAT_LIGHT_DIR
    length = math.hypot(light_dir[0], light_dir[1])
    if length != 0:
        light_dir = (light_dir[0] / length, light_dir[1] / length)
    for y in range(size):
        for x in range(size):
            dx = x - center + 0.5
            dy = y - center + 0.5
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            t = dist / max(1, radius)
            base_color = _lerp_color(SAT_COLOR_CORE, SAT_COLOR_EDGE, t)
            nx = dx / max(1e-6, radius)
            ny = dy / max(1e-6, radius)
            dot = max(0.0, nx * light_dir[0] + ny * light_dir[1])
            rim = min(1.0, (t ** 1.5) * (dot ** 2) * 2.0)
            color = _lerp_color(base_color, SAT_HIGHLIGHT_COLOR, rim)
            surface.set_at((x, y), (*color, 255))
    return surface


def draw_satellite(
    surface: pygame.Surface,
    position: tuple[int, int],
    planet_position: tuple[int, int] | None,
    radius: int,
) -> None:
    if radius <= 0:
        return

    if planet_position is not None:
        dx = planet_position[0] - position[0]
        dy = planet_position[1] - position[1]
        distance = math.hypot(dx, dy)
        if distance != 0:
            offset_scale = radius * 0.6
            offset_x = int(dx / distance * offset_scale)
            offset_y = int(dy / distance * offset_scale)
        else:
            offset_x = offset_y = 0
        shadow_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(
            shadow_surface,
            SAT_SHADOW_COLOR,
            (radius * 2, radius * 2),
            radius,
        )
        shadow_rect = shadow_surface.get_rect(
            center=(position[0] + offset_x, position[1] + offset_y)
        )
        surface.blit(shadow_surface, shadow_rect)

    sat_surface = _satellite_surface(radius)
    sat_rect = sat_surface.get_rect(center=position)
    surface.blit(sat_surface, sat_rect)


def draw_heating_glow(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
    intensity: float,
) -> None:
    if intensity <= 0.0 or radius <= 0:
        return
    intensity = clamp(intensity, 0.0, 1.0)
    glow_radius = max(2, int(radius * (1.8 + ATM_GLOW_RADIUS_FACTOR * intensity)))
    glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    center = glow_radius
    outer_alpha = int(ATM_GLOW_OUTER_ALPHA * intensity)
    inner_alpha = int(ATM_GLOW_INNER_ALPHA * intensity)
    if outer_alpha > 0:
        pygame.draw.circle(
            glow_surface,
            (*ATM_GLOW_COLOR, outer_alpha),
            (center, center),
            glow_radius,
        )
    if inner_alpha > 0:
        pygame.draw.circle(
            glow_surface,
            (*ATM_GLOW_COLOR, inner_alpha),
            (center, center),
            max(1, int(glow_radius * 0.55)),
        )
    glow_rect = glow_surface.get_rect(center=position)
    surface.blit(glow_surface, glow_rect)


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

    pygame.draw.polygon(
        surface,
        (*color, alpha),
        [base_left, base_right, tip],
    )

    pygame.draw.line(
        surface,
        (*color, min(255, int(alpha * 0.95))),
        position,
        tip,
        2,
    )


def draw_menu_planet(
    surface: pygame.Surface,
    center: tuple[int, int],
    diameter: int,
    *,
    image: pygame.Surface | None = None,
) -> None:
    if diameter <= 0:
        return

    if image is not None:
        width, height = image.get_size()
        if width <= 0 or height <= 0:
            return
        scale = diameter / max(width, height)
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        cache_key = (id(image), new_size[0], new_size[1])
        scaled = _MENU_PLANET_CACHE.get(cache_key)
        if scaled is None:
            scaled = pygame.transform.smoothscale(image, new_size).convert_alpha()
            _MENU_PLANET_CACHE[cache_key] = scaled
        rect = scaled.get_rect(center=center)
        surface.blit(scaled, rect)
        return

    radius = diameter // 2
    planet_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(
        planet_surface,
        PLANET_COLOR,
        (radius, radius),
        radius,
    )
    surface.blit(planet_surface, planet_surface.get_rect(center=center))


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


def generate_starfield(num_stars: int) -> list[dict[str, object]]:
    stars: list[dict[str, object]] = []
    for _ in range(num_stars):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.choice([1, 1, 1, 2])
        alpha = random.randint(80, 150)
        base = random.randint(200, 240)
        color = (
            max(0, base - random.randint(10, 25)),
            max(0, base - random.randint(5, 15)),
            base,
        )
        star_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            star_surface,
            (*color, alpha),
            (radius, radius),
            radius,
        )
        stars.append(
            {
                "pos": (x, y),
                "surface": star_surface,
                "radius": radius,
            }
        )
    return stars


def draw_starfield(surface: pygame.Surface, camera_center: np.ndarray, ppm: float) -> None:
    offset_x = camera_center[0] * ppm * STARFIELD_PARALLAX
    offset_y = camera_center[1] * ppm * STARFIELD_PARALLAX
    for star in STARFIELD:
        base_x, base_y = star["pos"]  # type: ignore[index]
        star_surface = star["surface"]  # type: ignore[index]
        radius = star["radius"]  # type: ignore[index]
        sx = int((base_x - offset_x) % WIDTH)
        sy = int((base_y + offset_y) % HEIGHT)
        surface.blit(star_surface, (sx - radius, sy - radius))


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
                    label_surf = get_text_surface(
                        tick_font, label_text, GRID_LABEL_COLOR
                    )
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
                    label_surf = get_text_surface(
                        tick_font, label_text, GRID_LABEL_COLOR
                    )
                    if GRID_LABEL_ALPHA < 255:
                        label_surf = label_surf.copy()
                        label_surf.set_alpha(GRID_LABEL_ALPHA)
                    rect = label_surf.get_rect()
                    rect.midright = (width - GRID_LABEL_MARGIN, sy)
                    if rect.left >= 0:
                        surface.blit(label_surf, rect)

    axis_color = GRID_LABEL_COLOR
    axis_label = get_text_surface(axis_font, "Y [Mm]", axis_color)
    if GRID_AXIS_LABEL_ALPHA < 255:
        axis_label = axis_label.copy()
        axis_label.set_alpha(GRID_AXIS_LABEL_ALPHA)
    axis_rect = axis_label.get_rect()
    axis_rect.bottomright = (width - GRID_LABEL_MARGIN, height - GRID_LABEL_MARGIN)
    surface.blit(axis_label, axis_rect)

    y_axis_label = get_text_surface(axis_font, "X [Mm]", axis_color)
    if GRID_AXIS_LABEL_ALPHA < 255:
        y_axis_label = y_axis_label.copy()
        y_axis_label.set_alpha(GRID_AXIS_LABEL_ALPHA)
    y_axis_rect = y_axis_label.get_rect()
    y_axis_rect.topright = (width - GRID_LABEL_MARGIN, GRID_LABEL_MARGIN)
    surface.blit(y_axis_label, y_axis_rect)


@dataclass
class ParallaxLayer:
    surface: pygame.Surface
    velocity: tuple[float, float]
    mouse_factor: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def advance(self, dt: float) -> None:
        if dt <= 0.0:
            return
        width = max(1, self.surface.get_width())
        height = max(1, self.surface.get_height())
        self.offset_x = (self.offset_x + self.velocity[0] * dt) % width
        self.offset_y = (self.offset_y + self.velocity[1] * dt) % height

    def draw(self, target: pygame.Surface, mouse_offset: tuple[float, float]) -> None:
        width = self.surface.get_width()
        height = self.surface.get_height()
        if width == 0 or height == 0:
            return
        mouse_shift_x = mouse_offset[0] * self.mouse_factor * width
        mouse_shift_y = mouse_offset[1] * self.mouse_factor * height
        offset_x = (self.offset_x + mouse_shift_x) % width
        offset_y = (self.offset_y + mouse_shift_y) % height
        ox = int(offset_x)
        oy = int(offset_y)

        base_x = -ox
        base_y = -oy
        target.blit(self.surface, (base_x, base_y))

        if ox:
            horizontal_rect = pygame.Rect(0, 0, ox, height)
            target.blit(
                self.surface,
                (width - ox, base_y),
                area=horizontal_rect,
            )
        if oy:
            vertical_rect = pygame.Rect(0, 0, width, oy)
            target.blit(
                self.surface,
                (base_x, height - oy),
                area=vertical_rect,
            )
        if ox and oy:
            corner_rect = pygame.Rect(0, 0, ox, oy)
            target.blit(
                self.surface,
                (width - ox, height - oy),
                area=corner_rect,
            )


def generate_menu_parallax_layers(width: int, height: int) -> list[ParallaxLayer]:
    rng = random.Random(5123)
    star_count = max(120, int(max(width, height) * 0.25))
    layer_specs = [
        {
            "count": max(1, int(star_count * 0.55)),
            "radius_choices": [1, 1, 1, 2],
            "alpha_range": (45, 110),
            "velocity": (-8.0, 0.0),
            "mouse_factor": 0.010,
        },
        {
            "count": max(1, int(star_count * 0.35)),
            "radius_choices": [1, 2, 2, 3],
            "alpha_range": (80, 160),
            "velocity": (-18.0, 4.0),
            "mouse_factor": 0.018,
        },
        {
            "count": max(1, int(star_count * 0.22)),
            "radius_choices": [2, 3, 3, 4],
            "alpha_range": (110, 210),
            "velocity": (-32.0, 12.0),
            "mouse_factor": 0.028,
        },
    ]
    layers: list[ParallaxLayer] = []
    for spec in layer_specs:
        layer_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for _ in range(spec["count"]):
            radius = rng.choice(spec["radius_choices"])
            alpha = rng.randint(*spec["alpha_range"])
            color = (
                rng.randint(120, 190),
                rng.randint(160, 220),
                255,
                alpha,
            )
            x = rng.randint(0, width)
            y = rng.randint(0, height)
            pygame.draw.circle(layer_surface, color, (x, y), radius)
        layer = ParallaxLayer(
            surface=layer_surface,
            velocity=(
                spec["velocity"][0] * (0.6 + rng.random() * 0.8),
                spec["velocity"][1] * (0.6 + rng.random() * 0.8),
            ),
            mouse_factor=spec["mouse_factor"],
        )
        layer.offset_x = rng.uniform(0, max(1, width))
        layer.offset_y = rng.uniform(0, max(1, height))
        layers.append(layer)
    return layers


def load_font(preferred_names: list[str], size: int, *, bold: bool = False) -> pygame.font.Font:
    for name in preferred_names:
        try:
            match = pygame.font.match_font(name, bold=bold)
        except Exception:
            match = None
        if match:
            return pygame.font.Font(match, size)
    fallback_name = preferred_names[0] if preferred_names else None
    return pygame.font.SysFont(fallback_name, size, bold=bold)


# =======================
#   FYSIK & KONSTANTER
# =======================
G = 6.674e-11                 # gravitationskonstant (SI)
M = 5.972e24                  # Jordens massa (kg)
MU = G * M
EARTH_RADIUS = 6_371_000      # m
# Startvillkor
R0 = np.array([7_000_000.0, 0.0])  # m
V0 = np.array([0.0, 7_600.0])      # m/s


@dataclass(frozen=True)
class Scenario:
    """Preset initial-conditions used by the Scenario Mode."""

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

# =======================
#   SIMULATOR-SETTINGS
# =======================
DT_PHYS = 0.25                 # fysikens tidssteg (sekunder)
REAL_TIME_SPEED = 240.0        # sim-sek per real-sek (startvärde)
MAX_SUBSTEPS = 20             # skydd mot för många fysiksteg/frame
LOG_EVERY_STEPS = 20           # logga var 20:e fysiksteg
ESCAPE_RADIUS_FACTOR = 20.0
ORBIT_PREDICTION_INTERVAL = 1.0
MAX_ORBIT_PREDICTION_SAMPLES = 2_000
MAX_RENDERED_ORBIT_POINTS = 800

# =======================
#   RIT- & KONTROLL-SETTINGS
# =======================
# Dessa värden sätts om efter att displayen initierats, men behöver
# startvärden för typkontroller och tooling.
WIDTH, HEIGHT = 1000, 800
BACKGROUND_COLOR = (0, 34, 72)
PLANET_COLOR = (255, 183, 77)
SAT_COLOR_CORE = (65, 224, 162)
SAT_COLOR_EDGE = (42, 170, 226)
SAT_HIGHLIGHT_COLOR = (206, 255, 250)
SAT_LIGHT_DIR = (-0.55, -0.4)
SAT_SHADOW_COLOR = (6, 12, 24, 120)
SAT_BASE_RADIUS = 6
HUD_TEXT_COLOR = (234, 241, 255)
HUD_TEXT_ALPHA_BASE = 255
CURRENT_HUD_ALPHA: float = float(HUD_TEXT_ALPHA_BASE)
HUD_SHADOW_COLOR = (10, 15, 30, 120)
ORBIT_PRIMARY_COLOR = (255, 255, 255, 180)
ORBIT_SECONDARY_COLOR = (220, 236, 255, 140)
ORBIT_LINE_WIDTH = 2
VEL_ARROW_COLOR = (255, 220, 180)
VEL_ARROW_SCALE = 0.004
VEL_ARROW_MIN_PIXELS = 0
VEL_ARROW_MAX_PIXELS = 90
VEL_ARROW_HEAD_LENGTH = 10
VEL_ARROW_HEAD_ANGLE_DEG = 26
BUTTON_COLOR = (8, 32, 64, int(255 * 0.78))
BUTTON_HOVER_COLOR = (18, 52, 94, int(255 * 0.88))
BUTTON_TEXT_COLOR = (234, 241, 255)
BUTTON_BORDER_COLOR = (88, 140, 255, int(255 * 0.55))
BUTTON_HOVER_BORDER_COLOR = (118, 180, 255, int(255 * 0.8))
BUTTON_RADIUS = 18
MENU_TITLE_COLOR = (234, 241, 255)
MENU_SUBTITLE_COLOR = (180, 198, 228)
MENU_BUTTON_COLOR = (9, 44, 92, 220)
MENU_BUTTON_HOVER_COLOR = (24, 74, 140, 235)
MENU_BUTTON_BORDER_COLOR = (255, 255, 255, 50)
MENU_BUTTON_TEXT_COLOR = (234, 241, 255)
MENU_BUTTON_RADIUS = 20
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
LABEL_MARKER_PINNED_PIN_COLOR = (248, 252, 255)
LABEL_MARKER_PINNED_GLOW_COLOR = (255, 255, 255)
LABEL_MARKER_PINNED_GLOW_ALPHA = 90
LABEL_MARKER_PINNED_OUTLINE_ALPHA = 210
LABEL_MARKER_PINNED_RADIUS_PIXELS = 8
LABEL_MARKER_PINNED_GLOW_RADIUS = 14
LABEL_PINNED_BACKGROUND_COLOR = (16, 28, 46, int(255 * 0.42))
LABEL_PINNED_BADGE_COLOR = (255, 255, 255, 220)
LABEL_PINNED_BADGE_TEXT_COLOR = (18, 36, 64)
MARKER_PIN_FEEDBACK_DURATION = 1.6
FPS_TEXT_ALPHA = int(255 * 0.6)
STARFIELD_PARALLAX = 0.12

ATM_ALTITUDE = 120_000.0  # m
ATM_BOUNDARY_RADIUS = EARTH_RADIUS + ATM_ALTITUDE
ATM_DRAG_COEFF = 1.4e-5
ATM_WARNING_DURATION = 2.0
ATM_WARNING_COLOR = (255, 176, 120)
ATM_GLOW_COLOR = (255, 120, 80)
ATM_GLOW_OUTER_ALPHA = 90
ATM_GLOW_INNER_ALPHA = 180
ATM_GLOW_RADIUS_FACTOR = 2.6

IMPACT_FREEZE_DELAY = 0.7
IMPACT_HUD_ALPHA_FACTOR = 0.35
IMPACT_HUD_FADE_DURATION = 0.6
IMPACT_OVERLAY_DELAY = 2.0
IMPACT_OVERLAY_FADE_DURATION = 1.0
IMPACT_OVERLAY_COLOR = (12, 18, 30, int(255 * 0.75))
IMPACT_TITLE_COLOR = (255, 214, 130)
IMPACT_TEXT_COLOR = (234, 241, 255)
SHOCK_RING_COLOR = (255, 66, 66)
SHOCK_RING_DURATION = 2.5
SHOCK_RING_EXPANSION_FACTOR = 6.0
SHOCK_RING_WIDTH = 6

GRID_SPACING_METERS = 1_000_000.0
GRID_MIN_PIXEL_SPACING = 42.0
GRID_LINE_COLOR = (200, 208, 220)
GRID_LINE_ALPHA = 40
GRID_LABEL_COLOR = (208, 216, 228)
GRID_LABEL_ALPHA = 180
GRID_AXIS_LABEL_ALPHA = 160
GRID_LABEL_MARGIN = 10

STARFIELD: list[dict[str, object]] = []
MENU_PLANET_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "menu_planet.png"
)
_MENU_PLANET_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}

def compute_pixels_per_meter(width: int, height: int) -> float:
    """Compute the rendering scale in pixels per meter.

    The original implementation used a heavy downscale (0.35× of the natural
    screen fit) which made the satellite appear glued to Earth despite the
    physical orbit radius being correct. Raising the factor to 0.6 keeps the
    simulation true-to-scale while better utilising the available canvas so the
    nominal ~7 000 km orbit has visibly more separation from the planet.
    """

    base_scale = min(width, height) / (2.0 * np.linalg.norm(R0))
    return 0.60 * base_scale


def update_display_metrics(width: int, height: int) -> None:
    global WIDTH, HEIGHT, PIXELS_PER_METER

    WIDTH = width
    HEIGHT = height
    PIXELS_PER_METER = compute_pixels_per_meter(width, height)


PIXELS_PER_METER = compute_pixels_per_meter(WIDTH, HEIGHT)
MIN_PPM = 1e-7
MAX_PPM = 1e-2

# =======================
#   HJÄLPMETODER
# =======================
def in_atmosphere(r: np.ndarray) -> bool:
    return np.linalg.norm(r) <= ATM_BOUNDARY_RADIUS


def atmosphere_depth_ratio(r_magnitude: float) -> float:
    if ATM_ALTITUDE <= 0.0:
        return 0.0
    return clamp((ATM_BOUNDARY_RADIUS - r_magnitude) / ATM_ALTITUDE, 0.0, 1.0)


def accel(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    rmag = np.linalg.norm(r)
    if rmag <= 0.0:
        return np.zeros_like(r)
    gravitational = -MU * r / (rmag**3)
    if rmag <= ATM_BOUNDARY_RADIUS:
        depth = atmosphere_depth_ratio(rmag)
        drag_coeff = ATM_DRAG_COEFF * (0.2 + 0.8 * depth)
        gravitational += -drag_coeff * v
    return gravitational


def rk4_step(r, v, dt):
    a1 = accel(r, v);                    k1_r = v;              k1_v = a1
    a2 = accel(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v);  k2_r = v + 0.5 * dt * k1_v; k2_v = a2
    a3 = accel(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v);  k3_r = v + 0.5 * dt * k2_v; k3_v = a3
    a4 = accel(r + dt * k3_r, v + dt * k3_v);              k4_r = v + dt * k3_v;     k4_v = a4
    r_next = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return r_next, v_next

def energy_specific(r, v):
    rmag = np.linalg.norm(r)
    vmag2 = v[0]*v[0] + v[1]*v[1]
    return 0.5*vmag2 - MU/rmag

def eccentricity(r, v):
    r3 = np.array([r[0], r[1], 0.0])
    v3 = np.array([v[0], v[1], 0.0])
    h = np.cross(r3, v3)
    e_vec = np.cross(v3, h)/MU - r3/np.linalg.norm(r3)
    return np.linalg.norm(e_vec[:2])

def world_to_screen(x, y, ppm, camera_center=(0.0, 0.0)):
    cx, cy = camera_center
    sx = WIDTH // 2 + int((x - cx) * ppm)
    sy = HEIGHT // 2 - int((y - cy) * ppm)
    return sx, sy

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_satellite_radius(r_magnitude: float) -> int:
    altitude = max(0.0, r_magnitude - EARTH_RADIUS)
    scale = 1.0 + clamp(altitude / 20_000_000.0, 0.0, 0.2)
    return max(3, int(round(SAT_BASE_RADIUS * scale)))


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


def compute_orbit_prediction(r_init: np.ndarray, v_init: np.ndarray) -> tuple[float | None, list[tuple[float, float]]]:
    eps = energy_specific(r_init, v_init)
    if eps >= 0.0:
        return None, []

    a = -MU / (2.0 * eps)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    estimated_samples = max(360, int(period / ORBIT_PREDICTION_INTERVAL))
    num_samples = max(2, min(MAX_ORBIT_PREDICTION_SAMPLES, estimated_samples))
    dt = period / num_samples

    r = r_init.copy()
    v = v_init.copy()
    points: list[tuple[float, float]] = []
    for _ in range(num_samples + 1):
        points.append((float(r[0]), float(r[1])))
        r, v = rk4_step(r, v, dt)

    return period, points


@dataclass(frozen=True)
class ButtonVisualStyle:
    base_color: tuple[int, int, int] | tuple[int, int, int, int]
    hover_color: tuple[int, int, int] | tuple[int, int, int, int]
    text_color: tuple[int, int, int]
    radius: int
    border_color: tuple[int, int, int] | tuple[int, int, int, int] | None = None
    border_width: int = 0


DEFAULT_BUTTON_STYLE = ButtonVisualStyle(
    base_color=BUTTON_COLOR,
    hover_color=BUTTON_HOVER_COLOR,
    text_color=BUTTON_TEXT_COLOR,
    radius=BUTTON_RADIUS,
)

MENU_BUTTON_STYLE = ButtonVisualStyle(
    base_color=MENU_BUTTON_COLOR,
    hover_color=MENU_BUTTON_HOVER_COLOR,
    text_color=MENU_BUTTON_TEXT_COLOR,
    radius=MENU_BUTTON_RADIUS,
    border_color=MENU_BUTTON_BORDER_COLOR,
    border_width=2,
)


class Button:
    """Simple rectangular button with hover feedback and callbacks."""

    def __init__(self, rect, text, callback, text_getter=None, *, style: ButtonVisualStyle | None = None):
        self.rect = pygame.Rect(rect)
        self._text = text
        self._callback = callback
        self._text_getter = text_getter
        self._cached_text_surface: pygame.Surface | None = None
        self._cached_text: str | None = None
        self._cached_font_id: int | None = None
        self._style = style

    def get_text(self):
        if self._text_getter is not None:
            return self._text_getter()
        return self._text

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        style = self._style or DEFAULT_BUTTON_STYLE
        color = style.hover_color if hovered else style.base_color
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surface,
            color,
            button_surface.get_rect(),
            border_radius=style.radius,
        )
        if style.border_color is not None and style.border_width > 0:
            pygame.draw.rect(
                button_surface,
                style.border_color,
                button_surface.get_rect(),
                style.border_width,
                border_radius=style.radius,
            )
        surface.blit(button_surface, self.rect.topleft)
        text_value = self.get_text()
        font_id = id(font)
        if (
            self._cached_text_surface is None
            or text_value != self._cached_text
            or font_id != self._cached_font_id
        ):
            self._cached_text_surface = get_text_surface(
                font, text_value, style.text_color
            )
            self._cached_text = text_value
            self._cached_font_id = font_id
        text_surf = self._cached_text_surface
        if CURRENT_HUD_ALPHA < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(int(CURRENT_HUD_ALPHA))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._callback()

# =======================
#   HUVUDPROGRAM
# =======================


def _set_display_mode_with_vsync(
    size: tuple[int, int],
    flags: int = 0,
) -> pygame.Surface:
    """Create the display surface with double buffering and vsync when available."""

    flags |= DOUBLEBUF

    try:
        return pygame.display.set_mode(size, flags, vsync=1)
    except TypeError:
        # Older pygame versions may not support the ``vsync`` keyword argument.
        return pygame.display.set_mode(size, flags)
    except pygame.error as err:
        # Some platforms reject the vsync request even if the keyword is supported.
        try:
            return pygame.display.set_mode(size, flags)
        except pygame.error:
            raise err


def main():
    pygame.init()
    pygame.display.set_caption("Gymnasiearbete - Simulering av omloppsbana")
    global CURRENT_HUD_ALPHA
    CURRENT_HUD_ALPHA = float(HUD_TEXT_ALPHA_BASE)
    font_fps = pygame.font.SysFont("consolas", 14)

    fullscreen_flags = FULLSCREEN | DOUBLEBUF
    borderless_flags = NOFRAME | DOUBLEBUF
    info = pygame.display.Info()
    fallback_resolution = (info.current_w or WIDTH, info.current_h or HEIGHT)

    screen: pygame.Surface | None = None
    if info.current_w and info.current_h:
        try:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
            screen = _set_display_mode_with_vsync(
                (info.current_w, info.current_h),
                borderless_flags,
            )
        except pygame.error:
            screen = None

    if screen is None:
        try:
            screen = _set_display_mode_with_vsync((0, 0), fullscreen_flags)
        except pygame.error:
            # Fallback till fönsterläge om fullscreen inte stöds.
            screen = _set_display_mode_with_vsync(fallback_resolution, DOUBLEBUF)

    screen_width, screen_height = screen.get_size()
    update_display_metrics(screen_width, screen_height)

    overlay_size = (WIDTH, HEIGHT)
    orbit_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
    label_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)
    grid_surface = pygame.Surface(overlay_size, pygame.SRCALPHA)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    scenario_font = pygame.font.SysFont("consolas", 16)
    grid_label_font = pygame.font.SysFont("consolas", 14)
    grid_axis_font = pygame.font.SysFont("consolas", 16)

    def render_marker_label(
        surface: pygame.Surface,
        marker_type: str,
        marker_pos: tuple[int, int],
        radius_world: float,
        *,
        pinned: bool = False,
    ) -> None:
        label = marker_display_name(marker_type)
        altitude_km = (radius_world - EARTH_RADIUS) / 1_000.0
        text = f"{label}: {altitude_km:,.1f} km"
        text_surf = get_text_surface(font, text, LABEL_TEXT_COLOR)
        if CURRENT_HUD_ALPHA < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(int(CURRENT_HUD_ALPHA))

        padding = 6
        bg_width = text_surf.get_width() + padding * 2
        bg_height = text_surf.get_height() + padding * 2

        background_color = LABEL_PINNED_BACKGROUND_COLOR if pinned else LABEL_BACKGROUND_COLOR
        connector_alpha = LABEL_MARKER_PINNED_OUTLINE_ALPHA if pinned else int(LABEL_MARKER_ALPHA * 0.6)
        connector_color = LABEL_MARKER_PINNED_GLOW_COLOR if pinned else LABEL_MARKER_COLOR
        line_length = 24 if pinned else 18
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
            background_color,
            bg_rect,
            border_radius=12 if pinned else 10,
        )

        surface.blit(text_surf, (bg_rect.left + padding, bg_rect.top + padding))

        if pinned:
            badge_text = "PINNED"
            badge_padding_x = 6
            badge_padding_y = 2
            badge_surf = get_text_surface(
                font_fps, badge_text, LABEL_PINNED_BADGE_TEXT_COLOR
            )
            badge_rect = pygame.Rect(
                0,
                0,
                badge_surf.get_width() + badge_padding_x * 2,
                badge_surf.get_height() + badge_padding_y * 2,
            )
            badge_rect.midbottom = (bg_rect.centerx, bg_rect.top - 4)
            pygame.draw.rect(
                surface,
                LABEL_PINNED_BADGE_COLOR,
                badge_rect,
                border_radius=badge_rect.height // 2,
            )
            surface.blit(
                badge_surf,
                (badge_rect.left + badge_padding_x, badge_rect.top + badge_padding_y),
            )

    min_dimension = min(WIDTH, HEIGHT)
    title_font_size = max(48, int(min_dimension * 0.075))
    subtitle_font_size = max(26, int(min_dimension * 0.032))

    background_color = BACKGROUND_COLOR
    menu_parallax_layers = generate_menu_parallax_layers(WIDTH, HEIGHT)
    menu_mouse_offset = [0.0, 0.0]
    menu_last_time = time.perf_counter()

    title_font = load_font(["montserrat", "futura", "avenir", "arial"], title_font_size, bold=True)
    subtitle_font = load_font(["montserrat", "futura", "avenir", "arial"], subtitle_font_size)
    menu_button_size = max(60, int(min_dimension * 0.08))
    menu_planet_diameter = max(700, min(320, int(min_dimension * 0.32)))
    menu_button_height = max(100, int(menu_button_size * 0.26))
    menu_button_font_size = min(menu_button_height - 12, max(26, int(menu_button_height * 0.52)))
    menu_button_font = load_font(
        ["montserrat", "futura", "avenir", "arial"],
        menu_button_font_size,
        bold=True,
    )

    assets_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    planet_image_path = os.path.join(assets_root, "main_menu_art.png")
    menu_planet_image: pygame.Surface | None = None
    if os.path.exists(planet_image_path):
        try:
            menu_planet_image = pygame.image.load(planet_image_path).convert_alpha()
        except pygame.error:
            menu_planet_image = None

    current_scenario_key = DEFAULT_SCENARIO_KEY
    scenario_flash_text: str | None = None
    scenario_flash_time = 0.0

    scenario_shortcut_map: dict[int, str] = {}
    number_key_codes = [
        pygame.K_1,
        pygame.K_2,
        pygame.K_3,
        pygame.K_4,
        pygame.K_5,
        pygame.K_6,
        pygame.K_7,
        pygame.K_8,
        pygame.K_9,
    ]
    keypad_key_codes = [
        pygame.K_KP1,
        pygame.K_KP2,
        pygame.K_KP3,
        pygame.K_KP4,
        pygame.K_KP5,
        pygame.K_KP6,
        pygame.K_KP7,
        pygame.K_KP8,
        pygame.K_KP9,
    ]
    for index, scenario_key in enumerate(SCENARIO_DISPLAY_ORDER, start=1):
        if index <= len(number_key_codes):
            scenario_shortcut_map[number_key_codes[index - 1]] = scenario_key
        if index <= len(keypad_key_codes):
            scenario_shortcut_map[keypad_key_codes[index - 1]] = scenario_key

    scenario_panel_title = "Scenario Mode – press 1-5 to switch"
    scenario_help_lines = [
        f"[{idx}] {SCENARIOS[key].name} – {SCENARIOS[key].description}"
        for idx, key in enumerate(SCENARIO_DISPLAY_ORDER, start=1)
    ]

    def get_current_scenario() -> Scenario:
        return SCENARIOS[current_scenario_key]

    def scenario_velocity_vector(key: str | None = None) -> np.ndarray:
        scenario = SCENARIOS[key or current_scenario_key]
        return scenario.velocity_vector()

    global STARFIELD
    if not STARFIELD:
        STARFIELD = generate_starfield(220)

    layout_center_y = HEIGHT // 2
    title_y = max(int(layout_center_y - HEIGHT * 0.27), int(HEIGHT * 0.14))
    subtitle_y = title_y + max(40, int(HEIGHT * 0.06))
    planet_center = (WIDTH // 2, layout_center_y - int(HEIGHT * 0.03))
    menu_button_width = max(200, int(menu_button_height * 6))
    menu_button_gap = max(20, int(menu_button_height * 0.36))
    button_block_height = menu_button_height * 2 + menu_button_gap
    default_button_y = planet_center[1] + menu_planet_diameter // 2 + max(48, int(HEIGHT * 0.07))
    max_button_y = HEIGHT - button_block_height - 80
    menu_button_y = min(default_button_y, max_button_y)
    menu_button_y = max(menu_button_y, subtitle_y + max(60, int(HEIGHT * 0.08)))
    menu_button_x = WIDTH // 2 - menu_button_width // 2
    menu_title_text = "SIMULERING AV OMLOPPSBANA"
    menu_subtitle_text = "av Axel Jönsson"

    # Simuleringsstate
    r = R0.copy()
    v = scenario_velocity_vector()
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
    pinned_markers: dict[str, tuple[float, float, float]] = {}
    pin_feedback_text: str | None = None
    pin_feedback_time = 0.0
    impact_info: dict[str, float] | None = None
    shock_ring_start: float | None = None
    impact_freeze_time: float | None = None
    impact_overlay_reveal_time: float | None = None
    impact_overlay_visible_since: float | None = None
    atmosphere_entry_time_sim: float | None = None
    atmosphere_entry_time_real: float | None = None
    atmosphere_warning_end_time: float = 0.0
    atmosphere_logged = False

    # tidsackumulator för fast fysik
    accumulator = 0.0
    last_time = time.perf_counter()

    # loggningsstate
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
        nonlocal pinned_markers, pin_feedback_text, pin_feedback_time
        nonlocal impact_info, shock_ring_start, impact_freeze_time
        nonlocal impact_overlay_reveal_time, impact_overlay_visible_since
        nonlocal atmosphere_entry_time_sim
        nonlocal atmosphere_entry_time_real, atmosphere_warning_end_time
        nonlocal atmosphere_logged, show_velocity_arrow
        global CURRENT_HUD_ALPHA
        close_logger()
        r = R0.copy()
        v = scenario_velocity_vector()
        t_sim = 0.0
        paused = False
        ppm = PIXELS_PER_METER
        ppm_target = ppm
        real_time_speed = REAL_TIME_SPEED
        accumulator = 0.0
        last_time = time.perf_counter()
        log_step_counter = 0
        prev_r = None
        prev_dr = None
        impact_logged = False
        escape_logged = False
        orbit_markers.clear()
        pinned_markers.clear()
        pin_feedback_text = None
        pin_feedback_time = 0.0
        camera_center[:] = (0.0, 0.0)
        camera_target[:] = (0.0, 0.0)
        camera_mode = "earth"
        show_velocity_arrow = True
        is_dragging_camera = False
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(r, v)
        impact_info = None
        shock_ring_start = None
        impact_freeze_time = None
        impact_overlay_reveal_time = None
        impact_overlay_visible_since = None
        atmosphere_entry_time_sim = None
        atmosphere_entry_time_real = None
        atmosphere_warning_end_time = 0.0
        atmosphere_logged = False
        CURRENT_HUD_ALPHA = float(HUD_TEXT_ALPHA_BASE)

    state = "menu"
    escape_radius_limit = ESCAPE_RADIUS_FACTOR * float(np.linalg.norm(R0))

    def marker_display_name(marker_type: str) -> str:
        return "Periapsis" if marker_type == "pericenter" else "Apoapsis"

    def get_latest_marker(marker_type: str) -> tuple[float, float, float] | None:
        for m_type, mx, my, mr in reversed(orbit_markers):
            if m_type == marker_type:
                return (mx, my, mr)
        return None

    def refresh_pinned_marker(marker_type: str) -> None:
        if marker_type not in pinned_markers:
            return

        latest = get_latest_marker(marker_type)
        if latest is not None:
            pinned_markers[marker_type] = latest
        else:
            del pinned_markers[marker_type]

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

    def toggle_marker_pin(marker_type: str) -> bool:
        if marker_type in pinned_markers:
            del pinned_markers[marker_type]
            return False
        latest = get_latest_marker(marker_type)
        if latest is None:
            return False
        pinned_markers[marker_type] = latest
        return True

    def set_scenario(new_key: str) -> None:
        nonlocal current_scenario_key, scenario_flash_text, scenario_flash_time
        if new_key not in SCENARIOS:
            return
        current_scenario_key = new_key
        scenario = get_current_scenario()
        scenario_flash_text = f"{scenario.name} scenario loaded"
        scenario_flash_time = time.perf_counter()
        reset()
        if state == "running":
            init_run_logging()

    def log_state(dt_eff: float) -> None:
        if logger is None:
            return
        rmag = float(np.linalg.norm(r))
        vmag = float(np.linalg.norm(v))
        eps = float(energy_specific(r, v))
        h_vec = np.cross(np.array([r[0], r[1], 0.0]), np.array([v[0], v[1], 0.0]))
        h_mag = float(np.linalg.norm(h_vec))
        e_val = float(eccentricity(r, v))
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
        v0_vector = scenario.velocity_vector()
        meta = {
            "scenario_key": scenario.key,
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "R0": R0.tolist(),
            "V0": v0_vector.tolist(),
            "v0": float(np.linalg.norm(v0_vector)),
            "G": G,
            "M": M,
            "mu": MU,
            "integrator": "RK4",
            "dt_phys": DT_PHYS,
            "start_speed": REAL_TIME_SPEED,
            "log_strategy": "every_20_steps",
            "code_version": "v1.0",
        }
        logger.write_meta(meta)
        log_step_counter = 0
        prev_r = float(np.linalg.norm(r))
        prev_dr = None
        impact_logged = False
        escape_logged = False
        log_state(0.0)

    def start_simulation():
        nonlocal state
        reset()
        state = "running"
        init_run_logging()

    def quit_app():
        close_logger()
        pygame.quit()
        sys.exit()

    menu_buttons = [
        Button(
            (menu_button_x, menu_button_y, menu_button_width, menu_button_height),
            "STARTA SIMULATION",
            start_simulation,
            style=MENU_BUTTON_STYLE,
        ),
        Button(
            (
                menu_button_x,
                menu_button_y + menu_button_height + menu_button_gap,
                menu_button_width,
                menu_button_height,
            ),
            "AVSLUTA",
            quit_app,
            style=MENU_BUTTON_STYLE,
        ),
    ]

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
        else:
            camera_mode = "earth"

    def load_next_scenario() -> None:
        nonlocal current_scenario_key, state, paused
        if not SCENARIO_DISPLAY_ORDER:
            return
        try:
            current_index = SCENARIO_DISPLAY_ORDER.index(current_scenario_key)
        except ValueError:
            current_index = 0
        next_index = (current_index + 1) % len(SCENARIO_DISPLAY_ORDER)
        previous_state = state
        set_scenario(SCENARIO_DISPLAY_ORDER[next_index])
        if previous_state == "impact":
            state = "running"
            paused = False
            init_run_logging()
        elif previous_state != "menu":
            state = "running"

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
        Button(
            (20, 20 + 2 * (button_height + button_gap), button_width, button_height),
            "Slower",
            slow_down,
        ),
        Button(
            (20, 20 + 3 * (button_height + button_gap), button_width, button_height),
            "Faster",
            speed_up,
        ),
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
    ]

    update_sim_button_layout()

    def is_over_button(pos: tuple[int, int]) -> bool:
        if state == "menu":
            return any(btn.rect.collidepoint(pos) for btn in menu_buttons)
        if state == "running":
            return any(btn.rect.collidepoint(pos) for btn in sim_buttons)
        return False

    # ========= LOOP =========
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
        orbit_drawn = False
        labels_drawn = False
        # --- Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_app()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_app()
                    continue
                if event.key in scenario_shortcut_map:
                    set_scenario(scenario_shortcut_map[event.key])
                    if state == "impact":
                        state = "running"
                        paused = False
                        init_run_logging()
                    continue
                if event.key == pygame.K_g:
                    grid_overlay_enabled = not grid_overlay_enabled
                    continue
                if state == "impact":
                    if event.key == pygame.K_r:
                        reset_and_continue()
                    elif event.key == pygame.K_n:
                        load_next_scenario()
                    continue
                if state == "running":
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        reset_and_continue()
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        ppm_target = clamp(ppm_target * 1.2, MIN_PPM, MAX_PPM)     # zoom in
                    elif event.key == pygame.K_MINUS:
                        ppm_target = clamp(ppm_target / 1.2, MIN_PPM, MAX_PPM)     # zoom out

                    # ---- Piltangenter styr simhastighet (ingen boost längre) ----
                    elif event.key == pygame.K_RIGHT:
                        real_time_speed = min(real_time_speed * 1.5, 10_000.0)
                    elif event.key == pygame.K_LEFT:
                        real_time_speed = max(real_time_speed / 1.5, 0.1)
                    elif event.key == pygame.K_UP:
                        real_time_speed = min(real_time_speed * 2.0, 10_000.0)
                    elif event.key == pygame.K_DOWN:
                        real_time_speed = max(real_time_speed / 2.0, 0.1)
                    elif event.key == pygame.K_n:
                        load_next_scenario()
                    elif event.key == pygame.K_c:
                        toggle_camera()
                    elif event.key == pygame.K_v:
                        show_velocity_arrow = not show_velocity_arrow
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and state == "running":
                    if not is_over_button(event.pos):
                        clicked_marker = marker_hit_test(event.pos)
                        if clicked_marker is not None:
                            is_now_pinned = toggle_marker_pin(clicked_marker)
                            if is_now_pinned:
                                refresh_pinned_marker(clicked_marker)
                            status = "pinned" if is_now_pinned else "unpinned"
                            pin_feedback_text = f"{marker_display_name(clicked_marker)} {status}"
                            pin_feedback_time = time.perf_counter()
                        else:
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
                    ppm_target = clamp(ppm_target * zoom_factor, MIN_PPM, MAX_PPM)
            if state == "menu":
                for btn in menu_buttons:
                    btn.handle_event(event)
            elif state == "running":
                for btn in sim_buttons:
                    btn.handle_event(event)

        if state == "menu":
            now_menu = time.perf_counter()
            menu_frame_dt = min(now_menu - menu_last_time, 0.05)
            menu_last_time = now_menu
            menu_mouse_pos = pygame.mouse.get_pos()
            for layer in menu_parallax_layers:
                layer.advance(menu_frame_dt)

            center_x = WIDTH * 0.5
            center_y = HEIGHT * 0.5
            target_offset_x = (menu_mouse_pos[0] - center_x) / max(1.0, WIDTH)
            target_offset_y = (menu_mouse_pos[1] - center_y) / max(1.0, HEIGHT)
            smoothing = 1.0 - math.exp(-menu_frame_dt * 6.0) if menu_frame_dt > 0.0 else 1.0
            menu_mouse_offset[0] += (target_offset_x - menu_mouse_offset[0]) * smoothing
            menu_mouse_offset[1] += (target_offset_y - menu_mouse_offset[1]) * smoothing

            screen.fill(background_color)
            for layer in menu_parallax_layers:
                layer.draw(screen, (menu_mouse_offset[0], menu_mouse_offset[1]))

            title_surf = get_text_surface(
                title_font, menu_title_text, MENU_TITLE_COLOR
            )
            title_rect = title_surf.get_rect(center=(WIDTH // 2, title_y))
            screen.blit(title_surf, title_rect)

            subtitle_surf = get_text_surface(
                subtitle_font, menu_subtitle_text, MENU_SUBTITLE_COLOR
            )
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, subtitle_y))
            screen.blit(subtitle_surf, subtitle_rect)

            draw_menu_planet(screen, planet_center, menu_planet_diameter, image=menu_planet_image)

            for btn in menu_buttons:
                btn.draw(screen, menu_button_font, menu_mouse_pos)

            panel_lines = [scenario_panel_title, *scenario_help_lines]
            if panel_lines:
                panel_padding = 14
                line_height = scenario_font.get_linesize()
                panel_width = max(scenario_font.size(line)[0] for line in panel_lines) + panel_padding * 2
                panel_height = line_height * len(panel_lines) + panel_padding * 2
                panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
                pygame.draw.rect(
                    panel_surface,
                    LABEL_BACKGROUND_COLOR,
                    panel_surface.get_rect(),
                    border_radius=12,
                )
                y = panel_padding
                for idx, line in enumerate(panel_lines):
                    if idx == 0:
                        color = HUD_TEXT_COLOR
                    else:
                        scenario_key = SCENARIO_DISPLAY_ORDER[idx - 1]
                        color = HUD_TEXT_COLOR if scenario_key == current_scenario_key else MENU_SUBTITLE_COLOR
                    text_surf = get_text_surface(scenario_font, line, color)
                    panel_surface.blit(text_surf, (panel_padding, y))
                    y += line_height
                panel_rect = panel_surface.get_rect()
                panel_rect.topleft = (20, HEIGHT - panel_height - 20)
                screen.blit(panel_surface, panel_rect)

            if scenario_flash_text and now_menu - scenario_flash_time < SCENARIO_FLASH_DURATION:
                flash_surf = get_text_surface(
                    scenario_font, scenario_flash_text, MENU_TITLE_COLOR
                )
                flash_rect = flash_surf.get_rect(center=(WIDTH // 2, menu_button_y - 40))
                screen.blit(flash_surf, flash_rect)

            # --- FPS Counter ---
            fps_value = clock.get_fps()
            fps_text = get_text_surface(
                font_fps, f"FPS: {fps_value:.1f}", (140, 180, 220)
            )
            # placera i nedre högra hörnet
            text_rect = fps_text.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
            screen.blit(fps_text, text_rect)


            pygame.display.flip()
            clock.tick(240)
            continue

        # --- Fysik ackumulator ---
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
                r, v = rk4_step(r, v, dt_step)
                t_sim += dt_step
                rmag = float(np.linalg.norm(r))
                current_in_atmosphere = in_atmosphere(r)
                if current_in_atmosphere and atmosphere_entry_time_sim is None:
                    atmosphere_entry_time_sim = float(t_sim)
                    atmosphere_entry_time_real = time.perf_counter()
                    atmosphere_warning_end_time = atmosphere_entry_time_real + ATM_WARNING_DURATION
                event_logged = False
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
                    refresh_pinned_marker(event_type)

                vmag = float(np.linalg.norm(v))
                eps = float(energy_specific(r, v))
                e_val = float(eccentricity(r, v))
                impact_triggered = not impact_logged and rmag <= EARTH_RADIUS

                if logger is not None:
                    log_step_counter += 1

                    if event_type is not None:
                        logger.log_event(
                            [
                                float(t_sim),
                                event_type,
                                rmag,
                                vmag,
                                json.dumps({"ecc": e_val, "energy": eps}),
                            ]
                        )
                        event_logged = True

                    if impact_triggered:
                        logger.log_event(
                            [
                                float(t_sim),
                                "impact",
                                rmag,
                                vmag,
                                json.dumps({"penetration": EARTH_RADIUS - rmag, "energy": eps}),
                            ]
                        )
                        event_logged = True

                    if (
                        atmosphere_entry_time_sim is not None
                        and not atmosphere_logged
                        and current_in_atmosphere
                    ):
                        logger.log_event(
                            [
                                float(t_sim),
                                "atmosphere_entry",
                                rmag,
                                vmag,
                                json.dumps({"altitude": rmag - EARTH_RADIUS, "energy": eps}),
                            ]
                        )
                        atmosphere_logged = True
                        event_logged = True

                    if not escape_logged and eps > 0.0 and rmag > escape_radius_limit:
                        logger.log_event(
                            [
                                float(t_sim),
                                "escape",
                                rmag,
                                vmag,
                                json.dumps({"energy": eps, "ecc": e_val}),
                            ]
                        )
                        escape_logged = True
                        event_logged = True

                    if log_step_counter >= LOG_EVERY_STEPS or event_logged:
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
                        overlay_ready_time = shock_ring_start + IMPACT_OVERLAY_DELAY
                        if atmosphere_warning_end_time > 0.0:
                            overlay_ready_time = max(overlay_ready_time, atmosphere_warning_end_time)
                        impact_overlay_reveal_time = overlay_ready_time
                        impact_overlay_visible_since = None
                        paused = True
                        state = "impact"
                        accumulator = 0.0
                        break

            accumulator = 0.0

        now_time = time.perf_counter()
        # --- Render ---
        target_hud_alpha = float(HUD_TEXT_ALPHA_BASE)
        if (
            impact_info is not None
            and shock_ring_start is not None
            and impact_freeze_time is not None
        ):
            elapsed_since_impact = now_time - shock_ring_start
            if elapsed_since_impact >= IMPACT_FREEZE_DELAY:
                fade_duration = max(IMPACT_HUD_FADE_DURATION, 1e-6)
                fade_progress = clamp(
                    (elapsed_since_impact - IMPACT_FREEZE_DELAY) / fade_duration,
                    0.0,
                    1.0,
                )
                target_factor = 1.0 - (1.0 - IMPACT_HUD_ALPHA_FACTOR) * fade_progress
                target_hud_alpha = HUD_TEXT_ALPHA_BASE * target_factor
        CURRENT_HUD_ALPHA += (target_hud_alpha - CURRENT_HUD_ALPHA) * 0.2
        CURRENT_HUD_ALPHA = clamp(CURRENT_HUD_ALPHA, 0.0, HUD_TEXT_ALPHA_BASE)

        # Smooth zoom mot mål
        ppm += (ppm_target - ppm) * 0.1
        ppm = clamp(ppm, MIN_PPM, MAX_PPM)

        # Kamera-targets
        if camera_mode == "earth":
            view_offset = (HEIGHT * 0.08) / max(ppm, 1e-6)
            camera_target[:] = (0.0, view_offset)
        elif camera_mode == "satellite":
            camera_target[:] = (r[0], r[1])
        else:
            camera_target[:] = camera_center
        camera_center += (camera_target - camera_center) * 0.1

        # Bakgrund
        screen.fill(background_color)
        draw_starfield(screen, camera_center, ppm)

        camera_center_tuple = (float(camera_center[0]), float(camera_center[1]))
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
        depth_ratio = atmosphere_depth_ratio(rmag)
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)

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
                orbit_drawn = True

        if orbit_drawn:
            screen.blit(orbit_layer, (0, 0))

        earth_radius_px = max(1, int(EARTH_RADIUS * ppm))
        draw_earth(
            screen,
            earth_screen_pos,
            earth_radius_px,
        )

        if impact_info is not None and shock_ring_start is not None:
            elapsed_ring = now_time - shock_ring_start
            if elapsed_ring <= SHOCK_RING_DURATION:
                progress = clamp(elapsed_ring / SHOCK_RING_DURATION, 0.0, 1.0)
                impact_position = impact_info.get("position") if impact_info else None
                if impact_position is not None:
                    ring_center = world_to_screen(
                        impact_position[0],
                        impact_position[1],
                        ppm,
                        camera_center_tuple,
                    )
                else:
                    ring_center = world_to_screen(
                        r[0],
                        r[1],
                        ppm,
                        camera_center_tuple,
                    )
                base_ring_radius = max(4, compute_satellite_radius(EARTH_RADIUS) * 2)
                ring_radius_px = max(
                    base_ring_radius,
                    int(
                        base_ring_radius
                        + earth_radius_px * SHOCK_RING_EXPANSION_FACTOR * progress
                    ),
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

        sat_pos = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
        sat_radius_px = compute_satellite_radius(rmag)
        heating_intensity = depth_ratio if rmag >= EARTH_RADIUS else 1.0
        if heating_intensity > 0.0:
            draw_heating_glow(screen, sat_pos, sat_radius_px, heating_intensity)
        draw_satellite(screen, sat_pos, earth_screen_pos, sat_radius_px)

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
                    draw_velocity_arrow(
                        screen,
                        sat_pos,
                        arrow_end,
                        head_length,
                        VEL_ARROW_HEAD_ANGLE_DEG,
                    )

        if atmosphere_entry_time_real is not None:
            elapsed_entry = now_time - atmosphere_entry_time_real
            if elapsed_entry < ATM_WARNING_DURATION:
                fade = clamp(1.0 - elapsed_entry / ATM_WARNING_DURATION, 0.0, 1.0)
                warning_alpha = int(255 * fade)
                warning_text = "Atmospheric entry"
                warning_surface = get_text_surface(
                    scenario_font, warning_text, ATM_WARNING_COLOR
                )
                if warning_alpha < 255:
                    warning_surface = warning_surface.copy()
                    warning_surface.set_alpha(warning_alpha)
                warning_bg_padding = 14
                warning_bg = pygame.Surface(
                    (warning_surface.get_width() + warning_bg_padding * 2,
                     warning_surface.get_height() + warning_bg_padding),
                    pygame.SRCALPHA,
                )
                pygame.draw.rect(
                    warning_bg,
                    (ATM_WARNING_COLOR[0], ATM_WARNING_COLOR[1], ATM_WARNING_COLOR[2], int(warning_alpha * 0.3)),
                    warning_bg.get_rect(),
                    border_radius=16,
                )
                warning_bg.blit(warning_surface, (warning_bg_padding, warning_bg_padding // 2))
                warning_rect = warning_bg.get_rect(center=(WIDTH // 2, int(HEIGHT * 0.14)))
                screen.blit(warning_bg, warning_rect)

        hovered_marker: tuple[str, tuple[int, int], float] | None = None
        hovered_distance = float("inf")
        pinned_label_entries: list[tuple[str, tuple[int, int], float]] = []
        if orbit_markers:
            markers_snapshot = list(orbit_markers)
            latest_index: dict[str, int] = {}
            for idx, (marker_type, _, _, _) in enumerate(markers_snapshot):
                latest_index[marker_type] = idx
            for idx, (marker_type, mx, my, mr) in enumerate(markers_snapshot):
                marker_pos = world_to_screen(mx, my, ppm, camera_center_tuple)
                dist_to_mouse = math.hypot(
                    marker_pos[0] - mouse_pos[0], marker_pos[1] - mouse_pos[1]
                )
                hovered = dist_to_mouse <= LABEL_MARKER_HOVER_RADIUS
                is_latest = latest_index.get(marker_type) == idx
                is_pinned = marker_type in pinned_markers and is_latest
                alpha = LABEL_MARKER_HOVER_ALPHA if hovered else LABEL_MARKER_ALPHA
                radius = LABEL_MARKER_HOVER_RADIUS_PIXELS if hovered else 4
                pin_color = LABEL_MARKER_PINNED_PIN_COLOR if is_pinned else LABEL_MARKER_COLOR
                if is_pinned:
                    alpha = max(alpha, 240)
                    radius = max(radius, LABEL_MARKER_PINNED_RADIUS_PIXELS)
                    pygame.draw.circle(
                        label_layer,
                        (*LABEL_MARKER_PINNED_GLOW_COLOR, LABEL_MARKER_PINNED_GLOW_ALPHA),
                        marker_pos,
                        LABEL_MARKER_PINNED_GLOW_RADIUS,
                    )
                    pygame.draw.circle(
                        label_layer,
                        (*LABEL_MARKER_PINNED_GLOW_COLOR, LABEL_MARKER_PINNED_OUTLINE_ALPHA),
                        marker_pos,
                        radius + 4,
                        2,
                    )
                    pinned_label_entries.append((marker_type, marker_pos, mr))
                draw_marker_pin(
                    label_layer,
                    marker_pos,
                    color=pin_color,
                    alpha=alpha,
                    orientation=marker_type,
                )
                pygame.draw.circle(
                    label_layer,
                    (*pin_color, alpha),
                    marker_pos,
                    radius,
                )
                if hovered:
                    if dist_to_mouse < hovered_distance:
                        hovered_distance = dist_to_mouse
                        hovered_marker = (marker_type, marker_pos, mr)
                    pygame.draw.circle(
                        label_layer,
                        (*pin_color, int(alpha * 0.4)),
                        marker_pos,
                        radius + 4,
                        1,
                    )
                labels_drawn = True

        visible_labels: list[tuple[str, tuple[int, int], float, bool]] = []
        for entry in pinned_label_entries:
            visible_labels.append((*entry, True))
        if hovered_marker is not None:
            if not any(entry[0] == hovered_marker[0] for entry in pinned_label_entries):
                visible_labels.append((*hovered_marker, False))

        for marker_type, marker_pos, mr, is_pinned_label in visible_labels:
            render_marker_label(
                label_layer,
                marker_type,
                marker_pos,
                mr,
                pinned=is_pinned_label,
            )
        if labels_drawn:
            screen.blit(label_layer, (0, 0))

        if pin_feedback_text is not None:
            elapsed_feedback = now_time - pin_feedback_time
            if elapsed_feedback < MARKER_PIN_FEEDBACK_DURATION:
                fade = clamp(1.0 - elapsed_feedback / MARKER_PIN_FEEDBACK_DURATION, 0.0, 1.0)
                feedback_alpha = int(255 * fade)
                feedback_text_surf = get_text_surface(
                    scenario_font, pin_feedback_text, LABEL_TEXT_COLOR
                )
                if feedback_alpha < 255:
                    feedback_text_surf = feedback_text_surf.copy()
                    feedback_text_surf.set_alpha(feedback_alpha)
                padding_x = 18
                padding_y = 8
                bubble_width = feedback_text_surf.get_width() + padding_x * 2
                bubble_height = feedback_text_surf.get_height() + padding_y * 2
                bubble = pygame.Surface((bubble_width, bubble_height), pygame.SRCALPHA)
                pygame.draw.rect(
                    bubble,
                    (*LABEL_MARKER_PINNED_GLOW_COLOR, int(feedback_alpha * 0.45)),
                    bubble.get_rect(),
                    border_radius=18,
                )
                bubble.blit(feedback_text_surf, (padding_x, padding_y))
                bubble_rect = bubble.get_rect(center=(WIDTH // 2, int(HEIGHT * 0.18)))
                screen.blit(bubble, bubble_rect)
            else:
                pin_feedback_text = None

        # HUD
        vmag = float(np.linalg.norm(v))
        eps = energy_specific(r, v)
        e = eccentricity(r, v)
        altitude_km = (rmag - EARTH_RADIUS) / 1_000.0
        scenario = get_current_scenario()
        hud_entries: list[tuple[str, tuple[int, int, int]]] = [
            (f"Scenario: {scenario.name}", HUD_TEXT_COLOR),
            (f"t {t_sim:,.0f} s   ×{real_time_speed:.1f}", HUD_TEXT_COLOR),
        ]
        altitude_color = HUD_TEXT_COLOR
        altitude_line = f"alt {altitude_km:,.1f} km"
        if depth_ratio > 0.0:
            altitude_line = f"alt {altitude_km:,.1f} km   ATM"
            altitude_color = ATM_WARNING_COLOR
        hud_entries.append((altitude_line, altitude_color))
        hud_entries.extend(
            [
                (f"|v| {vmag:,.1f} m/s   e {e:.3f}", HUD_TEXT_COLOR),
                (f"ε {eps: .2e} J/kg", HUD_TEXT_COLOR),
            ]
        )
        padding_x = 16
        padding_y = 14
        line_height = font.get_linesize()
        hud_width = max(font.size(line)[0] for line, _ in hud_entries) + padding_x * 2
        hud_height = line_height * len(hud_entries) + padding_y
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        pygame.draw.rect(
            hud_surface,
            LABEL_BACKGROUND_COLOR,
            hud_surface.get_rect(),
            border_radius=14,
        )
        for index, (line, color) in enumerate(hud_entries):
            text_surf = get_text_surface(font, line, color)
            if CURRENT_HUD_ALPHA < 255:
                text_surf = text_surf.copy()
                text_surf.set_alpha(int(CURRENT_HUD_ALPHA))
            hud_surface.blit(
                text_surf,
                (padding_x, int(padding_y / 2) + index * line_height),
            )
        if CURRENT_HUD_ALPHA < 255:
            hud_surface.set_alpha(int(CURRENT_HUD_ALPHA))
        screen.blit(hud_surface, (20, 20))

        for btn in sim_buttons:
            btn.draw(screen, font, mouse_pos)

        panel_lines = [scenario_panel_title, *scenario_help_lines]
        if panel_lines:
            panel_padding = 14
            line_height = scenario_font.get_linesize()
            panel_width = max(scenario_font.size(line)[0] for line in panel_lines) + panel_padding * 2
            panel_height = line_height * len(panel_lines) + panel_padding * 2
            panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            pygame.draw.rect(
                panel_surface,
                LABEL_BACKGROUND_COLOR,
                panel_surface.get_rect(),
                border_radius=12,
            )
            y = panel_padding
            for idx, line in enumerate(panel_lines):
                if idx == 0:
                    color = HUD_TEXT_COLOR
                else:
                    scenario_key = SCENARIO_DISPLAY_ORDER[idx - 1]
                    color = HUD_TEXT_COLOR if scenario_key == current_scenario_key else MENU_SUBTITLE_COLOR
                text_surf = get_text_surface(scenario_font, line, color)
                if CURRENT_HUD_ALPHA < 255:
                    text_surf = text_surf.copy()
                    text_surf.set_alpha(int(CURRENT_HUD_ALPHA))
                panel_surface.blit(text_surf, (panel_padding, y))
                y += line_height
            if CURRENT_HUD_ALPHA < 255:
                panel_surface.set_alpha(int(CURRENT_HUD_ALPHA))
            panel_rect = panel_surface.get_rect()
            panel_rect.topleft = (20, HEIGHT - panel_height - 20)
            screen.blit(panel_surface, panel_rect)

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
            else:
                overlay_alpha_factor = 0.0
        else:
            impact_overlay_visible_since = None

        if overlay_should_draw and impact_info is not None:
            impact_time_value = impact_info.get("time", 0.0)
            impact_speed_value = impact_info.get("speed", 0.0)
            impact_angle_value = impact_info.get("angle", 0.0)
            overlay_lines = [
                ("IMPACT DETECTED", IMPACT_TITLE_COLOR),
                (f"t = {impact_time_value:,.2f} s", IMPACT_TEXT_COLOR),
                (f"|v| = {impact_speed_value:,.1f} m/s", IMPACT_TEXT_COLOR),
                (
                    f"Impact angle = {impact_angle_value:.1f}° from horizon",
                    IMPACT_TEXT_COLOR,
                ),
                ("", IMPACT_TEXT_COLOR),
                ("Press R to reset scenario", IMPACT_TEXT_COLOR),
                ("Press N to load next scenario", IMPACT_TEXT_COLOR),
            ]

            overlay_padding_x = 24
            overlay_padding_y = 18
            line_height = font.get_linesize()
            overlay_width = max(font.size(text)[0] for text, _ in overlay_lines) + overlay_padding_x * 2
            overlay_height = line_height * len(overlay_lines) + overlay_padding_y * 2
            overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
            pygame.draw.rect(
                overlay_surface,
                IMPACT_OVERLAY_COLOR,
                overlay_surface.get_rect(),
                border_radius=18,
            )
            for idx, (text, color) in enumerate(overlay_lines):
                if not text:
                    continue
                text_surf = get_text_surface(font, text, color)
                overlay_surface.blit(
                    text_surf,
                    (overlay_padding_x, overlay_padding_y + idx * line_height),
                )

            if overlay_alpha_factor < 1.0:
                overlay_surface.set_alpha(int(255 * overlay_alpha_factor))

            overlay_rect = overlay_surface.get_rect()
            overlay_rect.center = (WIDTH // 2, HEIGHT // 2)
            screen.blit(overlay_surface, overlay_rect)

        if scenario_flash_text and now_time - scenario_flash_time < SCENARIO_FLASH_DURATION:
            flash_surf = get_text_surface(
                scenario_font, scenario_flash_text, HUD_TEXT_COLOR
            )
            if CURRENT_HUD_ALPHA < 255:
                flash_surf = flash_surf.copy()
                flash_surf.set_alpha(int(CURRENT_HUD_ALPHA))
            flash_rect = flash_surf.get_rect(center=(WIDTH // 2, 40))
            screen.blit(flash_surf, flash_rect)

        fps_value = clock.get_fps()
        fps_text = get_text_surface(font_fps, f"FPS: {fps_value:.1f}", HUD_TEXT_COLOR)
        combined_alpha = FPS_TEXT_ALPHA
        if CURRENT_HUD_ALPHA < 255:
            combined_alpha = min(combined_alpha, int(CURRENT_HUD_ALPHA))
        if combined_alpha < 255:
            fps_text = fps_text.copy()
            fps_text.set_alpha(combined_alpha)
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
