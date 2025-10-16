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
from itertools import islice

from logging_utils import RunLogger


def create_vertical_gradient(
    width: int,
    height: int,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
) -> pygame.Surface:
    surface = pygame.Surface((width, height))
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(
            int(top_color[i] + (bottom_color[i] - top_color[i]) * ratio)
            for i in range(3)
        )
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface.convert()


def draw_text_with_shadow(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    *,
    shadow: bool = False,
) -> None:
    text_surf = font.render(text, True, HUD_TEXT_COLOR)
    if HUD_TEXT_ALPHA < 255:
        text_surf.set_alpha(HUD_TEXT_ALPHA)
    if shadow:
        shadow_color = HUD_SHADOW_COLOR[:3]
        shadow_surf = font.render(text, True, shadow_color)
        if len(HUD_SHADOW_COLOR) == 4:
            shadow_surf.set_alpha(HUD_SHADOW_COLOR[3])
        x, y = position
        surface.blit(shadow_surf, (x + 1, y + 1))
    surface.blit(text_surf, position)


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


@lru_cache(maxsize=128)
def _planet_surface(radius: int) -> pygame.Surface:
    if radius <= 0:
        return pygame.Surface((1, 1), pygame.SRCALPHA)

    outer_radius = max(radius * 3, radius + 12)
    size = outer_radius * 2
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = outer_radius

    inner_halo_radius = max(radius * 2, radius + 4)
    outer_halo_radius = max(radius * 3, inner_halo_radius + 4)

    pygame.draw.circle(surface, INNER_HALO_COLOR, (center, center), inner_halo_radius)
    pygame.draw.circle(surface, OUTER_HALO_COLOR, (center, center), outer_halo_radius)

    diameter = radius * 2
    planet_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    planet_center = radius - 0.5
    for y in range(diameter):
        for x in range(diameter):
            dx = x - planet_center
            dy = y - planet_center
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            t = dist / max(1, radius)
            if t <= 0.5:
                color = _lerp_color(PLANET_COLOR_CORE, PLANET_COLOR_MID, t / 0.5)
            else:
                color = _lerp_color(PLANET_COLOR_MID, PLANET_COLOR_EDGE, (t - 0.5) / 0.5)
            planet_surface.set_at((x, y), (*color, 255))

    surface.blit(planet_surface, (center - radius, center - radius))
    return surface


def draw_earth(surface: pygame.Surface, position: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        return
    earth_surface = _planet_surface(radius)
    rect = earth_surface.get_rect(center=position)
    surface.blit(earth_surface, rect)


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
        scale = diameter / max(width, height)
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        scaled = pygame.transform.smoothscale(image, new_size)
        rect = scaled.get_rect(center=center)
        surface.blit(scaled, rect)
        return

    radius = diameter // 2
    glow_radius = int(radius * 1.35)
    placeholder_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(
        placeholder_surface,
        MENU_PLANET_GLOW_COLOR,
        (glow_radius, glow_radius),
        glow_radius,
    )

    planet_surface = _planet_surface(radius)
    planet_rect = planet_surface.get_rect(center=(glow_radius, glow_radius))
    placeholder_surface.blit(planet_surface, planet_rect)

    orbit_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    orbit_rect = pygame.Rect(0, 0, int(radius * 2.6), int(radius * 1.6))
    orbit_rect.center = (glow_radius, glow_radius + MENU_PLANET_ORBIT_OFFSET)
    pygame.draw.ellipse(
        orbit_surface,
        MENU_PLANET_RING_COLOR,
        orbit_rect,
        MENU_PLANET_RING_WIDTH,
    )
    orbit_surface = pygame.transform.rotate(orbit_surface, -18)
    placeholder_surface.blit(orbit_surface, (0, 0))

    satellite_radius = max(3, radius // 7)
    satellite_angle = math.radians(32)
    satellite_distance = radius + int(radius * 0.55)
    satellite_pos = (
        glow_radius + int(math.cos(satellite_angle) * satellite_distance),
        glow_radius - int(math.sin(satellite_angle) * satellite_distance),
    )
    pygame.draw.circle(
        placeholder_surface,
        MENU_PLANET_RING_COLOR,
        satellite_pos,
        satellite_radius,
    )

    placeholder_surface.set_alpha(MENU_PLANET_PLACEHOLDER_ALPHA)
    surface.blit(placeholder_surface, placeholder_surface.get_rect(center=center))


def draw_velocity_arrow(surface: pygame.Surface, start: tuple[int, int], end: tuple[int, int], head_length: int, head_angle: float) -> None:
    pygame.draw.line(surface, VEL_COLOR, start, end, 2)
    angle = math.atan2(start[1] - end[1], end[0] - start[0])
    left = (
        int(end[0] - head_length * math.cos(angle - head_angle)),
        int(end[1] + head_length * math.sin(angle - head_angle)),
    )
    right = (
        int(end[0] - head_length * math.cos(angle + head_angle)),
        int(end[1] + head_length * math.sin(angle + head_angle)),
    )
    pygame.draw.polygon(surface, VEL_COLOR, [end, left, right])


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
        ox = (self.offset_x + mouse_shift_x) % width
        oy = (self.offset_y + mouse_shift_y) % height
        base_x = -ox
        base_y = -oy
        for dx in (0, width):
            for dy in (0, height):
                target.blit(self.surface, (int(base_x + dx), int(base_y + dy)))


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

# =======================
#   RIT- & KONTROLL-SETTINGS
# =======================
# Dessa värden sätts om efter att displayen initierats, men behöver
# startvärden för typkontroller och tooling.
WIDTH, HEIGHT = 1000, 800
BG_COLOR_TOP = (0, 17, 40)
BG_COLOR_BOTTOM = (0, 34, 72)
PLANET_COLOR_CORE = (255, 183, 77)
PLANET_COLOR_MID = (240, 98, 146)
PLANET_COLOR_EDGE = (124, 77, 255)
INNER_HALO_COLOR = (124, 77, 255, int(255 * 0.10))
OUTER_HALO_COLOR = (46, 209, 195, int(255 * 0.08))
SAT_COLOR_CORE = (65, 224, 162)
SAT_COLOR_EDGE = (42, 170, 226)
SAT_HIGHLIGHT_COLOR = (206, 255, 250)
SAT_LIGHT_DIR = (-0.55, -0.4)
SAT_SHADOW_COLOR = (6, 12, 24, 120)
SAT_BASE_RADIUS = 6
HUD_TEXT_COLOR = (234, 241, 255)
HUD_TEXT_ALPHA = 255
HUD_SHADOW_COLOR = (10, 15, 30, 120)
ORBIT_PRIMARY_COLOR = (46, 209, 195)
ORBIT_SECONDARY_COLOR = (46, 209, 195, 160)
ORBIT_LINE_WIDTH = 2
TRAIL_COLOR = (46, 209, 195)
TRAIL_MAX_DURATION = 1.0
VEL_COLOR = (46, 209, 195)
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
MENU_PLANET_GLOW_COLOR = (88, 146, 255, 70)
MENU_PLANET_RING_COLOR = (93, 200, 255, 200)
MENU_PLANET_RING_WIDTH = 4
MENU_PLANET_ORBIT_OFFSET = 18
MENU_PLANET_PLACEHOLDER_ALPHA = 220
LABEL_BACKGROUND_COLOR = (12, 18, 30, int(255 * 0.18))
LABEL_MARKER_COLOR = (46, 209, 195)
LABEL_TEXT_COLOR = (234, 241, 255)
LABEL_MARKER_ALPHA = int(255 * 0.9)
FPS_TEXT_ALPHA = int(255 * 0.6)
STARFIELD_PARALLAX = 0.12

STARFIELD: list[dict[str, object]] = []
MENU_PLANET_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "menu_planet.png"
)
_MENU_PLANET_CACHE: dict[tuple[int, int], pygame.Surface] = {}

def compute_pixels_per_meter(width: int, height: int) -> float:
    return 0.25 * (min(width, height) / (2.0 * np.linalg.norm(R0)))


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
def accel(r: np.ndarray) -> np.ndarray:
    rmag = np.linalg.norm(r)
    return -MU * r / (rmag**3)

def rk4_step(r, v, dt):
    a1 = accel(r);                k1_r = v;              k1_v = a1
    a2 = accel(r + 0.5*dt*k1_r);  k2_r = v + 0.5*dt*k1_v; k2_v = a2
    a3 = accel(r + 0.5*dt*k2_r);  k3_r = v + 0.5*dt*k2_v; k3_v = a3
    a4 = accel(r + dt*k3_r);      k4_r = v + dt*k3_v;     k4_v = a4
    r_next = r + (dt/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
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
            self._cached_text_surface = font.render(text_value, True, style.text_color)
            self._cached_text = text_value
            self._cached_font_id = font_id
        text_surf = self._cached_text_surface
        if HUD_TEXT_ALPHA < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(HUD_TEXT_ALPHA)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._callback()

# =======================
#   HUVUDPROGRAM
# =======================
def main():
    pygame.init()
    pygame.display.set_caption("Gymnasiearbete - Simulering av omloppsbana")
    font_fps = pygame.font.SysFont("consolas", 14)

    fullscreen_flags = FULLSCREEN | DOUBLEBUF
    borderless_flags = NOFRAME | DOUBLEBUF
    info = pygame.display.Info()
    fallback_resolution = (info.current_w or WIDTH, info.current_h or HEIGHT)

    screen: pygame.Surface | None = None
    if info.current_w and info.current_h:
        try:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
            screen = pygame.display.set_mode((info.current_w, info.current_h), borderless_flags)
        except pygame.error:
            screen = None

    if screen is None:
        try:
            screen = pygame.display.set_mode((0, 0), fullscreen_flags)
        except pygame.error:
            # Fallback till fönsterläge om fullscreen inte stöds.
            screen = pygame.display.set_mode(fallback_resolution, DOUBLEBUF)

    screen_width, screen_height = screen.get_size()
    update_display_metrics(screen_width, screen_height)

    overlay_size = (WIDTH, HEIGHT)
    trail_surface = pygame.Surface(overlay_size, pygame.SRCALPHA)
    label_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    min_dimension = min(WIDTH, HEIGHT)
    title_font_size = max(48, int(min_dimension * 0.075))
    subtitle_font_size = max(26, int(min_dimension * 0.032))

    gradient_bg = create_vertical_gradient(WIDTH, HEIGHT, BG_COLOR_TOP, BG_COLOR_BOTTOM)
    menu_background = gradient_bg.copy()
    menu_parallax_layers = generate_menu_parallax_layers(WIDTH, HEIGHT)
    menu_mouse_offset = [0.0, 0.0]
    menu_last_time = time.perf_counter()

    menu_glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    glow_radius = int(min(WIDTH, HEIGHT) * 0.45)
    pygame.draw.circle(
        menu_glow_surface,
        (70, 120, 255, 45),
        (WIDTH // 2, HEIGHT // 2),
        glow_radius,
    )
    menu_background.blit(menu_glow_surface, (0, 0))

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
    v = V0.copy()
    t_sim = 0.0
    paused = False
    ppm = PIXELS_PER_METER
    ppm_target = ppm
    real_time_speed = REAL_TIME_SPEED
    camera_mode = "earth"
    camera_center = np.array([0.0, 0.0], dtype=float)
    camera_target = np.array([0.0, 0.0], dtype=float)
    is_dragging_camera = False
    drag_last_pos = (0, 0)

    orbit_prediction_period: float | None = None
    orbit_prediction_points: list[tuple[float, float]] = []

    orbit_markers: deque[tuple[str, float, float, float]] = deque(maxlen=20)
    trail_history: deque[tuple[float, float, float]] = deque(maxlen=1200)

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
        nonlocal camera_center, orbit_markers, camera_target, trail_history
        nonlocal camera_mode, is_dragging_camera
        nonlocal orbit_prediction_period, orbit_prediction_points
        close_logger()
        r = R0.copy()
        v = V0.copy()
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
        trail_history.clear()
        camera_center[:] = (0.0, 0.0)
        camera_target[:] = (0.0, 0.0)
        camera_mode = "earth"
        is_dragging_camera = False
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(r, v)

    state = "menu"
    escape_radius_limit = ESCAPE_RADIUS_FACTOR * float(np.linalg.norm(R0))

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
        meta = {
            "R0": R0.tolist(),
            "V0": V0.tolist(),
            "v0": float(np.linalg.norm(V0)),
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

    def reset_and_continue():
        reset()
        if state == "running":
            init_run_logging()

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
            trail_surface = pygame.Surface(overlay_size, pygame.SRCALPHA)
            label_layer = pygame.Surface(overlay_size, pygame.SRCALPHA)

        trail_surface.fill((0, 0, 0, 0))
        label_layer.fill((0, 0, 0, 0))
        trail_drawn = False
        labels_drawn = False
        # --- Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_app()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_app()
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
                    elif event.key == pygame.K_c:
                        toggle_camera()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and state == "running" and not is_over_button(event.pos):
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

            screen.blit(menu_background, (0, 0))
            for layer in menu_parallax_layers:
                layer.draw(screen, (menu_mouse_offset[0], menu_mouse_offset[1]))

            title_surf = title_font.render(menu_title_text, True, MENU_TITLE_COLOR)
            title_rect = title_surf.get_rect(center=(WIDTH // 2, title_y))
            screen.blit(title_surf, title_rect)

            subtitle_surf = subtitle_font.render(menu_subtitle_text, True, MENU_SUBTITLE_COLOR)
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, subtitle_y))
            screen.blit(subtitle_surf, subtitle_rect)

            draw_menu_planet(screen, planet_center, menu_planet_diameter, image=menu_planet_image)

            for btn in menu_buttons:
                btn.draw(screen, menu_button_font, menu_mouse_pos)

            # --- FPS Counter ---
            fps_value = clock.get_fps()
            fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, (140, 180, 220))
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

                trail_history.append((float(r[0]), float(r[1]), time.perf_counter()))

                if logger is not None:
                    log_step_counter += 1
                    vmag = float(np.linalg.norm(v))
                    eps = float(energy_specific(r, v))
                    e_val = float(eccentricity(r, v))

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

                    if not impact_logged and rmag <= EARTH_RADIUS:
                        logger.log_event(
                            [
                                float(t_sim),
                                "impact",
                                rmag,
                                vmag,
                                json.dumps({"penetration": EARTH_RADIUS - rmag, "energy": eps}),
                            ]
                        )
                        impact_logged = True
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

            accumulator = 0.0

        now_time = time.perf_counter()
        cutoff_time = now_time - TRAIL_MAX_DURATION
        while trail_history and trail_history[0][2] < cutoff_time:
            trail_history.popleft()

        # --- Render ---
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
        screen.blit(gradient_bg, (0, 0))
        draw_starfield(screen, camera_center, ppm)

        camera_center_tuple = (float(camera_center[0]), float(camera_center[1]))
        mouse_pos = pygame.mouse.get_pos()
        rmag = float(np.linalg.norm(r))
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)

        if orbit_prediction_points:
            if orbit_prediction_period is None or orbit_prediction_period <= 0.0:
                reveal_fraction = 1.0
            else:
                reveal_fraction = clamp(t_sim / orbit_prediction_period, 0.0, 1.0)
            if reveal_fraction >= 1.0:
                points_iter = orbit_prediction_points
            else:
                max_index = int(len(orbit_prediction_points) * reveal_fraction)
                points_iter = islice(orbit_prediction_points, max_index)
            screen_points = [
                world_to_screen(px, py, ppm, camera_center_tuple)
                for px, py in points_iter
            ]
            if len(screen_points) >= 2:
                draw_orbit_line(screen, ORBIT_PRIMARY_COLOR, screen_points, ORBIT_LINE_WIDTH)

        earth_radius_px = max(1, int(EARTH_RADIUS * ppm))
        draw_earth(screen, earth_screen_pos, earth_radius_px)

        if len(trail_history) >= 2:
            trail_points: list[tuple[tuple[int, int], int]] = []
            for px, py, ts in trail_history:
                age = now_time - ts
                if age > TRAIL_MAX_DURATION:
                    continue
                point = world_to_screen(px, py, ppm, camera_center_tuple)
                alpha = int(255 * clamp(1.0 - age / TRAIL_MAX_DURATION, 0.0, 1.0))
                trail_points.append((point, alpha))
            if len(trail_points) >= 2:
                for (p1, a1), (p2, a2) in zip(trail_points[:-1], trail_points[1:]):
                    alpha = min(a1, a2)
                    if alpha <= 0:
                        continue
                    color = (*TRAIL_COLOR, alpha)
                    pygame.draw.line(trail_surface, color, p1, p2, 2)
                trail_drawn = True
        if trail_drawn:
            screen.blit(trail_surface, (0, 0))

        sat_pos = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
        sat_radius_px = compute_satellite_radius(rmag)
        draw_satellite(screen, sat_pos, earth_screen_pos, sat_radius_px)

        if orbit_markers:
            for marker_type, mx, my, mr in orbit_markers:
                marker_pos = world_to_screen(mx, my, ppm, camera_center_tuple)
                pygame.draw.circle(
                    label_layer,
                    (*LABEL_MARKER_COLOR, LABEL_MARKER_ALPHA),
                    marker_pos,
                    4,
                )
                labels_drawn = True
                distance_px = math.hypot(marker_pos[0] - sat_pos[0], marker_pos[1] - sat_pos[1])
                if distance_px > 120:
                    continue
                label = "Periapsis" if marker_type == "pericenter" else "Apoapsis"
                altitude_km = (mr - EARTH_RADIUS) / 1_000.0
                text = f"{label}: {altitude_km:,.1f} km"
                text_surf = font.render(text, True, LABEL_TEXT_COLOR)
                if HUD_TEXT_ALPHA < 255:
                    text_surf.set_alpha(HUD_TEXT_ALPHA)
                padding = 6
                bg_rect = pygame.Rect(
                    0,
                    0,
                    text_surf.get_width() + padding * 2,
                    text_surf.get_height() + padding * 2,
                )
                direction = -1 if marker_type == "pericenter" else 1
                line_length = 18
                anchor_y = marker_pos[1] + direction * line_length
                bg_rect.center = (
                    marker_pos[0],
                    int(anchor_y + direction * (bg_rect.height / 2 + 6)),
                )
                pygame.draw.line(
                    label_layer,
                    (*LABEL_MARKER_COLOR, int(LABEL_MARKER_ALPHA * 0.6)),
                    marker_pos,
                    (marker_pos[0], anchor_y),
                    2,
                )
                pygame.draw.rect(
                    label_layer,
                    LABEL_BACKGROUND_COLOR,
                    bg_rect,
                    border_radius=10,
                )
                label_layer.blit(text_surf, (bg_rect.left + padding, bg_rect.top + padding))
        if labels_drawn:
            screen.blit(label_layer, (0, 0))

        # HUD
        vmag = float(np.linalg.norm(v))
        eps = energy_specific(r, v)
        e = eccentricity(r, v)
        altitude_km = (rmag - EARTH_RADIUS) / 1_000.0
        hud_lines = [
            f"t {t_sim:,.0f} s   ×{real_time_speed:.1f}",
            f"alt {altitude_km:,.1f} km",
            f"|v| {vmag:,.1f} m/s   e {e:.3f}",
            f"ε {eps: .2e} J/kg",
        ]
        padding_x = 16
        padding_y = 14
        line_height = font.get_linesize()
        hud_width = max(font.size(line)[0] for line in hud_lines) + padding_x * 2
        hud_height = line_height * len(hud_lines) + padding_y
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        pygame.draw.rect(
            hud_surface,
            LABEL_BACKGROUND_COLOR,
            hud_surface.get_rect(),
            border_radius=14,
        )
        for index, line in enumerate(hud_lines):
            text_surf = font.render(line, True, HUD_TEXT_COLOR)
            if HUD_TEXT_ALPHA < 255:
                text_surf.set_alpha(HUD_TEXT_ALPHA)
            hud_surface.blit(
                text_surf,
                (padding_x, int(padding_y / 2) + index * line_height),
            )
        screen.blit(hud_surface, (20, 20))

        for btn in sim_buttons:
            btn.draw(screen, font, mouse_pos)

        fps_value = clock.get_fps()
        fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, HUD_TEXT_COLOR)
        if FPS_TEXT_ALPHA < 255:
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
