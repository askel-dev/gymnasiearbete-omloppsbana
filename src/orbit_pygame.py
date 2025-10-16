# src/orbit_pygame.py
import json
import math
import os
import random
import time
import pygame
from pygame import gfxdraw
from pygame.locals import *  # noqa: F401,F403 - required for constants such as FULLSCREEN
import sys
import numpy as np
from collections import deque
from functools import lru_cache

from logging_utils import RunLogger


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(lerp(color_a[i], color_b[i], t)) for i in range(3))


def create_vertical_gradient(width: int, height: int, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> pygame.Surface:
    surface = pygame.Surface((width, height)).convert()
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = lerp_color(top_color, bottom_color, ratio)
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface


def color_with_alpha(color: tuple[int, ...], alpha: int | None = None) -> tuple[int, int, int, int]:
    if len(color) == 4:
        if alpha is not None:
            return (color[0], color[1], color[2], alpha)
        return (color[0], color[1], color[2], color[3])
    return (color[0], color[1], color[2], 255 if alpha is None else alpha)


def _draw_dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    dash_length: float,
    gap_length: float,
    width: int,
) -> None:
    start_vec = pygame.Vector2(start)
    end_vec = pygame.Vector2(end)
    delta = end_vec - start_vec
    distance = delta.length()
    if distance == 0:
        return
    direction = delta.normalize()
    progress = 0.0
    while progress < distance:
        segment_end = min(progress + dash_length, distance)
        seg_start = start_vec + direction * progress
        seg_end = start_vec + direction * segment_end
        pygame.draw.line(surface, color, seg_start, seg_end, width)
        pygame.draw.aaline(surface, color, seg_start, seg_end)
        progress = segment_end + gap_length


def draw_orbit_path(
    surface: pygame.Surface,
    points: list[tuple[int, int]],
    color: tuple[int, ...],
    width: int = ORBIT_LINE_WIDTH,
    dashed: bool = False,
    dash_length: float = 24.0,
    gap_length: float = 14.0,
) -> None:
    if len(points) < 2:
        return

    orbit_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    rgba = color_with_alpha(color)
    if dashed:
        for start, end in zip(points[:-1], points[1:]):
            _draw_dashed_line(orbit_surface, rgba, start, end, dash_length, gap_length, width)
    else:
        pygame.draw.lines(orbit_surface, rgba, False, points, width)
        pygame.draw.aalines(orbit_surface, rgba, False, points)

    surface.blit(orbit_surface, (0, 0))


def draw_info_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    topleft: tuple[int, int],
) -> None:
    if not lines:
        return

    text_surfaces: list[pygame.Surface] = []
    max_width = 0
    total_height = 0
    for line in lines:
        text_surface = font.render(line, True, HUD_TEXT_COLOR[:3]).convert_alpha()
        text_surface.set_alpha(HUD_TEXT_COLOR[3])
        text_surfaces.append(text_surface)
        max_width = max(max_width, text_surface.get_width())
        total_height += text_surface.get_height()
    total_height += INFO_PANEL_SPACING * (len(text_surfaces) - 1 if text_surfaces else 0)

    panel_width = max_width + INFO_PANEL_PADDING * 2
    panel_height = total_height + INFO_PANEL_PADDING * 2
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    pygame.draw.rect(panel_surface, INFO_PANEL_BG, panel_surface.get_rect(), border_radius=12)

    accent_rect = pygame.Rect(0, 0, 4, panel_height)
    pygame.draw.rect(panel_surface, color_with_alpha(ACCENT_COLOR, 160), accent_rect, border_radius=12)

    y = INFO_PANEL_PADDING
    for text_surface in text_surfaces:
        panel_surface.blit(text_surface, (INFO_PANEL_PADDING + 6, y))
        y += text_surface.get_height() + INFO_PANEL_SPACING

    surface.blit(panel_surface, topleft)


def draw_marker_annotation(
    overlay: pygame.Surface,
    font: pygame.font.Font,
    position: tuple[int, int],
    text: str,
    align_above: bool,
) -> None:
    marker_color = color_with_alpha(MARKER_PRIMARY_COLOR, 220)
    line_color = color_with_alpha(MARKER_PRIMARY_COLOR, 160)

    circle_size = MARKER_RADIUS * 2 + 6
    circle_surface = pygame.Surface((circle_size, circle_size), pygame.SRCALPHA)
    center = circle_size // 2
    gfxdraw.filled_circle(circle_surface, center, center, MARKER_RADIUS, marker_color)
    gfxdraw.aacircle(circle_surface, center, center, MARKER_RADIUS, marker_color)
    overlay.blit(circle_surface, (position[0] - center, position[1] - center))

    line_length = 18
    if align_above:
        line_end = (position[0], position[1] - line_length)
    else:
        line_end = (position[0], position[1] + line_length)
    pygame.draw.line(overlay, line_color, position, line_end, 2)
    pygame.draw.aaline(overlay, line_color, position, line_end)

    text_surface = font.render(text, True, HUD_TEXT_COLOR[:3]).convert_alpha()
    text_surface.set_alpha(HUD_TEXT_COLOR[3])
    label_padding = MARKER_TEXT_PADDING
    label_surface = pygame.Surface(
        (text_surface.get_width() + label_padding * 2, text_surface.get_height() + label_padding * 2),
        pygame.SRCALPHA,
    )
    pygame.draw.rect(label_surface, MARKER_BG_COLOR, label_surface.get_rect(), border_radius=8)
    label_surface.blit(text_surface, (label_padding, label_padding))
    label_rect = label_surface.get_rect()
    if align_above:
        label_rect.midbottom = (line_end[0], line_end[1] - 6)
    else:
        label_rect.midtop = (line_end[0], line_end[1] + 6)
    overlay.blit(label_surface, label_rect)


@lru_cache(maxsize=128)
def _planet_surface(radius: int) -> pygame.Surface:
    diameter = radius * 2
    surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    if radius <= 0:
        return surface

    for r in range(radius, 0, -1):
        t = (radius - r) / max(1, radius)
        if t < 0.5:
            inner_t = t / 0.5
            color = lerp_color(PLANET_CORE_COLOR, PLANET_MID_COLOR, inner_t)
        else:
            inner_t = (t - 0.5) / 0.5
            color = lerp_color(PLANET_MID_COLOR, PLANET_EDGE_COLOR, inner_t)
        gfxdraw.filled_circle(surface, radius, radius, r, (*color, 255))
    gfxdraw.aacircle(surface, radius, radius, radius, (*PLANET_EDGE_COLOR, 255))
    return surface


@lru_cache(maxsize=128)
def _planet_halo_surface(radius: int) -> pygame.Surface:
    if radius <= 0:
        return pygame.Surface((1, 1), pygame.SRCALPHA)
    outer_radius = max(1, int(radius * 3))
    size = outer_radius * 2 + 4
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = size // 2

    def draw_layer(color: tuple[int, int, int, int], layer_radius: int, falloff: float = 2.0) -> None:
        base_alpha = color[3]
        rgb = color[:3]
        for r in range(layer_radius, 0, -1):
            alpha = int(base_alpha * ((r / layer_radius) ** falloff))
            if alpha <= 0:
                continue
            gfxdraw.filled_circle(surface, center, center, r, (*rgb, alpha))
            gfxdraw.aacircle(surface, center, center, r, (*rgb, alpha))

    draw_layer(PLANET_HALO_OUTER_COLOR, outer_radius, falloff=3.2)
    draw_layer(PLANET_HALO_INNER_COLOR, max(1, int(radius * 2)), falloff=2.4)
    return surface


def draw_earth(surface: pygame.Surface, position: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        return
    halo_surface = _planet_halo_surface(radius)
    halo_rect = halo_surface.get_rect(center=position)
    surface.blit(halo_surface, halo_rect)

    planet_surface = _planet_surface(radius)
    planet_rect = planet_surface.get_rect(center=position)
    surface.blit(planet_surface, planet_rect)


@lru_cache(maxsize=32)
def _satellite_surface(radius: int) -> pygame.Surface:
    diameter = radius * 2
    surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    center = radius
    if radius <= 0:
        return surface

    highlight_center = (int(center + radius * 0.35), int(center - radius * 0.35))
    max_dist = max(1, radius)
    for y in range(diameter):
        for x in range(diameter):
            dx = x - center
            dy = y - center
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            t = dist / max_dist
            color = lerp_color(SAT_GRADIENT_START, SAT_GRADIENT_END, t)
            highlight_dx = x - highlight_center[0]
            highlight_dy = y - highlight_center[1]
            highlight_dist = math.hypot(highlight_dx, highlight_dy)
            if highlight_dist < radius * 0.6:
                glow = (1 - (highlight_dist / (radius * 0.6))) ** 2
                color = tuple(min(255, int(c + glow * 40)) for c in color)
            surface.set_at((x, y), (*color, 255))

    rim_radius = max(1, int(radius * 0.9))
    for r in range(rim_radius, max(1, rim_radius - 2), -1):
        gfxdraw.aacircle(surface, center, center, r, SAT_RIM_COLOR)

    return surface


def draw_satellite(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
    planet_pos: tuple[int, int],
) -> None:
    if radius <= 0:
        return

    sat_surface = _satellite_surface(radius)
    rect = sat_surface.get_rect(center=position)

    shadow_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    shadow_center = (shadow_surface.get_width() // 2, shadow_surface.get_height() // 2)
    for r in range(radius * 2, 0, -1):
        alpha = int(SAT_SHADOW_COLOR[3] * ((r / (radius * 2)) ** 2))
        if alpha <= 0:
            continue
        gfxdraw.filled_circle(shadow_surface, shadow_center[0], shadow_center[1], r, (*SAT_SHADOW_COLOR[:3], alpha))
    direction = (planet_pos[0] - position[0], planet_pos[1] - position[1])
    offset_length = max(2, int(math.hypot(*direction) * 0.02))
    if offset_length > radius * 2:
        offset_length = radius * 2
    if math.hypot(*direction) > 0:
        norm = math.hypot(*direction)
        offset = (
            int(direction[0] / norm * offset_length * -0.5),
            int(direction[1] / norm * offset_length * -0.5),
        )
    else:
        offset = (0, 0)
    surface.blit(shadow_surface, shadow_surface.get_rect(center=(position[0] + offset[0], position[1] + offset[1])))
    surface.blit(sat_surface, rect)


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
    color_choices = [
        (220, 230, 255),
        (200, 215, 255),
        (210, 235, 255),
        (185, 210, 240),
    ]
    for _ in range(num_stars):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        radius = random.choice([1, 1, 1, 2])
        base_color = random.choice(color_choices)
        alpha = random.randint(80, 150)
        star_surface = pygame.Surface((radius * 2 + 1, radius * 2 + 1), pygame.SRCALPHA)
        gfxdraw.filled_circle(
            star_surface,
            radius,
            radius,
            radius,
            (*base_color, alpha),
        )
        if radius > 1:
            gfxdraw.aacircle(star_surface, radius, radius, radius, (*base_color, alpha))
        stars.append({
            "pos": (x, y),
            "radius": radius,
            "surface": star_surface,
        })
    return stars


def draw_starfield(surface: pygame.Surface) -> None:
    for star in STARFIELD:
        sx, sy = star["pos"]  # type: ignore[index]
        star_surface: pygame.Surface = star["surface"]  # type: ignore[assignment]
        radius = star["radius"]  # type: ignore[index]
        surface.blit(star_surface, (int(sx) - radius, int(sy) - radius))

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

# =======================
#   RIT- & KONTROLL-SETTINGS
# =======================
# Dessa värden sätts om efter att displayen initierats, men behöver
# startvärden för typkontroller och tooling.
WIDTH, HEIGHT = 1000, 800
SCREEN_VERTICAL_OFFSET = 0
BG_COLOR_TOP = (11, 16, 32)
BG_COLOR_BOTTOM = (10, 14, 25)

ACCENT_COLOR = (46, 209, 195)

PLANET_CORE_COLOR = (255, 183, 77)
PLANET_MID_COLOR = (240, 98, 146)
PLANET_EDGE_COLOR = (124, 77, 255)
PLANET_HALO_INNER_COLOR = (124, 77, 255, int(255 * 0.10))
PLANET_HALO_OUTER_COLOR = (46, 209, 195, int(255 * 0.08))

SAT_GRADIENT_START = (65, 224, 162)
SAT_GRADIENT_END = (42, 170, 226)
SAT_RIM_COLOR = (232, 252, 255, 90)
SAT_SHADOW_COLOR = (9, 12, 24, 130)
SAT_BASE_RADIUS = 6
SAT_SCALE_MIN = 0.9
SAT_SCALE_MAX = 1.2

HUD_TEXT_COLOR = (234, 241, 255, int(255 * 0.85))

PREDICTION_COLOR_PRIMARY = ACCENT_COLOR
PREDICTION_COLOR_SECONDARY = (46, 209, 195, int(255 * 0.7))
ORBIT_LINE_WIDTH = 2

BUTTON_COLOR = (17, 26, 45, int(255 * 0.6))
BUTTON_HOVER_COLOR = (30, 46, 74, int(255 * 0.7))
BUTTON_TEXT_COLOR = HUD_TEXT_COLOR
BUTTON_PANEL_COLOR = (8, 12, 24, int(255 * 0.45))

MENU_TITLE_COLOR = (234, 241, 255)
MENU_SUBTITLE_COLOR = (160, 180, 210)

MARKER_PRIMARY_COLOR = ACCENT_COLOR
MARKER_BG_COLOR = (12, 18, 36, int(255 * 0.2))
MARKER_TEXT_PADDING = 6
MARKER_RADIUS = 4
MARKER_MAX_DISTANCE = 400_000.0
MARKER_MAX_ACTIVE = 2

FPS_TEXT_COLOR = (234, 241, 255, int(255 * 0.6))

INFO_PANEL_BG = (10, 16, 30, int(255 * 0.35))
INFO_PANEL_PADDING = 12
INFO_PANEL_SPACING = 6

TRAIL_DURATION = 1.0
TRAIL_MAX_POINTS = 240

STARFIELD: list[dict[str, object]] = []

def compute_pixels_per_meter(width: int, height: int) -> float:
    return 0.25 * (min(width, height) / (2.0 * np.linalg.norm(R0)))


def update_display_metrics(width: int, height: int) -> None:
    global WIDTH, HEIGHT, PIXELS_PER_METER, SCREEN_VERTICAL_OFFSET

    WIDTH = width
    HEIGHT = height
    SCREEN_VERTICAL_OFFSET = int(height * 0.06)
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
    sy = HEIGHT // 2 - int((y - cy) * ppm) + SCREEN_VERTICAL_OFFSET
    return sx, sy

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_orbit_prediction(r_init: np.ndarray, v_init: np.ndarray) -> tuple[float | None, list[tuple[float, float]]]:
    eps = energy_specific(r_init, v_init)
    if eps >= 0.0:
        return None, []

    a = -MU / (2.0 * eps)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    num_samples = max(360, int(period / DT_PHYS))
    dt = period / num_samples

    r = r_init.copy()
    v = v_init.copy()
    points: list[tuple[float, float]] = []
    for _ in range(num_samples + 1):
        points.append((float(r[0]), float(r[1])))
        r, v = rk4_step(r, v, dt)

    return period, points


class Button:
    """Simple rectangular button with hover feedback and callbacks."""

    def __init__(self, rect, text, callback, text_getter=None):
        self.rect = pygame.Rect(rect)
        self._text = text
        self._callback = callback
        self._text_getter = text_getter
        self._cached_text_surface: pygame.Surface | None = None
        self._cached_text: str | None = None
        self._cached_font_id: int | None = None

    def get_text(self):
        if self._text_getter is not None:
            return self._text_getter()
        return self._text

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        color = BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        button_surface.fill((0, 0, 0, 0))
        pygame.draw.rect(
            button_surface,
            color_with_alpha(color),
            button_surface.get_rect(),
            border_radius=8,
        )
        surface.blit(button_surface, self.rect.topleft)
        text_value = self.get_text()
        font_id = id(font)
        if (
            self._cached_text_surface is None
            or text_value != self._cached_text
            or font_id != self._cached_font_id
        ):
            text_surface = font.render(text_value, True, BUTTON_TEXT_COLOR[:3])
            text_surface = text_surface.convert_alpha()
            text_surface.set_alpha(BUTTON_TEXT_COLOR[3])
            self._cached_text_surface = text_surface
            self._cached_text = text_value
            self._cached_font_id = font_id
        text_surf = self._cached_text_surface
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
    pygame.display.set_caption("Omloppsbana i realtid (Pygame + RK4)")
    font_family = "dejavusansmono"
    font_fps = pygame.font.SysFont(font_family, 14)

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

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(font_family, 18)
    title_font = pygame.font.SysFont(font_family, 36, bold=True)
    subtitle_font = pygame.font.SysFont(font_family, 22)

    gradient_bg = create_vertical_gradient(WIDTH, HEIGHT, BG_COLOR_TOP, BG_COLOR_BOTTOM)
    global STARFIELD
    if not STARFIELD:
        STARFIELD = generate_starfield(230)

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

    orbit_markers: deque[tuple[str, float, float, float, float]] = deque(maxlen=20)
    satellite_trail: deque[tuple[float, float, float]] = deque(maxlen=TRAIL_MAX_POINTS)

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
        nonlocal camera_center, orbit_markers, camera_target, satellite_trail
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
        satellite_trail.clear()
        satellite_trail.append((float(r[0]), float(r[1]), float(t_sim)))
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

    menu_button_width = 260
    menu_button_height = 56
    menu_button_x = WIDTH // 2 - menu_button_width // 2
    menu_button_y = HEIGHT // 2 - menu_button_height
    menu_button_gap = 20

    menu_buttons = [
        Button(
            (menu_button_x, menu_button_y, menu_button_width, menu_button_height),
            "Start Simulation",
            start_simulation,
        ),
        Button(
            (
                menu_button_x,
                menu_button_y + menu_button_height + menu_button_gap,
                menu_button_width,
                menu_button_height,
            ),
            "Quit",
            quit_app,
        ),
    ]

    button_width = 168
    button_height = 44
    button_gap = 12
    button_x = 24
    button_y_start = 180

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
        Button((button_x, button_y_start, button_width, button_height), "Pause", toggle_pause, lambda: "Resume" if paused else "Pause"),
        Button((button_x, button_y_start + (button_height + button_gap), button_width, button_height), "Reset", reset_and_continue),
        Button(
            (button_x, button_y_start + 2 * (button_height + button_gap), button_width, button_height),
            "Slower",
            slow_down,
        ),
        Button(
            (button_x, button_y_start + 3 * (button_height + button_gap), button_width, button_height),
            "Faster",
            speed_up,
        ),
        Button(
            (button_x, button_y_start + 4 * (button_height + button_gap), button_width, button_height),
            "Camera",
            toggle_camera,
            lambda: (
                "Camera: Earth"
                if camera_mode == "earth"
                else "Camera: Sat" if camera_mode == "satellite" else "Camera: Free"
            ),
        ),
    ]

    button_panel_height = len(sim_buttons) * button_height + (len(sim_buttons) - 1) * button_gap + 32
    sim_button_panel_rect = pygame.Rect(
        button_x - 16,
        button_y_start - 16,
        button_width + 32,
        button_panel_height,
    )

    def is_over_button(pos: tuple[int, int]) -> bool:
        if state == "menu":
            return any(btn.rect.collidepoint(pos) for btn in menu_buttons)
        if state == "running":
            if sim_button_panel_rect.collidepoint(pos):
                return True
            return any(btn.rect.collidepoint(pos) for btn in sim_buttons)
        return False

    # ========= LOOP =========
    while True:
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
            screen.blit(gradient_bg, (0, 0))
            draw_starfield(screen)
            title_surf = title_font.render("Omloppsbana i realtid", True, MENU_TITLE_COLOR).convert_alpha()
            subtitle_surf = subtitle_font.render("Välj ett alternativ för att starta", True, MENU_SUBTITLE_COLOR).convert_alpha()
            title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 160))
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 110))
            screen.blit(title_surf, title_rect)
            screen.blit(subtitle_surf, subtitle_rect)

            menu_mouse_pos = pygame.mouse.get_pos()
            for btn in menu_buttons:
                btn.draw(screen, font, menu_mouse_pos)

            # --- FPS Counter ---
            fps_value = clock.get_fps()
            fps_surface = font_fps.render(f"FPS: {fps_value:5.1f}", True, FPS_TEXT_COLOR[:3]).convert_alpha()
            fps_surface.set_alpha(FPS_TEXT_COLOR[3])
            text_rect = fps_surface.get_rect(bottomright=(WIDTH - 24, HEIGHT - 24))
            screen.blit(fps_surface, text_rect)


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
                satellite_trail.append((float(r[0]), float(r[1]), float(t_sim)))
                while satellite_trail and t_sim - satellite_trail[0][2] > TRAIL_DURATION:
                    satellite_trail.popleft()
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
                    orbit_markers.append((event_type, float(r[0]), float(r[1]), rmag, float(t_sim)))

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

        # --- Render ---
        # Smooth zoom mot mål
        ppm += (ppm_target - ppm) * 0.1
        ppm = clamp(ppm, MIN_PPM, MAX_PPM)

        # Kamera-targets
        if camera_mode == "earth":
            camera_target[:] = (0.0, 0.0)
        elif camera_mode == "satellite":
            camera_target[:] = (r[0], r[1])
        else:
            camera_target[:] = camera_center
        camera_center += (camera_target - camera_center) * 0.1

        # Bakgrund
        screen.blit(gradient_bg, (0, 0))
        draw_starfield(screen)

        camera_center_tuple = (float(camera_center[0]), float(camera_center[1]))
        mouse_pos = pygame.mouse.get_pos()

        # Jorden
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)
        earth_radius_px = max(1, int(EARTH_RADIUS * ppm))
        draw_earth(screen, earth_screen_pos, earth_radius_px)

        rmag = float(np.linalg.norm(r))
        altitude = max(0.0, rmag - EARTH_RADIUS)
        vmag = float(np.linalg.norm(v))
        eps = float(energy_specific(r, v))
        e = float(eccentricity(r, v))

        if orbit_prediction_points:
            if orbit_prediction_period is None or orbit_prediction_period <= 0.0:
                reveal_fraction = 1.0
            else:
                reveal_fraction = clamp(t_sim / orbit_prediction_period, 0.0, 1.0)
            if reveal_fraction >= 1.0:
                partial_points = orbit_prediction_points
            else:
                max_index = int(len(orbit_prediction_points) * reveal_fraction)
                partial_points = orbit_prediction_points[:max_index]
            if len(partial_points) >= 2:
                screen_points = [
                    world_to_screen(px, py, ppm, camera_center_tuple)
                    for px, py in partial_points
                ]
                draw_orbit_path(screen, screen_points, PREDICTION_COLOR_PRIMARY, width=ORBIT_LINE_WIDTH)

                if reveal_fraction < 1.0 and len(partial_points) >= 2:
                    start_index = max(0, len(partial_points) - 1)
                    remaining_points = orbit_prediction_points[start_index:]
                    if len(remaining_points) >= 2:
                        screen_remaining = [
                            world_to_screen(px, py, ppm, camera_center_tuple)
                            for px, py in remaining_points
                        ]
                        draw_orbit_path(
                            screen,
                            screen_remaining,
                            PREDICTION_COLOR_SECONDARY,
                            width=ORBIT_LINE_WIDTH,
                            dashed=True,
                        )

        if len(satellite_trail) >= 2:
            trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            trail_points = list(satellite_trail)
            for (x1, y1, _), (x2, y2, t2) in zip(trail_points[:-1], trail_points[1:]):
                age = clamp((t_sim - t2) / TRAIL_DURATION, 0.0, 1.0)
                alpha = int(180 * (1.0 - age))
                if alpha <= 0:
                    continue
                start_pt = world_to_screen(x1, y1, ppm, camera_center_tuple)
                end_pt = world_to_screen(x2, y2, ppm, camera_center_tuple)
                color = color_with_alpha(ACCENT_COLOR, alpha)
                pygame.draw.line(trail_surface, color, start_pt, end_pt, 2)
                pygame.draw.aaline(trail_surface, color, start_pt, end_pt)
            screen.blit(trail_surface, (0, 0))

        sat_pos = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
        scale_factor = SAT_SCALE_MIN + (SAT_SCALE_MAX - SAT_SCALE_MIN) * (1.0 - math.exp(-altitude / 15_000_000.0))
        scale_factor = clamp(scale_factor, SAT_SCALE_MIN, SAT_SCALE_MAX)
        sat_radius_px = max(3, int(SAT_BASE_RADIUS * scale_factor))
        draw_satellite(screen, sat_pos, sat_radius_px, earth_screen_pos)

        marker_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        active_markers = 0
        for marker_type, mx, my, mr, mt in reversed(orbit_markers):
            if active_markers >= MARKER_MAX_ACTIVE:
                break
            distance_to_marker = math.hypot(r[0] - mx, r[1] - my)
            if distance_to_marker > MARKER_MAX_DISTANCE:
                continue
            marker_pos = world_to_screen(mx, my, ppm, camera_center_tuple)
            label = "Periapsis" if marker_type == "pericenter" else "Apoapsis"
            altitude_km = (mr - EARTH_RADIUS) / 1_000.0
            text = f"{label}: {altitude_km:,.1f} km"
            align_above = marker_pos[1] > sat_pos[1]
            draw_marker_annotation(marker_overlay, font, marker_pos, text, align_above)
            active_markers += 1
        if active_markers > 0:
            screen.blit(marker_overlay, (0, 0))

        info_lines = [
            f"t {t_sim:,.0f} s   ×{real_time_speed:.2f}",
            f"alt {altitude/1_000:.1f} km   vel {vmag:,.1f} m/s",
            f"ecc {e:.4f}   energy {eps: .3e} J/kg",
        ]
        draw_info_panel(screen, font, info_lines, (24, 24))

        fps_value = clock.get_fps()
        fps_surface = font_fps.render(f"FPS: {fps_value:5.1f}", True, FPS_TEXT_COLOR[:3]).convert_alpha()
        fps_surface.set_alpha(FPS_TEXT_COLOR[3])
        fps_rect = fps_surface.get_rect(bottomright=(WIDTH - 24, HEIGHT - 24))
        screen.blit(fps_surface, fps_rect)

        button_panel_surface = pygame.Surface(sim_button_panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(button_panel_surface, BUTTON_PANEL_COLOR, button_panel_surface.get_rect(), border_radius=18)
        pygame.draw.rect(
            button_panel_surface,
            color_with_alpha(ACCENT_COLOR, 90),
            button_panel_surface.get_rect(),
            width=1,
            border_radius=18,
        )
        screen.blit(button_panel_surface, sim_button_panel_rect.topleft)

        for btn in sim_buttons:
            btn.draw(screen, font, mouse_pos)

        pygame.display.flip()
        clock.tick()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()
