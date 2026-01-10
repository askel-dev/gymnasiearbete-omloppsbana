# src/orbit_pygame.py
"""Simplified orbital mechanics simulation using Pygame."""

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

from physics import G, M, EARTH_RADIUS, rk4_step
from logging_utils import RunLogger

# =======================
#   PHYSICS CONSTANTS
# =======================
MU = G * M

# Initial conditions
R0 = np.array([7_000_000.0, 0.0])  # m
V0 = np.array([0.0, 7_600.0])      # m/s

# =======================
#   SCENARIOS
# =======================
@dataclass(frozen=True)
class Scenario:
    """Preset initial conditions for different orbit types."""
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

# =======================
#   DISPLAY SETTINGS
# =======================
WIDTH, HEIGHT = 1000, 800
WINDOWED_DEFAULT_SIZE = (WIDTH, HEIGHT)
FULLSCREEN_ENABLED = False      # Start in fullscreen by default

# Colors
BACKGROUND_COLOR = (0, 34, 72)
PLANET_COLOR = (255, 183, 77)
SATELLITE_COLOR = (255, 255, 255)
SATELLITE_PIXEL_RADIUS = 6
HUD_TEXT_COLOR = (234, 241, 255)
HUD_SHADOW_COLOR = (10, 15, 30, 120)
ORBIT_PRIMARY_COLOR = (255, 255, 255, 180)
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
BUTTON_RADIUS = 18
MENU_SUBTITLE_COLOR = (180, 198, 228)
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
#   EARTH SPRITE
# =======================
_EARTH_SPRITE_CACHE: dict[int, pygame.Surface] = {}
# Generate a random seed once per session so Earth is consistent across zooms
# but different every time the program runs.
_EARTH_SEED = random.randint(0, 999999)


def _generate_procedural_earth(diameter: int) -> pygame.Surface:
    """Generates a procedural Earth surface with continents and clouds."""
    # Use a fixed seed so the Earth looks consistent across different zoom levels (diameters)
    rng = random.Random(_EARTH_SEED)
    
    radius = diameter / 2.0
    # Create the texture surface (Ocean + Continents + Clouds)
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    texture.fill((0, 105, 148))  # Deep Ocean Blue
    
    # Draw Continents (Random polygons/circles)
    num_continents = rng.randint(5, 9)
    for _ in range(num_continents):
        # Use random() (0.0-1.0) to ensure resolution independence
        cx = int(rng.random() * diameter)
        cy = int(rng.random() * diameter)
        
        # Scale continent size relative to Earth radius
        # Radius factor between 0.3 and 0.8
        r_factor = 0.3 + rng.random() * 0.5
        cr = int(radius * r_factor)
        
        # Color variation
        color = (34, 139, 34)  # Forest Green
        if rng.random() > 0.55:
             color = (160, 82, 45)  # Sienna/Brown
        
        pygame.draw.circle(texture, color, (cx, cy), cr)
        
    # Draw Clouds
    num_clouds = rng.randint(10, 18)
    for _ in range(num_clouds):
        cx = int(rng.random() * diameter)
        cy = int(rng.random() * diameter)
        
        w_factor = 0.3 + rng.random() * 0.5  # 0.3 to 0.8 of radius
        h_factor = 0.15 + rng.random() * 0.25 # 0.15 to 0.4 of radius
        
        cw = int(radius * w_factor)
        ch = int(radius * h_factor)
        
        # Draw cloud on a temporary surface to handle alpha
        cloud_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        # White with transparency
        pygame.draw.ellipse(cloud_surf, (255, 255, 255, 150), cloud_surf.get_rect())
        texture.blit(cloud_surf, (cx - cw // 2, cy - ch // 2))

    # Create a mask to clip everything into a circle
    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (int(radius), int(radius)), int(radius))

    # Apply the mask
    # BLEND_RGBA_MULT will multiply the alpha channels, clipping the texture to the circle
    texture.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    
    # Optional: Atmosphere glow/border to make it look nicer
    pygame.draw.circle(texture, (100, 200, 255), (int(radius), int(radius)), int(radius), 1)

    return texture


def _get_scaled_earth_sprite(diameter: int) -> pygame.Surface:
    if diameter <= 0:
        raise ValueError("Earth sprite diameter must be positive")
    
    cached = _EARTH_SPRITE_CACHE.get(diameter)
    if cached is not None:
        return cached
    
    # Generate new procedural Earth
    generated = _generate_procedural_earth(diameter)
    _EARTH_SPRITE_CACHE[diameter] = generated
    return generated


# =======================
#   HELPER FUNCTIONS
# =======================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def compute_pixels_per_meter(width: int, height: int) -> float:
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


def world_to_screen(x, y, ppm, camera_center=(0.0, 0.0)):
    cx, cy = camera_center
    sx = WIDTH // 2 + int((x - cx) * ppm)
    sy = HEIGHT // 2 - int((y - cy) * ppm)
    return sx, sy


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
def draw_earth(surface: pygame.Surface, position: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        return
    diameter = radius * 2
    
    # Optimization: If Earth is massive (zoomed in very close),
    # just draw a simple circle to avoid generating/blitting huge textures
    # which would cause lag or Out Of Memory errors.
    if diameter > 2048:
        pygame.draw.circle(surface, (0, 105, 148), position, radius)
        return

    sprite = _get_scaled_earth_sprite(diameter)
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
        color = BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surface,
            color,
            button_surface.get_rect(),
            border_radius=BUTTON_RADIUS,
        )
        surface.blit(button_surface, self.rect.topleft)
        text_value = self.get_text()
        text_surf = font.render(text_value, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._callback()


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
#   MAIN FUNCTION
# =======================
def main():
    pygame.init()
    pygame.display.set_caption("Gymnasiearbete - Simulering av omloppsbana")
    
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

    # Scenario setup
    current_scenario_key = DEFAULT_SCENARIO_KEY
    scenario_flash_text: str | None = None
    scenario_flash_time = 0.0

    scenario_shortcut_map: dict[int, str] = {}
    number_key_codes = [
        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
        pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9,
    ]
    keypad_key_codes = [
        pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP4, pygame.K_KP5,
        pygame.K_KP6, pygame.K_KP7, pygame.K_KP8, pygame.K_KP9,
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

    # Simulation state
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
        r = R0.copy()
        v = scenario_velocity_vector()
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
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(r, v)
        impact_info = None
        shock_ring_start = None
        impact_freeze_time = None
        impact_overlay_reveal_time = None
        impact_overlay_visible_since = None

    state = "menu"  # "menu", "running", "impact"
    escape_radius_limit = ESCAPE_RADIUS_FACTOR * float(np.linalg.norm(R0))

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

    def render_marker_label(
        surface: pygame.Surface,
        marker_type: str,
        marker_pos: tuple[int, int],
        radius_world: float,
    ) -> None:
        label = marker_display_name(marker_type)
        altitude_km = (radius_world - EARTH_RADIUS) / 1_000.0
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

    # ========= MAIN LOOP =========
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
                    quit_app()
                    continue
                if event.key == pygame.K_F11:
                    toggle_fullscreen_mode()
                    continue
                if event.key in scenario_shortcut_map:
                    set_scenario(scenario_shortcut_map[event.key])
                    if state == "impact":
                        state = "running"
                        paused = False
                        init_run_logging()
                    elif state == "menu":
                        start_simulation()
                    continue
                if event.key == pygame.K_g:
                    grid_overlay_enabled = not grid_overlay_enabled
                    continue
                
                # Menu: Press Space to start
                if state == "menu":
                    if event.key == pygame.K_SPACE:
                        start_simulation()
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
                    elif event.key == pygame.K_n:
                        load_next_scenario()
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
                    ppm_target = clamp(ppm_target * zoom_factor, MIN_PPM, MAX_PPM)
            
            if state == "running":
                for btn in sim_buttons:
                    btn.handle_event(event)

        # === MENU STATE ===
        if state == "menu":
            screen.fill(BACKGROUND_COLOR)
            
            # Title
            title_text = "SIMULERING AV OMLOPPSBANA"
            title_surf = title_font.render(title_text, True, HUD_TEXT_COLOR)
            title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 3))
            screen.blit(title_surf, title_rect)
            
            # Subtitle
            subtitle_text = "av Axel Jönsson"
            subtitle_surf = font.render(subtitle_text, True, MENU_SUBTITLE_COLOR)
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, HEIGHT // 3 + 50))
            screen.blit(subtitle_surf, subtitle_rect)
            
            # Press Space prompt
            prompt_text = "Press SPACE to start"
            prompt_surf = font.render(prompt_text, True, HUD_TEXT_COLOR)
            prompt_rect = prompt_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            screen.blit(prompt_surf, prompt_rect)
            
            # Scenario panel
            panel_lines = [scenario_panel_title, *scenario_help_lines]
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
                text_surf = scenario_font.render(line, True, color)
                panel_surface.blit(text_surf, (panel_padding, y))
                y += line_height
            panel_rect = panel_surface.get_rect()
            panel_rect.topleft = (20, HEIGHT - panel_height - 20)
            screen.blit(panel_surface, panel_rect)
            
            # FPS
            fps_value = clock.get_fps()
            fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, (140, 180, 220))
            fps_rect = fps_text.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
            screen.blit(fps_text, fps_rect)
            
            pygame.display.flip()
            clock.tick(60)
            continue

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
                r, v = rk4_step(r, v, dt_step)
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
                eps = float(energy_specific(r, v))
                e_val = float(eccentricity(r, v))
                impact_triggered = not impact_logged and rmag <= EARTH_RADIUS

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
                            json.dumps({"penetration": EARTH_RADIUS - rmag, "energy": eps}),
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
            camera_target[:] = camera_center
        camera_center += (camera_target - camera_center) * 0.1

        # === RENDER ===
        screen.fill(BACKGROUND_COLOR)

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
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)

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

        earth_radius_px = max(1, int(EARTH_RADIUS * ppm))
        draw_earth(screen, earth_screen_pos, earth_radius_px)

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
                base_ring_radius = max(4, compute_satellite_radius(EARTH_RADIUS) * 2)
                ring_radius_px = max(
                    base_ring_radius,
                    int(base_ring_radius + earth_radius_px * SHOCK_RING_EXPANSION_FACTOR * progress),
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
        eps = energy_specific(r, v)
        e = eccentricity(r, v)
        altitude_km = (rmag - EARTH_RADIUS) / 1_000.0
        scenario = get_current_scenario()
        hud_entries = [
            f"Scenario: {scenario.name}",
            f"t {t_sim:,.0f} s   ×{real_time_speed:.1f}",
            f"alt {altitude_km:,.1f} km",
            f"|v| {vmag:,.1f} m/s   e {e:.3f}",
            f"ε {eps: .2e} J/kg",
        ]
        padding_x = 16
        padding_y = 14
        line_height = font.get_linesize()
        hud_width = max(font.size(line)[0] for line in hud_entries) + padding_x * 2
        hud_height = line_height * len(hud_entries) + padding_y
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        pygame.draw.rect(hud_surface, LABEL_BACKGROUND_COLOR, hud_surface.get_rect(), border_radius=14)
        for index, line in enumerate(hud_entries):
            text_surf = font.render(line, True, HUD_TEXT_COLOR)
            hud_surface.blit(text_surf, (padding_x, int(padding_y / 2) + index * line_height))
        screen.blit(hud_surface, (20, 20))

        # Sim buttons
        for btn in sim_buttons:
            btn.draw(screen, font, mouse_pos)

        # Scenario panel
        panel_lines = [scenario_panel_title, *scenario_help_lines]
        panel_padding = 14
        line_height = scenario_font.get_linesize()
        panel_width = max(scenario_font.size(line)[0] for line in panel_lines) + panel_padding * 2
        panel_height = line_height * len(panel_lines) + panel_padding * 2
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, LABEL_BACKGROUND_COLOR, panel_surface.get_rect(), border_radius=12)
        y = panel_padding
        for idx, line in enumerate(panel_lines):
            if idx == 0:
                color = HUD_TEXT_COLOR
            else:
                scenario_key = SCENARIO_DISPLAY_ORDER[idx - 1]
                color = HUD_TEXT_COLOR if scenario_key == current_scenario_key else MENU_SUBTITLE_COLOR
            text_surf = scenario_font.render(line, True, color)
            panel_surface.blit(text_surf, (panel_padding, y))
            y += line_height
        panel_rect = panel_surface.get_rect()
        panel_rect.topleft = (20, HEIGHT - panel_height - 20)
        screen.blit(panel_surface, panel_rect)

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
                (f"Impact angle = {impact_angle_value:.1f}° from horizon", IMPACT_TEXT_COLOR),
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
