# src/orbit_pygame.py
import json
import math
import random
import time
import pygame
import sys
import numpy as np
from collections import deque
from functools import lru_cache

from logging_utils import RunLogger


def create_vertical_gradient(width: int, height: int, top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> pygame.Surface:
    surface = pygame.Surface((width, height))
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(
            int(top_color[i] + (bottom_color[i] - top_color[i]) * ratio)
            for i in range(3)
        )
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface.convert()


def draw_text_with_shadow(surface: pygame.Surface, font: pygame.font.Font, text: str, position: tuple[int, int]) -> None:
    shadow_color = HUD_SHADOW_COLOR[:3]
    shadow_surf = font.render(text, True, shadow_color)
    if len(HUD_SHADOW_COLOR) == 4:
        shadow_surf.set_alpha(HUD_SHADOW_COLOR[3])
    text_surf = font.render(text, True, HUD_TEXT_COLOR)
    x, y = position
    surface.blit(shadow_surf, (x + 2, y + 2))
    surface.blit(text_surf, position)


def create_radial_surface(
    radius: int,
    inner_color: tuple[int, ...],
    outer_color: tuple[int, ...],
) -> pygame.Surface:
    size = radius * 2
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = radius
    inner_rgba = list(inner_color) + [255] if len(inner_color) == 3 else list(inner_color)
    outer_rgba = list(outer_color) + [0] if len(outer_color) == 3 else list(outer_color)
    for r in range(radius, 0, -1):
        t = r / radius
        color = [
            int(inner_rgba[i] * t + outer_rgba[i] * (1 - t))
            for i in range(4)
        ]
        pygame.draw.circle(surface, color, (cx, cy), r)
    return surface


@lru_cache(maxsize=128)
def _earth_surface(radius: int) -> pygame.Surface:
    glow_radius = max(radius + 20, int(radius * 1.4))
    surface_size = glow_radius * 2
    surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
    glow_surface = create_radial_surface(glow_radius, EARTH_CORE_COLOR + (255,), EARTH_GLOW_COLOR + (0,))
    surface.blit(glow_surface, (0, 0))
    pygame.draw.circle(surface, EARTH_CORE_COLOR, (glow_radius, glow_radius), radius)
    return surface


def draw_earth(surface: pygame.Surface, position: tuple[int, int], radius: int) -> None:
    if radius <= 0:
        return
    earth_surface = _earth_surface(radius)
    rect = earth_surface.get_rect(center=position)
    surface.blit(earth_surface, rect)


@lru_cache(maxsize=1)
def _satellite_surface() -> pygame.Surface:
    halo_radius = 14
    surface = pygame.Surface((halo_radius * 2, halo_radius * 2), pygame.SRCALPHA)
    halo_surface = create_radial_surface(halo_radius, SAT_HALO_COLOR, (0, 0, 0, 0))
    surface.blit(halo_surface, (0, 0))
    pygame.draw.circle(surface, SAT_COLOR, (halo_radius, halo_radius), 5)
    return surface


def draw_satellite(surface: pygame.Surface, position: tuple[int, int]) -> None:
    sat_surface = _satellite_surface()
    rect = sat_surface.get_rect(center=position)
    surface.blit(sat_surface, rect)


@lru_cache(maxsize=128)
def get_earth_glow_surface(radius: int) -> pygame.Surface:
    radius = max(1, radius)
    size = radius * 2
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(surface, EARTH_GLOW_OVERLAY_COLOR, (radius, radius), radius)
    return surface


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


def generate_starfield(num_stars: int) -> list[dict[str, float | tuple[int, int]]]:
    stars: list[dict[str, float | tuple[int, int]]] = []
    for _ in range(num_stars):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        size = random.choice([1, 1, 1, 2])
        brightness = random.randint(140, 220)
        parallax = random.uniform(0.05, 0.35)
        stars.append({
            "pos": (x, y),
            "size": size,
            "brightness": brightness,
            "parallax": parallax,
        })
    return stars


def draw_starfield(surface: pygame.Surface, camera_center: np.ndarray, ppm: float) -> None:
    offset_x = camera_center[0] * ppm
    offset_y = camera_center[1] * ppm
    for star in STARFIELD:
        base_x, base_y = star["pos"]  # type: ignore[index]
        parallax = star["parallax"]  # type: ignore[index]
        sx = int((base_x - offset_x * parallax) % WIDTH)
        sy = int((base_y + offset_y * parallax) % HEIGHT)
        brightness = star["brightness"]  # type: ignore[index]
        color = (brightness, brightness, brightness)
        size = star["size"]  # type: ignore[index]
        surface.fill(color, (sx, sy, size, size))

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
TRAIL_FADE_ALPHA = 12

# =======================
#   RIT- & KONTROLL-SETTINGS
# =======================
WIDTH, HEIGHT = 1000, 800
BG_COLOR_TOP = (5, 10, 25)
BG_COLOR_BOTTOM = (10, 30, 60)
EARTH_CORE_COLOR = (70, 170, 255)
EARTH_GLOW_COLOR = (30, 110, 200)
SAT_COLOR = (255, 255, 230)
SAT_HALO_COLOR = (255, 200, 120, 120)
TRAIL_COLOR = (140, 240, 200)
TRAIL_COLOR_ALPHA = (140, 240, 200, 220)
HUD_TEXT_COLOR = (245, 245, 245)
HUD_SHADOW_COLOR = (0, 0, 0, 160)
PREDICTION_COLOR = (120, 180, 255)
VEL_COLOR = (255, 120, 120)
BUTTON_COLOR = (35, 55, 90)
BUTTON_HOVER_COLOR = (70, 110, 160)
BUTTON_TEXT_COLOR = (240, 245, 250)
MENU_TITLE_COLOR = (220, 230, 255)
MENU_SUBTITLE_COLOR = (150, 165, 200)
PERICENTER_COLOR = (255, 180, 120)
APOCENTER_COLOR = (120, 200, 255)

EARTH_GLOW_OVERLAY_COLOR = (80, 180, 255, 40)
EARTH_GLOW_CACHE_STEP = 4
EARTH_SCALE_CACHE_PRECISION = 200
EARTH_SCALE_CACHE_MAX = 256

STARFIELD: list[dict[str, float | tuple[int, int]]] = []

PIXELS_PER_METER = 0.25 * (min(WIDTH, HEIGHT) / (2.0 * np.linalg.norm(R0)))
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
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        text_value = self.get_text()
        font_id = id(font)
        if (
            self._cached_text_surface is None
            or text_value != self._cached_text
            or font_id != self._cached_font_id
        ):
            self._cached_text_surface = font.render(text_value, True, BUTTON_TEXT_COLOR)
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
    font_fps = pygame.font.SysFont("consolas", 18)
    pygame.display.set_caption("Omloppsbana i realtid (Pygame + RK4)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    earth_img = pygame.image.load("assets/earth_sprite.png").convert_alpha()
    earth_img_width = earth_img.get_width()
    earth_img_height = earth_img.get_height()
    earth_scale_cache: dict[int, pygame.Surface] = {}
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    title_font = pygame.font.SysFont("consolas", 40, bold=True)
    subtitle_font = pygame.font.SysFont("consolas", 24)

    gradient_bg = create_vertical_gradient(WIDTH, HEIGHT, BG_COLOR_TOP, BG_COLOR_BOTTOM)
    global STARFIELD
    if not STARFIELD:
        STARFIELD = generate_starfield(450)

    # Simuleringsstate
    r = R0.copy()
    v = V0.copy()
    t_sim = 0.0
    paused = False
    show_trail = True
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

    trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    trail_surface.fill((0, 0, 0, 0))
    trail_prev_screen_pos: tuple[int, int] | None = None
    trail_last_ppm = ppm
    trail_last_camera = camera_center.copy()

    orbit_markers: deque[tuple[str, float, float]] = deque(maxlen=20)

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
        nonlocal trail_prev_screen_pos, trail_last_ppm, trail_last_camera, camera_center, orbit_markers, camera_target
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
        trail_surface.fill((0, 0, 0, 0))
        trail_prev_screen_pos = None
        trail_last_ppm = ppm
        orbit_markers.clear()
        camera_center[:] = (0.0, 0.0)
        camera_target[:] = (0.0, 0.0)
        camera_mode = "earth"
        is_dragging_camera = False
        trail_last_camera = camera_center.copy()
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

    button_width = 140
    button_height = 40
    button_gap = 10

    def toggle_pause():
        nonlocal paused
        paused = not paused

    def toggle_trail():
        nonlocal show_trail
        show_trail = not show_trail

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
            "Trail",
            toggle_trail,
            lambda: "Trail: On" if show_trail else "Trail: Off",
        ),
        Button(
            (20, 20 + 3 * (button_height + button_gap), button_width, button_height),
            "Slower",
            slow_down,
        ),
        Button(
            (20, 20 + 4 * (button_height + button_gap), button_width, button_height),
            "Faster",
            speed_up,
        ),
        Button(
            (20, 20 + 5 * (button_height + button_gap), button_width, button_height),
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
                    elif event.key == pygame.K_t:
                        show_trail = not show_trail
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
            title_surf = title_font.render("Omloppsbana i realtid", True, MENU_TITLE_COLOR)
            subtitle_surf = subtitle_font.render("Välj ett alternativ för att starta", True, MENU_SUBTITLE_COLOR)
            title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
            subtitle_rect = subtitle_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
            screen.blit(title_surf, title_rect)
            screen.blit(subtitle_surf, subtitle_rect)

            menu_mouse_pos = pygame.mouse.get_pos()
            for btn in menu_buttons:
                btn.draw(screen, subtitle_font, menu_mouse_pos)

            # --- FPS Counter ---
            fps_value = clock.get_fps()
            fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, (200, 200, 200))
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
                    orbit_markers.append((event_type, float(r[0]), float(r[1])))

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
        draw_starfield(screen, camera_center, ppm)

        camera_center_tuple = (float(camera_center[0]), float(camera_center[1]))
        mouse_pos = pygame.mouse.get_pos()

        # Jorden med sprite
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm, camera_center_tuple)


        # --- FPS Counter ---
        fps_value = clock.get_fps()
        fps_text = font_fps.render(f"FPS: {fps_value:.1f}", True, (200, 200, 200))

        # placera i nedre högra hörnet
        text_rect = fps_text.get_rect(bottomright=(WIDTH - 10, HEIGHT - 10))
        screen.blit(fps_text, text_rect)

        # Skala bilden beroende på zoomnivå
        scale_factor = EARTH_RADIUS * ppm * 2 / earth_img_width
        scale_key = max(1, int(round(scale_factor * EARTH_SCALE_CACHE_PRECISION)))
        earth_scaled = earth_scale_cache.get(scale_key)
        if earth_scaled is None:
            quantized_scale = scale_key / EARTH_SCALE_CACHE_PRECISION
            quantized_size = (
                max(1, int(earth_img_width * quantized_scale)),
                max(1, int(earth_img_height * quantized_scale)),
            )
            earth_scaled = pygame.transform.smoothscale(earth_img, quantized_size)
            if len(earth_scale_cache) >= EARTH_SCALE_CACHE_MAX:
                earth_scale_cache.clear()
            earth_scale_cache[scale_key] = earth_scaled

        glow_radius = int(EARTH_RADIUS * ppm * 1.5)
        if glow_radius > 0:
            glow_radius = max(
                EARTH_GLOW_CACHE_STEP,
                (glow_radius + EARTH_GLOW_CACHE_STEP - 1) // EARTH_GLOW_CACHE_STEP * EARTH_GLOW_CACHE_STEP,
            )
            glow_surface = get_earth_glow_surface(glow_radius)
            screen.blit(glow_surface, (earth_screen_pos[0]-glow_radius, earth_screen_pos[1]-glow_radius))


        # Centrera bilden runt origo
        rect = earth_scaled.get_rect(center=earth_screen_pos)
        screen.blit(earth_scaled, rect)


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
                pygame.draw.lines(screen, PREDICTION_COLOR, False, screen_points, 2)

        # Spår
        if show_trail:
            camera_delta = np.linalg.norm(camera_center - trail_last_camera)
            if abs(ppm - trail_last_ppm) > 1e-5 or camera_delta > 5.0:
                trail_surface.fill((0, 0, 0, 0))
                trail_prev_screen_pos = None
                trail_last_ppm = ppm
                if trail_last_camera is not None:
                    np.copyto(trail_last_camera, camera_center)
                else:
                    trail_last_camera = camera_center.copy()

            trail_surface.fill((0, 0, 0, TRAIL_FADE_ALPHA), special_flags=pygame.BLEND_RGBA_SUB)
        else:
            trail_surface.fill((0, 0, 0, 0))
            trail_prev_screen_pos = None

        # Satellit
        sat_pos = world_to_screen(r[0], r[1], ppm, camera_center_tuple)
        if show_trail:
            if trail_prev_screen_pos is not None:
                pygame.draw.line(trail_surface, TRAIL_COLOR_ALPHA, trail_prev_screen_pos, sat_pos, 3)
            trail_prev_screen_pos = sat_pos
            trail_last_ppm = ppm
            if trail_last_camera is not None:
                np.copyto(trail_last_camera, camera_center)
            else:
                trail_last_camera = camera_center.copy()
            screen.blit(trail_surface, (0, 0))
        draw_satellite(screen, sat_pos)

        for marker_type, mx, my in orbit_markers:
            marker_pos = world_to_screen(mx, my, ppm, camera_center_tuple)
            color = PERICENTER_COLOR if marker_type == "pericenter" else APOCENTER_COLOR
            pygame.draw.circle(screen, color, marker_pos, 6)
            pygame.draw.circle(screen, (255, 255, 255), marker_pos, 6, 2)

        # HUD
        rmag = np.linalg.norm(r)
        vmag = np.linalg.norm(v)
        eps = energy_specific(r, v)
        e = eccentricity(r, v)
        vx, vy = v
        hud_lines = [
            f"t = {t_sim:,.0f} s    (speed: {real_time_speed:.2f}x)",
            f"|r| = {rmag/1e6:.2f} Mm    |v| = {vmag:,.1f} m/s",
            f"v_x = {vx: .1f} m/s    v_y = {vy: .1f} m/s",
            f"energy = {eps: .3e} J/kg    e = {e:.4f}",
            "SPACE: paus | R: reset | T: trail | +/-: zoom | C: kamera",
            "Mushjul: zoom | Dra med vänster musknapp för att panorera kameran",
            "←/→: -/+ 1.5x speed   ↑/↓: ×2/÷2 speed   ESC: quit",
            "Knapparna till vänster speglar kontrollerna",
            "Kamera-knappen eller C växlar fokus mellan jord, satellit och fri kamera",
        ]
        y = 10
        #for line in hud_lines:
            #draw_text_with_shadow(screen, font, line, (10, y))
            #y += 24

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
