# src/orbit_pygame.py
import json
import math
import os
import random
import time
from pathlib import Path

import pygame
from pygame.locals import *  # noqa: F401,F403 - required for constants such as FULLSCREEN
import sys
import numpy as np
from collections import deque
from dataclasses import dataclass

from orbit_sim.core.config import PHYSICS_CFG, RENDER_CFG
from orbit_sim.core.logging_utils import RunLogger
from orbit_sim.core.model import Body, Satellite, SimState
from orbit_sim.core.physics import (
    atmosphere_depth_ratio,
    clamp,
    compute_orbit_prediction,
    eccentricity,
    energy_specific,
    in_atmosphere,
    rk4_step,
)
from orbit_sim.core.timekeeping import FixedStepAccumulator, FrameTimer
from orbit_sim.data.scenarios import (
    DEFAULT_SCENARIO_KEY,
    SCENARIO_DISPLAY_ORDER,
    SCENARIO_FLASH_DURATION,
    SCENARIOS,
    Scenario,
)

from orbit_sim.render.assets import AssetLibrary, get_text_surface, load_font
from orbit_sim.render.camera import Camera
from orbit_sim.render.draw import (
    draw_coordinate_grid,
    draw_earth,
    draw_heating_glow,
    draw_marker_pin,
    draw_menu_planet,
    draw_orbit_line,
    draw_satellite,
    draw_starfield,
    draw_velocity_arrow,
    downsample_points,
    generate_starfield,
)
from orbit_sim.render.ui import (
    Button,
    ButtonVisualStyle,
    build_text_panel,
    get_hud_alpha,
    set_hud_alpha,
)


SETTINGS_DIR = Path.home() / ".orbit_sim"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


def load_user_settings() -> dict[str, object]:
    """Return persisted runtime settings if the JSON file is readable."""

    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


def save_user_settings(settings: dict[str, object]) -> None:
    """Persist runtime settings, ignoring filesystem errors."""

    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with SETTINGS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, sort_keys=True)
    except OSError:
        # Silently ignore failures – the simulation should continue shutting down.
        pass




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
    set_hud_alpha(float(HUD_TEXT_ALPHA_BASE))
    font_fps = pygame.font.SysFont("consolas", 14)
    assets = AssetLibrary()

    user_settings = load_user_settings()

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

    fullscreen_setting = user_settings.get("fullscreen")
    fullscreen_preference = fullscreen_setting if isinstance(fullscreen_setting, bool) else True

    fullscreen_enabled = fullscreen_preference
    try:
        if fullscreen_preference:
            screen = create_fullscreen_surface()
        else:
            screen = create_windowed_surface()
            fullscreen_enabled = False
    except pygame.error:
        if fullscreen_preference:
            try:
                screen = create_windowed_surface()
                fullscreen_enabled = False
            except pygame.error:
                screen = _set_display_mode_with_vsync(WINDOWED_DEFAULT_SIZE, windowed_flags)
                fullscreen_enabled = False
        else:
            try:
                screen = create_fullscreen_surface()
                fullscreen_enabled = True
            except pygame.error:
                screen = _set_display_mode_with_vsync(WINDOWED_DEFAULT_SIZE, windowed_flags)
                fullscreen_enabled = False

    current_size = screen.get_size()
    update_display_metrics(*current_size)
    camera = Camera(current_size, PIXELS_PER_METER, min_ppm=MIN_PPM, max_ppm=MAX_PPM)

    overlay_size = current_size
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
        hud_alpha = get_hud_alpha()
        if hud_alpha < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(int(hud_alpha))

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

    menu_planet_image = assets.load_menu_planet_image("main_menu_art.png")

    saved_scenario_key = user_settings.get("scenario_key")
    if not (isinstance(saved_scenario_key, str) and saved_scenario_key in SCENARIOS):
        saved_scenario_key = DEFAULT_SCENARIO_KEY
    current_scenario_key = saved_scenario_key
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
        STARFIELD = generate_starfield(220, size=(WIDTH, HEIGHT))

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
    earth_body = Body(name="Earth", radius=EARTH_RADIUS, mu=MU)
    sim_state = SimState(
        body=earth_body,
        satellite=Satellite(
            position=R0.copy(),
            velocity=scenario_velocity_vector(),
        ),
        time=0.0,
        paused=False,
    )
    real_time_speed = REAL_TIME_SPEED
    grid_overlay_enabled = False
    show_velocity_arrow = True
    camera_mode = "earth"
    is_dragging_camera = False

    saved_zoom = user_settings.get("zoom_ppm")
    if isinstance(saved_zoom, (int, float)):
        camera.set_zoom(clamp(float(saved_zoom), MIN_PPM, MAX_PPM))
    else:
        camera.set_zoom(clamp(camera.ppm, MIN_PPM, MAX_PPM))

    saved_speed = user_settings.get("real_time_speed")
    if isinstance(saved_speed, (int, float)):
        real_time_speed = clamp(float(saved_speed), 0.1, 10_000.0)

    saved_grid = user_settings.get("grid_overlay_enabled")
    if isinstance(saved_grid, bool):
        grid_overlay_enabled = saved_grid

    saved_velocity_toggle = user_settings.get("show_velocity_arrow")
    if isinstance(saved_velocity_toggle, bool):
        show_velocity_arrow = saved_velocity_toggle

    saved_camera_mode = user_settings.get("camera_mode")
    if isinstance(saved_camera_mode, str) and saved_camera_mode in {"earth", "satellite", "manual"}:
        camera_mode = saved_camera_mode

    saved_camera_center = user_settings.get("camera_center")
    if (
        isinstance(saved_camera_center, (list, tuple))
        and len(saved_camera_center) == 2
    ):
        try:
            camera.set_center(
                (
                    float(saved_camera_center[0]),
                    float(saved_camera_center[1]),
                )
            )
        except (TypeError, ValueError):
            pass

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
    stepper = FixedStepAccumulator(step=DT_PHYS, max_substeps=MAX_SUBSTEPS)
    frame_timer = FrameTimer()

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
        nonlocal sim_state, real_time_speed
        nonlocal stepper, frame_timer, log_step_counter, prev_r, prev_dr
        nonlocal impact_logged, escape_logged
        nonlocal orbit_markers
        nonlocal camera_mode, is_dragging_camera
        nonlocal orbit_prediction_period, orbit_prediction_points
        nonlocal pinned_markers, pin_feedback_text, pin_feedback_time
        nonlocal impact_info, shock_ring_start, impact_freeze_time
        nonlocal impact_overlay_reveal_time, impact_overlay_visible_since
        nonlocal atmosphere_entry_time_sim
        nonlocal atmosphere_entry_time_real, atmosphere_warning_end_time
        nonlocal atmosphere_logged, show_velocity_arrow
        nonlocal camera
        close_logger()
        sim_state.reset(R0, scenario_velocity_vector())
        camera.set_zoom_target(camera.ppm_target)
        camera.set_zoom(camera.ppm)
        real_time_speed = clamp(real_time_speed, 0.1, 10_000.0)
        stepper.clear()
        frame_timer = FrameTimer()
        log_step_counter = 0
        prev_r = None
        prev_dr = None
        impact_logged = False
        escape_logged = False
        orbit_markers.clear()
        pinned_markers.clear()
        pin_feedback_text = None
        pin_feedback_time = 0.0
        camera.set_target((float(camera.center[0]), float(camera.center[1])))
        is_dragging_camera = False
        orbit_prediction_period, orbit_prediction_points = compute_orbit_prediction(
            sim_state.satellite.position,
            sim_state.satellite.velocity,
        )
        impact_info = None
        shock_ring_start = None
        impact_freeze_time = None
        impact_overlay_reveal_time = None
        impact_overlay_visible_since = None
        atmosphere_entry_time_sim = None
        atmosphere_entry_time_real = None
        atmosphere_warning_end_time = 0.0
        atmosphere_logged = False
        set_hud_alpha(float(HUD_TEXT_ALPHA_BASE))

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
        closest_type: str | None = None
        closest_distance = float("inf")
        for marker_type, mx, my, _ in reversed(orbit_markers):
            marker_pos = camera.world_to_screen(mx, my)
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
        r_vec = sim_state.satellite.position
        v_vec = sim_state.satellite.velocity
        rmag = float(np.linalg.norm(r_vec))
        vmag = float(np.linalg.norm(v_vec))
        eps = float(energy_specific(r_vec, v_vec))
        h_vec = np.cross(np.array([r_vec[0], r_vec[1], 0.0]), np.array([v_vec[0], v_vec[1], 0.0]))
        h_mag = float(np.linalg.norm(h_vec))
        e_val = float(eccentricity(r_vec, v_vec))
        logger.log_ts(
            [
                float(sim_state.time),
                float(r_vec[0]),
                float(r_vec[1]),
                float(v_vec[0]),
                float(v_vec[1]),
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
        prev_r = float(np.linalg.norm(sim_state.satellite.position))
        prev_dr = None
        impact_logged = False
        escape_logged = False
        log_state(0.0)

    def start_simulation():
        nonlocal state
        reset()
        state = "running"
        init_run_logging()

    def collect_user_settings() -> dict[str, object]:
        return {
            "scenario_key": current_scenario_key,
            "camera_center": [float(camera.center[0]), float(camera.center[1])],
            "zoom_ppm": float(camera.ppm_target),
            "camera_mode": camera_mode,
            "grid_overlay_enabled": bool(grid_overlay_enabled),
            "show_velocity_arrow": bool(show_velocity_arrow),
            "real_time_speed": float(real_time_speed),
            "fullscreen": bool(fullscreen_enabled),
        }

    def quit_app():
        save_user_settings(collect_user_settings())
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
        sim_state.paused = not sim_state.paused

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
        nonlocal current_scenario_key, state
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
            sim_state.paused = False
            init_run_logging()
        elif previous_state != "menu":
            state = "running"

    def reset_and_continue():
        nonlocal state, impact_info, shock_ring_start
        reset()
        state = "running"
        sim_state.paused = False
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
        Button(
            (20, 20, button_width, button_height),
            "Pause",
            toggle_pause,
            lambda: "Resume" if sim_state.paused else "Pause",
            style=DEFAULT_BUTTON_STYLE,
        ),
        Button(
            (20, 20 + (button_height + button_gap), button_width, button_height),
            "Reset",
            reset_and_continue,
            style=DEFAULT_BUTTON_STYLE,
        ),
        Button(
            (20, 20 + 2 * (button_height + button_gap), button_width, button_height),
            "Slower",
            slow_down,
            style=DEFAULT_BUTTON_STYLE,
        ),
        Button(
            (20, 20 + 3 * (button_height + button_gap), button_width, button_height),
            "Faster",
            speed_up,
            style=DEFAULT_BUTTON_STYLE,
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
            style=DEFAULT_BUTTON_STYLE,
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
            camera.update_size(current_size)
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
                if event.key == pygame.K_F11:
                    toggle_fullscreen_mode()
                    continue
                if event.key in scenario_shortcut_map:
                    set_scenario(scenario_shortcut_map[event.key])
                    if state == "impact":
                        state = "running"
                        sim_state.paused = False
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
                        sim_state.paused = not sim_state.paused
                    elif event.key == pygame.K_r:
                        reset_and_continue()
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        camera.zoom_by_factor(1.2)     # zoom in
                    elif event.key == pygame.K_MINUS:
                        camera.zoom_by_factor(1 / 1.2)     # zoom out

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
                            camera.begin_pan(event.pos)
                            camera_mode = "manual"
                            camera.set_target(
                                (
                                    float(camera.center[0]),
                                    float(camera.center[1]),
                                )
                            )
                elif event.button == 4 and state == "running":
                    camera.zoom_by_factor(1.1)
                elif event.button == 5 and state == "running":
                    camera.zoom_by_factor(1 / 1.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_dragging_camera = False
                    camera.end_pan()
            elif event.type == pygame.MOUSEMOTION:
                if is_dragging_camera and state == "running":
                    camera.pan(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                if state == "running" and event.y != 0:
                    zoom_factor = 1.1 ** event.y
                    camera.zoom_by_factor(zoom_factor)
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

            draw_menu_planet(
                screen,
                planet_center,
                menu_planet_diameter,
                render_cfg=RENDER_CFG,
                assets=assets,
                image=menu_planet_image,
            )

            for btn in menu_buttons:
                btn.draw(screen, menu_button_font, menu_mouse_pos)

            panel_lines = [scenario_panel_title, *scenario_help_lines]
            if panel_lines:
                line_specs = []
                for idx, line in enumerate(panel_lines):
                    if idx == 0:
                        color = HUD_TEXT_COLOR
                    else:
                        scenario_key = SCENARIO_DISPLAY_ORDER[idx - 1]
                        color = (
                            HUD_TEXT_COLOR
                            if scenario_key == current_scenario_key
                            else MENU_SUBTITLE_COLOR
                        )
                    line_specs.append((line, color))
                panel_surface = build_text_panel(
                    scenario_font,
                    line_specs,
                    background_color=LABEL_BACKGROUND_COLOR,
                    padding=(14, 14),
                )
                panel_rect = panel_surface.get_rect()
                panel_rect.topleft = (
                    20,
                    HEIGHT - panel_surface.get_height() - 20,
                )
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
        frame_dt_real = frame_timer.tick()
        sim_dt_target = frame_dt_real * real_time_speed

        if sim_state.paused:
            stepper.clear()
            steps_to_run, dt_step = 0, 0.0
        else:
            stepper.accrue(sim_dt_target)
            steps_to_run, dt_step = stepper.consume()

        if steps_to_run > 0 and dt_step > 0.0:

            for _ in range(steps_to_run):
                position, velocity = rk4_step(
                    sim_state.satellite.position,
                    sim_state.satellite.velocity,
                    dt_step,
                )
                sim_state.satellite.position = position
                sim_state.satellite.velocity = velocity
                sim_state.time += dt_step
                rmag = float(np.linalg.norm(position))
                current_in_atmosphere = in_atmosphere(position)
                if current_in_atmosphere and atmosphere_entry_time_sim is None:
                    atmosphere_entry_time_sim = float(sim_state.time)
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
                    orbit_markers.append(
                        (event_type, float(position[0]), float(position[1]), rmag)
                    )
                    refresh_pinned_marker(event_type)

                vmag = float(np.linalg.norm(velocity))
                eps = float(energy_specific(position, velocity))
                e_val = float(eccentricity(position, velocity))
                impact_triggered = not impact_logged and rmag <= EARTH_RADIUS

                if logger is not None:
                    log_step_counter += 1

                    if event_type is not None:
                        logger.log_event(
                            [
                                float(sim_state.time),
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
                                float(sim_state.time),
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
                                float(sim_state.time),
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
                                float(sim_state.time),
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
                        normal = position / max(rmag, 1e-9)
                        tangent = np.array([-normal[1], normal[0]])
                        tangential_speed = float(np.dot(velocity, tangent))
                        radial_speed = float(np.dot(velocity, normal))
                        angle_rad = math.atan2(abs(radial_speed), abs(tangential_speed))
                        impact_info = {
                            "time": float(sim_state.time),
                            "speed": vmag,
                            "angle": math.degrees(angle_rad),
                            "radial_speed": radial_speed,
                            "tangential_speed": tangential_speed,
                            "position": (float(position[0]), float(position[1])),
                        }
                        shock_ring_start = time.perf_counter()
                        impact_freeze_time = shock_ring_start + IMPACT_FREEZE_DELAY
                        overlay_ready_time = shock_ring_start + IMPACT_OVERLAY_DELAY
                        if atmosphere_warning_end_time > 0.0:
                            overlay_ready_time = max(overlay_ready_time, atmosphere_warning_end_time)
                        impact_overlay_reveal_time = overlay_ready_time
                        impact_overlay_visible_since = None
                        sim_state.paused = True
                        state = "impact"
                        stepper.clear()
                        break

            stepper.clear()

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
        current_alpha = get_hud_alpha()
        new_alpha = current_alpha + (target_hud_alpha - current_alpha) * 0.2
        new_alpha = clamp(new_alpha, 0.0, HUD_TEXT_ALPHA_BASE)
        set_hud_alpha(new_alpha)
        hud_alpha = get_hud_alpha()

        r_vec = sim_state.satellite.position
        v_vec = sim_state.satellite.velocity
        rmag = float(np.linalg.norm(r_vec))

        # Smooth zoom mot mål
        if camera_mode == "earth":
            view_offset = (HEIGHT * 0.08) / max(camera.ppm, 1e-6)
            camera.set_target((0.0, view_offset))
        elif camera_mode == "satellite":
            camera.set_target((float(r_vec[0]), float(r_vec[1])))
        camera.update()
        ppm = camera.ppm

        # Bakgrund
        screen.fill(background_color)
        draw_starfield(
            screen,
            STARFIELD,
            camera.center,
            ppm,
            render_cfg=RENDER_CFG,
        )

        camera_center_tuple = (float(camera.center[0]), float(camera.center[1]))
        if grid_overlay_enabled:
            draw_coordinate_grid(
                grid_surface,
                ppm,
                camera_center_tuple,
                render_cfg=RENDER_CFG,
                tick_font=grid_label_font,
                axis_font=grid_axis_font,
            )
            screen.blit(grid_surface, (0, 0))
        mouse_pos = pygame.mouse.get_pos()
        depth_ratio = atmosphere_depth_ratio(rmag)
        earth_screen_pos = camera.world_to_screen(0.0, 0.0)

        if orbit_prediction_points:
            if orbit_prediction_period is None or orbit_prediction_period <= 0.0:
                reveal_fraction = 1.0
            else:
                reveal_fraction = clamp(sim_state.time / orbit_prediction_period, 0.0, 1.0)
            total_points = len(orbit_prediction_points)
            if reveal_fraction >= 1.0:
                subset = orbit_prediction_points
            else:
                max_index = max(2, int(total_points * reveal_fraction))
                subset = orbit_prediction_points[:max_index]
            sampled_points = downsample_points(subset, MAX_RENDERED_ORBIT_POINTS)
            screen_points = [
                camera.world_to_screen(px, py)
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
            assets=assets,
        )

        if impact_info is not None and shock_ring_start is not None:
            elapsed_ring = now_time - shock_ring_start
            if elapsed_ring <= SHOCK_RING_DURATION:
                progress = clamp(elapsed_ring / SHOCK_RING_DURATION, 0.0, 1.0)
                impact_position = impact_info.get("position") if impact_info else None
                if impact_position is not None:
                    ring_center = camera.world_to_screen(
                        impact_position[0],
                        impact_position[1],
                    )
                else:
                    ring_center = camera.world_to_screen(
                        r_vec[0],
                        r_vec[1],
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

        sat_pos = camera.world_to_screen(r_vec[0], r_vec[1])
        sat_radius_px = compute_satellite_radius(rmag)
        heating_intensity = depth_ratio if rmag >= EARTH_RADIUS else 1.0
        if heating_intensity > 0.0:
            draw_heating_glow(
                screen,
                sat_pos,
                sat_radius_px,
                heating_intensity,
                color=ATM_GLOW_COLOR,
                outer_alpha=ATM_GLOW_OUTER_ALPHA,
                inner_alpha=ATM_GLOW_INNER_ALPHA,
                radius_factor=ATM_GLOW_RADIUS_FACTOR,
            )
        draw_satellite(screen, sat_pos, sat_radius_px, color=SATELLITE_COLOR)

        if show_velocity_arrow:
            vx, vy = float(v_vec[0]), float(v_vec[1])
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
                    draw_velocity_arrow(
                        screen,
                        sat_pos,
                        arrow_end,
                        render_cfg=RENDER_CFG,
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
                marker_pos = camera.world_to_screen(mx, my)
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
                    width=LABEL_MARKER_PIN_WIDTH,
                    height=LABEL_MARKER_PIN_HEIGHT,
                    offset=LABEL_MARKER_PIN_OFFSET,
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
        vmag = float(np.linalg.norm(v_vec))
        eps = energy_specific(r_vec, v_vec)
        e = eccentricity(r_vec, v_vec)
        altitude_km = (rmag - EARTH_RADIUS) / 1_000.0
        scenario = get_current_scenario()
        hud_entries: list[tuple[str, tuple[int, int, int]]] = [
            (f"Scenario: {scenario.name}", HUD_TEXT_COLOR),
            (f"t {sim_state.time:,.0f} s   ×{real_time_speed:.1f}", HUD_TEXT_COLOR),
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
        alpha_value = int(min(255, hud_alpha))
        hud_surface = build_text_panel(
            font,
            hud_entries,
            background_color=LABEL_BACKGROUND_COLOR,
            padding=(padding_x, padding_y),
            alpha=alpha_value,
        )
        screen.blit(hud_surface, (20, 20))

        for btn in sim_buttons:
            btn.draw(screen, font, mouse_pos)

            panel_lines = [scenario_panel_title, *scenario_help_lines]
            if panel_lines:
                line_specs: list[tuple[str, tuple[int, int, int]]] = []
                for idx, line in enumerate(panel_lines):
                    if idx == 0:
                        color = HUD_TEXT_COLOR
                    else:
                        scenario_key = SCENARIO_DISPLAY_ORDER[idx - 1]
                        color = (
                            HUD_TEXT_COLOR
                            if scenario_key == current_scenario_key
                            else MENU_SUBTITLE_COLOR
                        )
                    line_specs.append((line, color))
                alpha_value = int(min(255, hud_alpha))
                panel_surface = build_text_panel(
                    scenario_font,
                    line_specs,
                    background_color=LABEL_BACKGROUND_COLOR,
                    padding=(14, 14),
                    alpha=alpha_value,
                )
                panel_rect = panel_surface.get_rect()
                panel_rect.topleft = (
                    20,
                    HEIGHT - panel_surface.get_height() - 20,
                )
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
            if hud_alpha < 255:
                flash_surf = flash_surf.copy()
                flash_surf.set_alpha(int(hud_alpha))
            flash_rect = flash_surf.get_rect(center=(WIDTH // 2, 40))
            screen.blit(flash_surf, flash_rect)

        fps_value = clock.get_fps()
        fps_text = get_text_surface(font_fps, f"FPS: {fps_value:.1f}", HUD_TEXT_COLOR)
        combined_alpha = FPS_TEXT_ALPHA
        if hud_alpha < 255:
            combined_alpha = min(combined_alpha, int(hud_alpha))
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
