from __future__ import annotations

import math
import random
from typing import Iterable, Sequence, TYPE_CHECKING

import numpy as np
import pygame

from .assets import AssetLibrary, Color, get_text_surface

if TYPE_CHECKING:  # pragma: no cover
    from orbit_sim.core.config import RenderCfg


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def draw_earth(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
    *,
    assets: AssetLibrary,
) -> None:
    if radius <= 0:
        return
    diameter = radius * 2
    sprite = assets.get_scaled_earth_sprite(diameter)
    rect = sprite.get_rect(center=position)
    surface.blit(sprite, rect)


def draw_satellite(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
    *,
    color: tuple[int, int, int],
) -> None:
    if radius <= 0:
        return
    pygame.draw.circle(surface, color, position, radius)


def draw_heating_glow(
    surface: pygame.Surface,
    position: tuple[int, int],
    radius: int,
    intensity: float,
    *,
    color: tuple[int, int, int],
    outer_alpha: int,
    inner_alpha: int,
    radius_factor: float,
) -> None:
    if intensity <= 0.0 or radius <= 0:
        return
    intensity = _clamp(intensity, 0.0, 1.0)
    glow_radius = max(2, int(radius * (1.8 + radius_factor * intensity)))
    glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    center = glow_radius
    outer = int(outer_alpha * intensity)
    inner = int(inner_alpha * intensity)
    if outer > 0:
        pygame.draw.circle(glow_surface, (*color, outer), (center, center), glow_radius)
    if inner > 0:
        pygame.draw.circle(
            glow_surface,
            (*color, inner),
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
    width: int,
    height: int,
    offset: int,
) -> None:
    if alpha <= 0:
        return

    half_width = width // 2
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
    render_cfg: RenderCfg,
    assets: AssetLibrary,
    image: pygame.Surface | None = None,
) -> None:
    if diameter <= 0:
        return
    if image is not None:
        width, height = image.get_size()
        if width <= 0 or height <= 0:
            return
        scale = diameter / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        scaled = assets.get_scaled_surface(image, new_size)
        rect = scaled.get_rect(center=center)
        surface.blit(scaled, rect)
        return

    radius = diameter // 2
    planet_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(
        planet_surface,
        render_cfg.planet_color,
        (radius, radius),
        radius,
    )
    surface.blit(planet_surface, planet_surface.get_rect(center=center))


def draw_velocity_arrow(
    surface: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    render_cfg: RenderCfg,
) -> None:
    pygame.draw.line(surface, render_cfg.velocity_arrow_color, start, end, 2)
    angle = math.atan2(start[1] - end[1], end[0] - start[0])
    head_angle = math.radians(render_cfg.velocity_arrow_head_angle_deg)
    head_length = render_cfg.velocity_arrow_head_length
    left = (
        int(end[0] - head_length * math.cos(angle - head_angle)),
        int(end[1] + head_length * math.sin(angle - head_angle)),
    )
    right = (
        int(end[0] - head_length * math.cos(angle + head_angle)),
        int(end[1] + head_length * math.sin(angle + head_angle)),
    )
    pygame.draw.polygon(surface, render_cfg.velocity_arrow_color, [end, left, right])


def generate_starfield(
    num_stars: int,
    *,
    size: tuple[int, int],
    rng: random.Random | None = None,
) -> list[dict[str, object]]:
    rng = rng or random.Random()
    width, height = size
    stars: list[dict[str, object]] = []
    for _ in range(num_stars):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        radius = rng.choice([1, 1, 1, 2])
        alpha = rng.randint(80, 150)
        base = rng.randint(200, 240)
        color = (
            max(0, base - rng.randint(10, 25)),
            max(0, base - rng.randint(5, 15)),
            base,
        )
        star_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(star_surface, (*color, alpha), (radius, radius), radius)
        stars.append({"pos": (x, y), "surface": star_surface, "radius": radius})
    return stars


def draw_starfield(
    surface: pygame.Surface,
    starfield: Iterable[dict[str, object]],
    camera_center: np.ndarray,
    ppm: float,
    *,
    render_cfg: RenderCfg,
) -> None:
    width, height = surface.get_size()
    offset_x = camera_center[0] * ppm * render_cfg.starfield_parallax
    offset_y = camera_center[1] * ppm * render_cfg.starfield_parallax
    for star in starfield:
        base_x, base_y = star["pos"]  # type: ignore[index]
        star_surface = star["surface"]  # type: ignore[index]
        radius = star["radius"]  # type: ignore[index]
        sx = int((base_x - offset_x) % width)
        sy = int((base_y + offset_y) % height)
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
    render_cfg: RenderCfg,
    tick_font: pygame.font.Font,
    axis_font: pygame.font.Font,
) -> None:
    if ppm <= 0.0:
        return

    spacing = render_cfg.grid_spacing_meters
    spacing_px = spacing * ppm
    if spacing_px <= 0.0:
        return
    if spacing_px < render_cfg.grid_min_pixel_spacing:
        multiplier = max(1, math.ceil(render_cfg.grid_min_pixel_spacing / spacing_px))
        spacing *= multiplier
        spacing_px = spacing * ppm
        if spacing_px < render_cfg.grid_min_pixel_spacing:
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

    line_color = (*render_cfg.grid_line_color, render_cfg.grid_line_alpha)
    origin_line_color = (
        *render_cfg.grid_line_color,
        min(255, render_cfg.grid_line_alpha + 150),
    )
    origin_line_width = 3

    start_x = math.floor(world_left / spacing) * spacing
    max_vertical = int(math.ceil((world_right - world_left) / spacing)) + 3
    for i in range(max_vertical):
        x_world = start_x + i * spacing
        if x_world > world_right + spacing:
            break
        sx = int(surface.get_width() // 2 + (x_world - cx) * ppm)
        if 0 <= sx <= width:
            draw_x = int(_clamp(sx, 0, width))
            is_origin = abs(x_world) < spacing * 0.5
            color = origin_line_color if is_origin else line_color
            width_px = origin_line_width if is_origin else 1
            pygame.draw.line(surface, color, (draw_x, 0), (draw_x, height), width_px)
            if 0 <= sx <= width and render_cfg.grid_label_margin < sx < width - 140:
                label_text = _format_megameters(x_world)
                if label_text:
                    label_surf = get_text_surface(
                        tick_font, label_text, render_cfg.grid_label_color
                    )
                    if render_cfg.grid_label_alpha < 255:
                        label_surf = label_surf.copy()
                        label_surf.set_alpha(render_cfg.grid_label_alpha)
                    rect = label_surf.get_rect()
                    rect.midtop = (sx, render_cfg.grid_label_margin)
                    if rect.bottom <= height:
                        surface.blit(label_surf, rect)

    start_y = math.floor(world_bottom / spacing) * spacing
    max_horizontal = int(math.ceil((world_top - world_bottom) / spacing)) + 3
    for i in range(max_horizontal):
        y_world = start_y + i * spacing
        if y_world > world_top + spacing:
            break
        sy = int(surface.get_height() // 2 - (y_world - cy) * ppm)
        if 0 <= sy <= height:
            draw_y = int(_clamp(sy, 0, height))
            is_origin = abs(y_world) < spacing * 0.5
            color = origin_line_color if is_origin else line_color
            width_px = origin_line_width if is_origin else 1
            pygame.draw.line(surface, color, (0, draw_y), (width, draw_y), width_px)
            if (
                0 <= sy <= height
                and render_cfg.grid_label_margin * 2 < sy < height - render_cfg.grid_label_margin
            ):
                label_text = _format_megameters(y_world)
                if label_text:
                    label_surf = get_text_surface(
                        tick_font, label_text, render_cfg.grid_label_color
                    )
                    if render_cfg.grid_label_alpha < 255:
                        label_surf = label_surf.copy()
                        label_surf.set_alpha(render_cfg.grid_label_alpha)
                    rect = label_surf.get_rect()
                    rect.midright = (width - render_cfg.grid_label_margin, sy)
                    if rect.left >= 0:
                        surface.blit(label_surf, rect)

    axis_color = render_cfg.grid_label_color
    axis_label = get_text_surface(axis_font, "Y [Mm]", axis_color)
    if render_cfg.grid_axis_label_alpha < 255:
        axis_label = axis_label.copy()
        axis_label.set_alpha(render_cfg.grid_axis_label_alpha)
    axis_rect = axis_label.get_rect()
    axis_rect.bottomright = (
        width - render_cfg.grid_label_margin,
        height - render_cfg.grid_label_margin,
    )
    surface.blit(axis_label, axis_rect)

    y_axis_label = get_text_surface(axis_font, "X [Mm]", axis_color)
    if render_cfg.grid_axis_label_alpha < 255:
        y_axis_label = y_axis_label.copy()
        y_axis_label.set_alpha(render_cfg.grid_axis_label_alpha)
    y_axis_rect = y_axis_label.get_rect()
    y_axis_rect.topright = (
        width - render_cfg.grid_label_margin,
        render_cfg.grid_label_margin,
    )
    surface.blit(y_axis_label, y_axis_rect)


def draw_orbit_line(
    surface: pygame.Surface,
    color: tuple[int, int, int] | tuple[int, int, int, int],
    points: Sequence[tuple[int, int]],
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
    points: Sequence[tuple[float, float]], max_points: int
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    step = max(1, math.ceil(len(points) / max_points))
    sampled = list(points[::step])
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled
