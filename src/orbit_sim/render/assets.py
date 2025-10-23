from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import pygame


Color = tuple[int, int, int] | tuple[int, int, int, int]


class AssetLibrary:
    """Cache for frequently accessed render assets."""

    def __init__(self, asset_dir: Path | None = None) -> None:
        self._asset_dir = asset_dir or Path(__file__).resolve().parents[3] / "assets"
        self._earth_sprite_base: pygame.Surface | None = None
        self._earth_sprite_cache: dict[int, pygame.Surface] = {}
        self._menu_planet_cache: dict[tuple[int, int, int], pygame.Surface] = {}

    @property
    def asset_dir(self) -> Path:
        return self._asset_dir

    def load_earth_sprite_base(self, filename: str = "earth_sprite_2.png") -> pygame.Surface:
        if self._earth_sprite_base is None:
            path = self._asset_dir / filename
            self._earth_sprite_base = pygame.image.load(path.as_posix()).convert_alpha()
        return self._earth_sprite_base

    def get_scaled_earth_sprite(self, diameter: int, *, filename: str = "earth_sprite_2.png") -> pygame.Surface:
        if diameter <= 0:
            raise ValueError("Earth sprite diameter must be positive")
        cached = self._earth_sprite_cache.get(diameter)
        if cached is not None:
            return cached
        base = self.load_earth_sprite_base(filename)
        scaled = pygame.transform.smoothscale(base, (diameter, diameter))
        self._earth_sprite_cache[diameter] = scaled
        return scaled

    def load_menu_planet_image(self, filename: str = "menu_planet.png") -> pygame.Surface | None:
        path = self._asset_dir / filename
        try:
            return pygame.image.load(path.as_posix()).convert_alpha()
        except pygame.error:
            return None

    def get_scaled_surface(self, surface: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
        cache_key = (id(surface), size[0], size[1])
        cached = self._menu_planet_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(surface, size).convert_alpha()
        self._menu_planet_cache[cache_key] = scaled
        return scaled


_TEXT_SURFACE_CACHE_MAX_SIZE = 256
_TEXT_SURFACE_CACHE: OrderedDict[tuple[int, str, Color], pygame.Surface] = OrderedDict()


def get_text_surface(font: pygame.font.Font, text: str, color: Color) -> pygame.Surface:
    """Return a cached rendered surface for the given font, text and color."""

    key = (id(font), text, color)
    cached = _TEXT_SURFACE_CACHE.get(key)
    if cached is not None:
        _TEXT_SURFACE_CACHE.move_to_end(key)
        return cached
    rendered = font.render(text, True, color)
    _TEXT_SURFACE_CACHE[key] = rendered
    if len(_TEXT_SURFACE_CACHE) > _TEXT_SURFACE_CACHE_MAX_SIZE:
        _TEXT_SURFACE_CACHE.popitem(last=False)
    return rendered


def load_font(preferred_names: Iterable[str], size: int, *, bold: bool = False) -> pygame.font.Font:
    for name in preferred_names:
        try:
            match = pygame.font.match_font(name, bold=bold)
        except Exception:
            match = None
        if match:
            return pygame.font.Font(match, size)
    fallback = next(iter(preferred_names), None)
    return pygame.font.SysFont(fallback, size, bold=bold)
