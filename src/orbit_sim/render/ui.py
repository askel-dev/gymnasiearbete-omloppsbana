from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import pygame

from .assets import Color, get_text_surface


CURRENT_HUD_ALPHA: float = 255.0


def set_hud_alpha(value: float) -> None:
    global CURRENT_HUD_ALPHA
    CURRENT_HUD_ALPHA = value


def get_hud_alpha() -> float:
    return CURRENT_HUD_ALPHA


@dataclass(frozen=True)
class ButtonVisualStyle:
    base_color: Color
    hover_color: Color
    text_color: tuple[int, int, int]
    radius: int
    border_color: Color | None = None
    border_width: int = 0


class Button:
    """Simple rectangular button with hover feedback and callbacks."""

    def __init__(
        self,
        rect: tuple[int, int, int, int],
        text: str,
        callback: Callable[[], None],
        text_getter: Callable[[], str] | None = None,
        *,
        style: ButtonVisualStyle | None = None,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self._text = text
        self._callback = callback
        self._text_getter = text_getter
        self._cached_text_surface: pygame.Surface | None = None
        self._cached_text: str | None = None
        self._cached_font_id: int | None = None
        self._style = style

    def get_text(self) -> str:
        if self._text_getter is not None:
            return self._text_getter()
        return self._text

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        mouse_pos: tuple[int, int] | None = None,
        *,
        style: ButtonVisualStyle | None = None,
    ) -> None:
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        effective_style = style or self._style
        if effective_style is None:
            raise ValueError("Button style must be provided")
        color = effective_style.hover_color if hovered else effective_style.base_color
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surface,
            color,
            button_surface.get_rect(),
            border_radius=effective_style.radius,
        )
        if effective_style.border_color is not None and effective_style.border_width > 0:
            pygame.draw.rect(
                button_surface,
                effective_style.border_color,
                button_surface.get_rect(),
                effective_style.border_width,
                border_radius=effective_style.radius,
            )
        surface.blit(button_surface, self.rect.topleft)
        text_value = self.get_text()
        font_id = id(font)
        if (
            self._cached_text_surface is None
            or text_value != self._cached_text
            or font_id != self._cached_font_id
        ):
            self._cached_text_surface = get_text_surface(font, text_value, effective_style.text_color)
            self._cached_text = text_value
            self._cached_font_id = font_id
        text_surf = self._cached_text_surface
        if CURRENT_HUD_ALPHA < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(int(CURRENT_HUD_ALPHA))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._callback()


def build_text_panel(
    font: pygame.font.Font,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    *,
    background_color: Color,
    padding: tuple[int, int] = (14, 14),
    alpha: int | None = None,
) -> pygame.Surface:
    if not lines:
        raise ValueError("lines must not be empty")
    padding_x, padding_y = padding
    line_height = font.get_linesize()
    width = max(font.size(text)[0] for text, _ in lines) + padding_x * 2
    height = line_height * len(lines) + padding_y * 2
    panel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(
        panel_surface,
        background_color,
        panel_surface.get_rect(),
        border_radius=12,
    )
    for idx, (text, color) in enumerate(lines):
        if not text:
            continue
        text_surf = get_text_surface(font, text, color)
        if alpha is not None and alpha < 255:
            text_surf = text_surf.copy()
            text_surf.set_alpha(alpha)
        panel_surface.blit(text_surf, (padding_x, padding_y + idx * line_height))
    if alpha is not None and alpha < 255:
        panel_surface.set_alpha(alpha)
    return panel_surface
