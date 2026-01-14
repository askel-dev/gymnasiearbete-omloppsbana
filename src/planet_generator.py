# src/planet_generator.py
"""Procedural planet generation with configurable visual and physical properties."""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pygame

from physics import G


# =======================
#   SPRITE CACHE
# =======================
# Cache sprites by (planet_seed, diameter) to avoid regenerating
# the same sprite multiple times during zoom operations
_PLANET_SPRITE_CACHE: dict[tuple[int, int], pygame.Surface] = {}
_CACHE_MAX_SIZE = 50  # Maximum number of cached sprites


def _get_cache_key(planet: "Planet", diameter: int) -> tuple[int, int]:
    """Generate a cache key for a planet sprite."""
    return (planet.seed, diameter)


def clear_sprite_cache() -> None:
    """Clear all cached planet sprites."""
    _PLANET_SPRITE_CACHE.clear()


def get_cache_size() -> int:
    """Return the current number of cached sprites."""
    return len(_PLANET_SPRITE_CACHE)


# =======================
#   PLANET TYPES
# =======================
class PlanetType(Enum):
    """Categories of planets with distinct visual and physical characteristics."""
    ROCKY_TERRESTRIAL = auto()  # Earth-like with oceans, continents, clouds
    ROCKY_BARREN = auto()       # Mars/Moon-like with craters, no atmosphere
    GAS_GIANT = auto()          # Jupiter/Saturn-like with bands and storms
    ICE_GIANT = auto()          # Neptune/Uranus-like with ice/methane colors
    MOLTEN = auto()             # Volcanic worlds with lava flows


# =======================
#   COLOR PALETTES
# =======================
# Default color palettes for each planet type
DEFAULT_PALETTES: dict[PlanetType, tuple[tuple[int, int, int], ...]] = {
    PlanetType.ROCKY_TERRESTRIAL: (
        (0, 105, 148),    # Ocean blue
        (34, 139, 34),    # Forest green (land)
        (160, 82, 45),    # Sienna/brown (land)
        (255, 255, 255),  # White (clouds)
        (100, 200, 255),  # Atmosphere glow
    ),
    PlanetType.ROCKY_BARREN: (
        (139, 90, 43),    # Mars rust
        (205, 133, 63),   # Sandy brown
        (101, 67, 33),    # Dark brown
        (169, 169, 169),  # Gray (craters)
        (120, 80, 50),    # Darker rust
    ),
    PlanetType.GAS_GIANT: (
        (201, 144, 57),   # Jupiter tan
        (166, 124, 82),   # Brown band
        (234, 214, 183),  # Light cream band
        (139, 90, 43),    # Dark band
        (255, 87, 51),    # Storm red (Great Red Spot)
    ),
    PlanetType.ICE_GIANT: (
        (100, 149, 237),  # Cornflower blue
        (72, 209, 204),   # Medium turquoise
        (176, 224, 230),  # Powder blue
        (70, 130, 180),   # Steel blue
        (135, 206, 250),  # Light sky blue
    ),
    PlanetType.MOLTEN: (
        (30, 30, 30),     # Dark rock
        (50, 40, 35),     # Charred surface
        (255, 69, 0),     # Lava orange-red
        (255, 140, 0),    # Lava orange
        (255, 215, 0),    # Lava yellow (hottest)
    ),
}


# =======================
#   PLANET DATACLASS
# =======================
@dataclass
class Planet:
    """
    Represents a planet with physical and visual properties.
    
    Attributes:
        name: Display name of the planet
        planet_type: Category determining visual generation style
        radius: Radius in meters
        mass: Mass in kilograms
        base_colors: Color palette for procedural generation (optional, uses defaults)
        seed: Random seed for reproducible generation
    """
    name: str
    planet_type: PlanetType
    radius: float
    mass: float
    base_colors: Optional[tuple[tuple[int, int, int], ...]] = None
    seed: int = field(default_factory=lambda: random.randint(0, 999999))
    
    def __post_init__(self):
        """Set default colors if not provided."""
        if self.base_colors is None:
            self.base_colors = DEFAULT_PALETTES.get(self.planet_type)
    
    @property
    def mu(self) -> float:
        """Gravitational parameter (G * mass) in m³/s²."""
        return G * self.mass
    
    @property
    def surface_gravity(self) -> float:
        """Surface gravity (g = G*M/R²) in m/s²."""
        return G * self.mass / (self.radius ** 2)
    
    @property
    def escape_velocity(self) -> float:
        """Escape velocity (sqrt(2*G*M/R)) in m/s."""
        return math.sqrt(2 * G * self.mass / self.radius)
    
    @property
    def circular_orbit_velocity(self) -> float:
        """Velocity for circular orbit at surface level in m/s."""
        return math.sqrt(G * self.mass / self.radius)


# =======================
#   PLANET FACTORY
# =======================
def create_planet(
    name: str,
    planet_type: PlanetType,
    radius: float,
    mass: float,
    seed: Optional[int] = None,
    base_colors: Optional[tuple[tuple[int, int, int], ...]] = None,
) -> Planet:
    """
    Create a planet with specified properties.
    
    Args:
        name: Display name of the planet
        planet_type: Category determining visual style
        radius: Radius in meters
        mass: Mass in kilograms
        seed: Random seed for reproducible generation (auto-generated if None)
        base_colors: Custom color palette (uses defaults if None)
    
    Returns:
        A new Planet instance
    """
    if seed is None:
        seed = random.randint(0, 999999)
    return Planet(
        name=name,
        planet_type=planet_type,
        radius=radius,
        mass=mass,
        base_colors=base_colors,
        seed=seed,
    )


def generate_random_planet(seed: Optional[int] = None) -> Planet:
    """
    Generate a completely random planet.
    
    Args:
        seed: Random seed for reproducibility (auto-generated if None)
    
    Returns:
        A randomly generated Planet
    """
    if seed is None:
        seed = random.randint(0, 999999)
    
    rng = random.Random(seed)
    
    # Pick random type
    planet_type = rng.choice(list(PlanetType))
    
    # Generate physical properties based on type
    if planet_type == PlanetType.ROCKY_TERRESTRIAL:
        # Earth-like: 0.5 to 2 Earth radii, density similar to Earth
        radius_factor = 0.5 + rng.random() * 1.5
        radius = 6_371_000 * radius_factor
        # Mass scales roughly with radius^3 for similar density
        mass = 5.972e24 * (radius_factor ** 3) * (0.8 + rng.random() * 0.4)
    elif planet_type == PlanetType.ROCKY_BARREN:
        # Smaller rocky bodies: 0.1 to 1 Earth radii
        radius_factor = 0.1 + rng.random() * 0.9
        radius = 6_371_000 * radius_factor
        mass = 5.972e24 * (radius_factor ** 3) * (0.6 + rng.random() * 0.4)
    elif planet_type == PlanetType.GAS_GIANT:
        # Giant: 5 to 15 Earth radii, but less dense
        radius_factor = 5 + rng.random() * 10
        radius = 6_371_000 * radius_factor
        # Gas giants are less dense
        mass = 5.972e24 * (radius_factor ** 3) * (0.1 + rng.random() * 0.2)
    elif planet_type == PlanetType.ICE_GIANT:
        # Medium: 3 to 6 Earth radii
        radius_factor = 3 + rng.random() * 3
        radius = 6_371_000 * radius_factor
        mass = 5.972e24 * (radius_factor ** 3) * (0.15 + rng.random() * 0.15)
    else:  # MOLTEN
        # Variable size, slightly denser
        radius_factor = 0.3 + rng.random() * 1.2
        radius = 6_371_000 * radius_factor
        mass = 5.972e24 * (radius_factor ** 3) * (1.0 + rng.random() * 0.3)
    
    # Generate a random name
    prefixes = ["Nova", "Zeta", "Alpha", "Theta", "Omega", "Kappa", "Delta", "Sigma"]
    suffixes = ["Prime", "Minor", "Major", "VII", "IX", "III", "X", ""]
    name = f"{rng.choice(prefixes)}-{rng.randint(100, 999)}{rng.choice(suffixes)}"
    
    return Planet(
        name=name,
        planet_type=planet_type,
        radius=radius,
        mass=mass,
        seed=seed,
    )


# =======================
#   VISUAL GENERATORS
# =======================
def _generate_rocky_terrestrial(
    diameter: int,
    colors: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> pygame.Surface:
    """Generate an Earth-like planet with oceans, continents, and clouds."""
    radius = diameter / 2.0
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Ocean base
    ocean_color = colors[0] if len(colors) > 0 else (0, 105, 148)
    texture.fill(ocean_color)
    
    # Draw continents
    land_colors = colors[1:3] if len(colors) >= 3 else [(34, 139, 34), (160, 82, 45)]
    num_continents = rng.randint(5, 9)
    for _ in range(num_continents):
        cx = int(rng.random() * diameter)
        cy = int(rng.random() * diameter)
        r_factor = 0.3 + rng.random() * 0.5
        cr = int(radius * r_factor)
        color = rng.choice(land_colors) if land_colors else (34, 139, 34)
        pygame.draw.circle(texture, color, (cx, cy), cr)
    
    # Draw clouds
    cloud_color = colors[3] if len(colors) > 3 else (255, 255, 255)
    num_clouds = rng.randint(10, 18)
    for _ in range(num_clouds):
        cx = int(rng.random() * diameter)
        cy = int(rng.random() * diameter)
        cw = int(radius * (0.3 + rng.random() * 0.5))
        ch = int(radius * (0.15 + rng.random() * 0.25))
        if cw > 0 and ch > 0:
            cloud_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
            pygame.draw.ellipse(cloud_surf, (*cloud_color, 150), cloud_surf.get_rect())
            texture.blit(cloud_surf, (cx - cw // 2, cy - ch // 2))
    
    # Apply circular mask
    texture = _apply_circular_mask(texture, diameter)
    
    # Atmosphere glow
    atmo_color = colors[4] if len(colors) > 4 else (100, 200, 255)
    pygame.draw.circle(texture, atmo_color, (int(radius), int(radius)), int(radius), 2)
    
    return texture


def _generate_rocky_barren(
    diameter: int,
    colors: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> pygame.Surface:
    """Generate a Mars/Moon-like barren planet with craters."""
    radius = diameter / 2.0
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Base surface color
    base_color = colors[0] if len(colors) > 0 else (139, 90, 43)
    texture.fill(base_color)
    
    # Add surface variation with patches
    patch_colors = colors[1:3] if len(colors) >= 3 else [(205, 133, 63), (101, 67, 33)]
    num_patches = rng.randint(8, 15)
    for _ in range(num_patches):
        px = int(rng.random() * diameter)
        py = int(rng.random() * diameter)
        pr = int(radius * (0.2 + rng.random() * 0.4))
        color = rng.choice(patch_colors) if patch_colors else (205, 133, 63)
        pygame.draw.circle(texture, color, (px, py), pr)
    
    # Draw craters
    crater_rim_color = colors[3] if len(colors) > 3 else (169, 169, 169)
    crater_floor_color = colors[4] if len(colors) > 4 else (120, 80, 50)
    num_craters = rng.randint(6, 14)
    for _ in range(num_craters):
        cx = int(rng.random() * diameter)
        cy = int(rng.random() * diameter)
        cr = int(radius * (0.05 + rng.random() * 0.15))
        if cr > 2:
            # Crater rim (lighter)
            pygame.draw.circle(texture, crater_rim_color, (cx, cy), cr)
            # Crater floor (darker, smaller)
            pygame.draw.circle(texture, crater_floor_color, (cx, cy), max(1, int(cr * 0.7)))
    
    # Apply circular mask
    texture = _apply_circular_mask(texture, diameter)
    
    return texture


def _generate_gas_giant(
    diameter: int,
    colors: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> pygame.Surface:
    """Generate a Jupiter/Saturn-like gas giant with bands and storms."""
    radius = diameter / 2.0
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Base color
    base_color = colors[0] if len(colors) > 0 else (201, 144, 57)
    texture.fill(base_color)
    
    # Draw horizontal bands
    band_colors = colors[1:4] if len(colors) >= 4 else [
        (166, 124, 82), (234, 214, 183), (139, 90, 43)
    ]
    num_bands = rng.randint(8, 16)
    band_height = diameter // num_bands
    
    for i in range(num_bands):
        if rng.random() > 0.4:  # Not all bands are visible
            band_y = i * band_height
            color = rng.choice(band_colors) if band_colors else (166, 124, 82)
            # Add some waviness to bands
            wave_offset = int(rng.random() * 4 - 2)
            height_var = band_height + rng.randint(-2, 2)
            pygame.draw.rect(
                texture, color,
                (0, band_y + wave_offset, diameter, max(1, height_var))
            )
    
    # Draw storm spots
    storm_color = colors[4] if len(colors) > 4 else (255, 87, 51)
    num_storms = rng.randint(1, 4)
    for _ in range(num_storms):
        sx = int(rng.random() * diameter)
        sy = int(rng.random() * diameter)
        sw = int(radius * (0.1 + rng.random() * 0.2))
        sh = int(sw * (0.4 + rng.random() * 0.3))
        if sw > 2 and sh > 2:
            storm_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
            pygame.draw.ellipse(storm_surf, (*storm_color, 200), storm_surf.get_rect())
            texture.blit(storm_surf, (sx - sw // 2, sy - sh // 2))
    
    # Apply circular mask
    texture = _apply_circular_mask(texture, diameter)
    
    return texture


def _generate_ice_giant(
    diameter: int,
    colors: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> pygame.Surface:
    """Generate a Neptune/Uranus-like ice giant with smooth gradients."""
    radius = diameter / 2.0
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Base color
    base_color = colors[0] if len(colors) > 0 else (100, 149, 237)
    texture.fill(base_color)
    
    # Create smooth gradient bands
    band_colors = colors[1:5] if len(colors) >= 5 else [
        (72, 209, 204), (176, 224, 230), (70, 130, 180), (135, 206, 250)
    ]
    num_bands = rng.randint(4, 8)
    band_height = diameter // num_bands
    
    for i in range(num_bands):
        if rng.random() > 0.3:
            band_y = i * band_height
            color = rng.choice(band_colors) if band_colors else (72, 209, 204)
            # Softer, more translucent bands
            band_surf = pygame.Surface((diameter, band_height), pygame.SRCALPHA)
            band_surf.fill((*color, 80 + rng.randint(0, 60)))
            texture.blit(band_surf, (0, band_y))
    
    # Add subtle atmospheric swirls
    num_swirls = rng.randint(2, 5)
    for _ in range(num_swirls):
        sx = int(rng.random() * diameter)
        sy = int(rng.random() * diameter)
        sr = int(radius * (0.1 + rng.random() * 0.15))
        if sr > 2:
            swirl_color = rng.choice(band_colors) if band_colors else (176, 224, 230)
            swirl_surf = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
            pygame.draw.circle(swirl_surf, (*swirl_color, 60), (sr, sr), sr)
            texture.blit(swirl_surf, (sx - sr, sy - sr))
    
    # Apply circular mask
    texture = _apply_circular_mask(texture, diameter)
    
    # Subtle atmosphere ring
    atmo_color = colors[0] if len(colors) > 0 else (100, 149, 237)
    pygame.draw.circle(texture, (*atmo_color, 100), (int(radius), int(radius)), int(radius), 1)
    
    return texture


def _generate_molten(
    diameter: int,
    colors: tuple[tuple[int, int, int], ...],
    rng: random.Random,
) -> pygame.Surface:
    """Generate a volcanic/molten planet with lava cracks."""
    radius = diameter / 2.0
    texture = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    
    # Dark rocky base
    base_color = colors[0] if len(colors) > 0 else (30, 30, 30)
    texture.fill(base_color)
    
    # Add rocky texture variation
    rock_color = colors[1] if len(colors) > 1 else (50, 40, 35)
    num_rocks = rng.randint(12, 20)
    for _ in range(num_rocks):
        rx = int(rng.random() * diameter)
        ry = int(rng.random() * diameter)
        rr = int(radius * (0.1 + rng.random() * 0.3))
        pygame.draw.circle(texture, rock_color, (rx, ry), rr)
    
    # Draw lava cracks/veins
    lava_colors = colors[2:5] if len(colors) >= 5 else [
        (255, 69, 0), (255, 140, 0), (255, 215, 0)
    ]
    num_cracks = rng.randint(8, 16)
    for _ in range(num_cracks):
        # Random starting point
        x1 = int(rng.random() * diameter)
        y1 = int(rng.random() * diameter)
        # Random length and direction
        length = int(radius * (0.3 + rng.random() * 0.5))
        angle = rng.random() * 2 * math.pi
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        
        # Draw crack with glow effect
        lava_color = rng.choice(lava_colors) if lava_colors else (255, 69, 0)
        # Outer glow
        pygame.draw.line(texture, (*lava_color, 100), (x1, y1), (x2, y2), 4)
        # Inner bright line
        hot_color = lava_colors[-1] if lava_colors else (255, 215, 0)
        pygame.draw.line(texture, hot_color, (x1, y1), (x2, y2), 2)
    
    # Add glowing pools
    num_pools = rng.randint(3, 7)
    for _ in range(num_pools):
        px = int(rng.random() * diameter)
        py = int(rng.random() * diameter)
        pr = int(radius * (0.05 + rng.random() * 0.1))
        if pr > 1:
            pool_color = rng.choice(lava_colors) if lava_colors else (255, 140, 0)
            pygame.draw.circle(texture, pool_color, (px, py), pr)
            # Brighter center
            hot_color = lava_colors[-1] if lava_colors else (255, 215, 0)
            pygame.draw.circle(texture, hot_color, (px, py), max(1, pr // 2))
    
    # Apply circular mask
    texture = _apply_circular_mask(texture, diameter)
    
    return texture


def _apply_circular_mask(surface: pygame.Surface, diameter: int) -> pygame.Surface:
    """Apply a circular mask to clip the surface to a sphere shape."""
    radius = diameter / 2.0
    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (int(radius), int(radius)), int(radius))
    surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return surface


# =======================
#   SPRITE GENERATION
# =======================
def generate_planet_sprite(planet: Planet, diameter: int) -> pygame.Surface:
    """
    Generate a pygame Surface sprite for the given planet (no caching).
    
    For better performance with zooming, use get_planet_sprite() instead
    which caches generated sprites.
    
    Args:
        planet: The Planet instance to generate a sprite for
        diameter: The diameter in pixels for the generated sprite
    
    Returns:
        A pygame.Surface with the procedurally generated planet
    """
    if diameter <= 0:
        raise ValueError("Diameter must be positive")
    
    rng = random.Random(planet.seed)
    colors = planet.base_colors or DEFAULT_PALETTES.get(planet.planet_type, ())
    
    generators = {
        PlanetType.ROCKY_TERRESTRIAL: _generate_rocky_terrestrial,
        PlanetType.ROCKY_BARREN: _generate_rocky_barren,
        PlanetType.GAS_GIANT: _generate_gas_giant,
        PlanetType.ICE_GIANT: _generate_ice_giant,
        PlanetType.MOLTEN: _generate_molten,
    }
    
    generator = generators.get(planet.planet_type, _generate_rocky_terrestrial)
    return generator(diameter, colors, rng)


def get_planet_sprite(planet: Planet, diameter: int) -> pygame.Surface:
    """
    Get a cached planet sprite, generating it if not already cached.
    
    This function caches generated sprites by (seed, diameter) to avoid
    regenerating the same sprite multiple times during zoom operations.
    
    For very large planets (diameter > 2048), returns a simple circle
    to avoid performance issues.
    
    Args:
        planet: The Planet instance to get a sprite for
        diameter: The diameter in pixels for the sprite
    
    Returns:
        A pygame.Surface with the planet sprite
    
    Raises:
        ValueError: If diameter is not positive
    """
    if diameter <= 0:
        raise ValueError("Diameter must be positive")
    
    # For very large sprites, return a simple circle to avoid performance issues
    if diameter > 2048:
        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        colors = planet.base_colors or DEFAULT_PALETTES.get(planet.planet_type, ())
        base_color = colors[0] if colors else (128, 128, 128)
        radius = diameter // 2
        pygame.draw.circle(surface, base_color, (radius, radius), radius)
        return surface
    
    cache_key = _get_cache_key(planet, diameter)
    
    # Check cache
    cached = _PLANET_SPRITE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    
    # Generate new sprite
    sprite = generate_planet_sprite(planet, diameter)
    
    # Manage cache size - remove oldest entries if cache is full
    if len(_PLANET_SPRITE_CACHE) >= _CACHE_MAX_SIZE:
        # Remove first (oldest) entry
        oldest_key = next(iter(_PLANET_SPRITE_CACHE))
        del _PLANET_SPRITE_CACHE[oldest_key]
    
    # Store in cache
    _PLANET_SPRITE_CACHE[cache_key] = sprite
    
    return sprite


# =======================
#   PRESET PLANETS
# =======================
# Realistic solar system bodies with accurate physical properties
# Data sources: NASA planetary fact sheets

PRESET_PLANETS: dict[str, Planet] = {
    # Rocky Terrestrial
    "earth": Planet(
        name="Earth",
        planet_type=PlanetType.ROCKY_TERRESTRIAL,
        radius=6_371_000,           # 6,371 km
        mass=5.972e24,              # kg
        seed=42,                    # Consistent Earth appearance
    ),
    "venus": Planet(
        name="Venus",
        planet_type=PlanetType.ROCKY_TERRESTRIAL,
        radius=6_051_800,           # 6,051.8 km
        mass=4.867e24,              # kg
        base_colors=(
            (205, 175, 149),        # Yellowish cloud base
            (218, 165, 32),         # Goldenrod
            (255, 215, 0),          # Gold
            (255, 248, 220),        # Cornsilk clouds
            (255, 235, 205),        # Blanched almond atmosphere
        ),
        seed=101,
    ),
    
    # Rocky Barren
    "mars": Planet(
        name="Mars",
        planet_type=PlanetType.ROCKY_BARREN,
        radius=3_389_500,           # 3,389.5 km
        mass=6.39e23,               # kg
        base_colors=(
            (193, 68, 14),          # Mars red-orange
            (210, 105, 30),         # Chocolate
            (139, 69, 19),          # Saddle brown
            (160, 82, 45),          # Sienna (craters)
            (120, 60, 30),          # Dark brown
        ),
        seed=201,
    ),
    "moon": Planet(
        name="Moon",
        planet_type=PlanetType.ROCKY_BARREN,
        radius=1_737_400,           # 1,737.4 km
        mass=7.342e22,              # kg
        base_colors=(
            (128, 128, 128),        # Gray base
            (169, 169, 169),        # Dark gray
            (105, 105, 105),        # Dim gray
            (192, 192, 192),        # Silver (crater rims)
            (80, 80, 80),           # Dark (crater floors)
        ),
        seed=301,
    ),
    "mercury": Planet(
        name="Mercury",
        planet_type=PlanetType.ROCKY_BARREN,
        radius=2_439_700,           # 2,439.7 km
        mass=3.285e23,              # kg
        base_colors=(
            (112, 128, 144),        # Slate gray
            (119, 136, 153),        # Light slate gray
            (105, 105, 105),        # Dim gray
            (169, 169, 169),        # Dark gray (craters)
            (70, 70, 70),           # Very dark gray
        ),
        seed=401,
    ),
    
    # Gas Giants
    "jupiter": Planet(
        name="Jupiter",
        planet_type=PlanetType.GAS_GIANT,
        radius=69_911_000,          # 69,911 km
        mass=1.898e27,              # kg
        base_colors=(
            (201, 144, 57),         # Jupiter tan
            (166, 124, 82),         # Brown band
            (234, 214, 183),        # Light cream band
            (139, 90, 43),          # Dark band
            (205, 92, 92),          # Great Red Spot
        ),
        seed=501,
    ),
    "saturn": Planet(
        name="Saturn",
        planet_type=PlanetType.GAS_GIANT,
        radius=58_232_000,          # 58,232 km
        mass=5.683e26,              # kg
        base_colors=(
            (210, 180, 140),        # Tan base
            (238, 232, 170),        # Pale goldenrod
            (245, 245, 220),        # Beige
            (189, 183, 107),        # Dark khaki
            (218, 165, 32),         # Goldenrod storms
        ),
        seed=601,
    ),
    
    # Ice Giants
    "uranus": Planet(
        name="Uranus",
        planet_type=PlanetType.ICE_GIANT,
        radius=25_362_000,          # 25,362 km
        mass=8.681e25,              # kg
        base_colors=(
            (175, 238, 238),        # Pale turquoise
            (127, 255, 212),        # Aquamarine
            (224, 255, 255),        # Light cyan
            (152, 251, 152),        # Pale green
            (176, 224, 230),        # Powder blue
        ),
        seed=701,
    ),
    "neptune": Planet(
        name="Neptune",
        planet_type=PlanetType.ICE_GIANT,
        radius=24_764_000,          # 24,764 km
        mass=1.024e26,              # kg
        base_colors=(
            (65, 105, 225),         # Royal blue
            (100, 149, 237),        # Cornflower blue
            (72, 61, 139),          # Dark slate blue
            (106, 90, 205),         # Slate blue
            (135, 206, 250),        # Light sky blue
        ),
        seed=801,
    ),
    
    # Fictional/exotic examples
    "io": Planet(
        name="Io",
        planet_type=PlanetType.MOLTEN,
        radius=1_821_600,           # 1,821.6 km (Jupiter's volcanic moon)
        mass=8.932e22,              # kg
        base_colors=(
            (255, 215, 0),          # Gold/sulfur base
            (255, 165, 0),          # Orange
            (255, 69, 0),           # Red-orange lava
            (255, 140, 0),          # Dark orange
            (255, 255, 0),          # Yellow (hottest)
        ),
        seed=901,
    ),
}


def get_preset(name: str) -> Planet:
    """
    Get a preset planet by name.
    
    Args:
        name: Name of the preset (case-insensitive). 
              Available: earth, venus, mars, moon, mercury, 
                        jupiter, saturn, uranus, neptune, io
    
    Returns:
        A copy of the preset Planet
    
    Raises:
        KeyError: If the preset name is not found
    """
    key = name.lower()
    if key not in PRESET_PLANETS:
        available = ", ".join(sorted(PRESET_PLANETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    
    preset = PRESET_PLANETS[key]
    # Return a copy to avoid modifying the original
    return Planet(
        name=preset.name,
        planet_type=preset.planet_type,
        radius=preset.radius,
        mass=preset.mass,
        base_colors=preset.base_colors,
        seed=preset.seed,
    )


def list_presets() -> list[str]:
    """Return a list of available preset planet names."""
    return sorted(PRESET_PLANETS.keys())