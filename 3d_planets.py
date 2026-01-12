import pygame
import numpy as np
from opensimplex import OpenSimplex

# --- Constants ---
WIDTH, HEIGHT = 800, 600
PLANET_RADIUS = 200
MAP_WIDTH = 1000  # The width of the texture we will wrap
MAP_HEIGHT = 500  # The height of the texture
LIGHT_DIR = np.array([0.5, 0.5, 1.0]) # Direction of the sun
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR) # Normalize it

# --- Colors ---
C_OCEAN = (30, 60, 150)
C_LAND = (100, 180, 80)
C_SNOW = (255, 255, 255)
C_SPACE = (10, 10, 15)

def generate_texture():
    """Generates a flat 2D noise map (like your previous code) to use as a texture."""
    print("Generating Texture Map...")
    gen = OpenSimplex(seed=123)
    texture_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
    pixels = pygame.PixelArray(texture_surface)
    
    # Simple noise loop to create a texture (Faster version)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            # Wrap x so the texture matches left-to-right (Seamless)
            nx = np.cos(x / MAP_WIDTH * 2 * np.pi) 
            ny = np.sin(x / MAP_WIDTH * 2 * np.pi) 
            nz = y / MAP_HEIGHT
            
            # Sample 3D noise to make it seamless on the X-axis
            val = gen.noise3(nx, ny, nz * 2)
            
            if val < -0.2: color = C_OCEAN
            elif val > 0.4: color = C_SNOW
            else: color = C_LAND
            pixels[x, y] = color
    
    pixels.close()
    return texture_surface

def precompute_sphere_data(radius):
    """
    Creates a grid of (x,y) and calculates the Z depth and Normals 
    for the sphere ONCE, so we don't do it every frame.
    """
    # Create a grid of X and Y coordinates centered at 0
    y, x = np.ogrid[-radius:radius, -radius:radius]
    
    # Mask: True where pixels are inside the circle
    mask = x**2 + y**2 < radius**2
    
    # Calculate Z depth: z = sqrt(r^2 - x^2 - y^2)
    # We use 'maximum' to avoid negative numbers in sqrt (for pixels outside circle)
    z = np.sqrt(np.maximum(radius**2 - x**2 - y**2, 0))
    
    # Normal vectors (Normalized X, Y, Z) for lighting
    # We stack them into a (Size, Size, 3) array
    norm_x = x / radius
    norm_y = y / radius
    norm_z = z / radius
    normals = np.dstack((norm_x, norm_y, norm_z))
    
    # Lighting intensity (Dot Product)
    # How much does the face point towards the light?
    intensity = np.dot(normals, LIGHT_DIR)
    intensity = np.maximum(intensity, 0.1) # Ambient light (min brightness)
    
    # Sphere UV Mapping (Math to turn Sphere XYZ -> Flat UV)
    # V (Y-axis texture coordinate) is based on the Y height on sphere
    v_coords = (np.arcsin(norm_y) / np.pi) + 0.5 
    
    # U (X-axis texture coordinate) needs to be calculated dynamically for rotation
    # so we return the raw Norm_X and Norm_Z to calculate U in the loop
    return mask, intensity, norm_x, norm_z, v_coords

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    # 1. Generate the Planet Texture
    texture = generate_texture()
    texture_array = pygame.surfarray.array3d(texture) # Convert to numpy for fast access
    
    # 2. Pre-calculate the Sphere Math
    mask, intensity, nx, nz, v_coords = precompute_sphere_data(PLANET_RADIUS)
    
    rotation = 0.0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Render Logic ---
        
        # 1. Calculate Rotation
        rotation += 0.02
        
        # 2. Calculate U Coordinates (Longitude) based on rotation
        # Formula: atan2(z, x) gives the angle around the sphere
        raw_u = np.arctan2(nz, nx) / (2 * np.pi)
        
        # Add rotation and modulo 1.0 to wrap around
        u_coords = (raw_u + rotation) % 1.0
        
        # 3. Map UVs to Texture Coordinates (Integers)
        # u_coords (0.0-1.0) -> texture_x (0-1000)
        tex_x = (u_coords * (MAP_WIDTH - 1)).astype(int)
        tex_y = (v_coords * (MAP_HEIGHT - 1)).astype(int)
        
        # 4. Lookup Colors from Texture using Numpy Indexing (Very Fast)
        # We grab the pixels from the texture array using our calculated X/Y indices
        rendered_planet = texture_array[tex_x, tex_y]
        
        # 5. Apply Lighting (Multiply color by intensity)
        # We need to reshape intensity to match the color array shape (r,g,b)
        rendered_planet = rendered_planet * intensity[:, :, np.newaxis]
        
        # 6. Apply Mask (Cut out the circle)
        # Set pixels outside the sphere to background color
        # (This is a simplified way to handle the background)
        final_planet_surf = pygame.surfarray.make_surface(rendered_planet.astype(np.uint8))
        final_planet_surf.set_colorkey((0,0,0)) # Make black parts transparent if needed
        
        # --- Drawing ---
        screen.fill(C_SPACE)
        
        # Center the planet
        dest_x = WIDTH//2 - PLANET_RADIUS
        dest_y = HEIGHT//2 - PLANET_RADIUS
        
        # We created a square surface, we need to mask it to a circle
        # Pygame doesn't have a simple alpha mask for surfaces, so we draw a circle mask
        # Alternatively, since we calculated 'mask' earlier:
        
        # Create a surface from the numpy array
        surf = pygame.surfarray.make_surface(rendered_planet)
        
        # Apply the circular mask (hacky way for Pygame)
        # We create a new surface with an alpha channel
        pixel_alpha = np.zeros((PLANET_RADIUS*2, PLANET_RADIUS*2), dtype=np.uint8)
        pixel_alpha[mask] = 255 # Opaque inside circle
        
        # We need to copy the alpha into the surface (Requires pixel copy)
        # Optimization: Just draw a black "frame" with a transparent hole on top?
        # Let's just blit the square and assume space is black (lazy but works)
        
        screen.blit(surf, (dest_x, dest_y))
        
        # Draw a black "porthole" to hide the corners of the square image
        # (Because our numpy math calculated corners as garbage data usually)
        # A simple trick: Draw a thick black circle outline outside the planet
        pygame.draw.circle(screen, C_SPACE, (WIDTH//2, HEIGHT//2), PLANET_RADIUS+50, 50)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()