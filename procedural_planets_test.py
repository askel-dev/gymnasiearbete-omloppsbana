import pygame
import moderngl
import struct
import random
import numpy as np
import math

# --- Constants ---
INITIAL_SCALE = 800.0 
# Increased precision limit to reset BEFORE jitter becomes noticeable
PRECISION_LIMIT = 0.05 
ROTATION_SPEED = 0.04      

# --- GLSL Shaders ---

vertex_shader = '''
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
'''

fragment_shader = '''
#version 330
out vec4 f_color;
in vec2 v_uv;

uniform vec2 u_resolution;
uniform float u_scale;
uniform vec2 u_mouse_offset;

uniform float u_old_scale;
uniform vec2 u_old_offset;

uniform float u_time;
uniform vec2 u_seed;
uniform vec2 u_old_seed;
uniform float u_fade;
uniform float u_mood;    // 0.0 = Day, 1.0 = Night
uniform int u_show_borders;
uniform int u_show_caves;

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

vec3 voronoi(vec2 x, vec2 seed) {
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 mg, mr;
    float md = 8.0;
    for(int j=-1; j<=1; j++) {
        for(int i=-1; i<=1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g + seed);
            o = 0.5 + 0.5 * sin(u_time * 0.1 + 6.2831 * o);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if(d < md) { md = d; mr = r; mg = g; }
        }
    }
    md = 8.0;
    for(int j=-2; j<=2; j++) {
        for(int i=-2; i<=2; i++) {
            vec2 g = mg + vec2(float(i), float(j));
            vec2 o = hash2(n + g + seed);
            o = 0.5 + 0.5 * sin(u_time * 0.1 + 6.2831 * o);
            vec2 r = g + o - f;
            if(dot(mr-r, mr-r) > 0.00001)
                md = min(md, dot(0.5*(mr+r), normalize(r-mr)));
        }
    }
    return vec3(md, mr);
}

vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ; m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float fBm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 8; i++) {
        value += amplitude * snoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

vec3 get_biome_color(float elev, float moist, float temp) {
    vec3 col;
    if (elev < -0.25) col = vec3(15, 30, 80)/255.0;
    else if (elev < 0.0) col = vec3(35, 70, 140)/255.0;
    else if (elev < 0.08) col = vec3(240, 220, 160)/255.0;
    else {
        temp -= (elev * 0.8);
        if (temp < -0.3) col = (moist < -0.2) ? vec3(140, 130, 120)/255.0 : vec3(230, 240, 255)/255.0;
        else if (temp < 0.3) {
            if (moist < -0.3) col = vec3(140, 160, 110)/255.0;
            else if (moist < 0.4) col = vec3(40, 100, 40)/255.0;
            else col = vec3(10, 60, 10)/255.0;
        } else {
            if (moist < -0.4) col = vec3(210, 170, 100)/255.0;
            else if (moist < 0.2) col = vec3(130, 150, 70)/255.0;
            else col = vec3(30, 80, 20)/255.0;
        }
    }

    // Aggressive night dimming + blue shift
    vec3 night_filter = col * vec3(0.1, 0.15, 0.35); 
    return mix(col, night_filter, u_mood);
}

vec3 render_world(vec2 st, vec2 seed) {
    vec3 v = voronoi(st * 0.1, seed); 
    float border_dist = v.x;
    float elev = fBm(st * 0.4);
    elev += smoothstep(0.2, 0.0, border_dist) * 0.6;
    float moist = fBm(st * 0.2 + seed * 1.5);
    float temp = fBm(st * 0.1 - seed * 2.0);
    
    vec3 color = get_biome_color(elev, moist, temp);
    
    if (u_show_borders == 1) {
        float pulse = 0.5 + 0.5 * sin(u_time * 2.0);
        float border_line = smoothstep(0.04, 0.0, border_dist);
        vec3 glow = mix(vec3(0.0), vec3(0.3, 0.6, 1.0) * pulse, u_mood);
        color = mix(color, glow, border_line * (0.4 + u_mood * 0.5));
    }
    
    if (u_show_caves == 1 && elev > 0.3) {
        float cave_pattern = snoise(st * 20.0);
        if (cave_pattern > 0.4) color *= 0.15; 
    }
    
    return color;
}

void main() {
    float angle = u_time * 0.04; 
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    vec2 uv_centered = (gl_FragCoord.xy - u_resolution.xy * 0.5);

    // Coordinate space for new world
    vec2 st_new = (rot * (uv_centered / u_scale)) + u_mouse_offset;
    vec3 color_new = render_world(st_new, u_seed);

    if (u_fade > 0.001) {
        // Coordinate space for old world (maintaining its scale and offset)
        vec2 st_old = (rot * (uv_centered / u_old_scale)) + u_old_offset;
        vec3 color_old = render_world(st_old, u_old_seed);
        f_color = vec4(mix(color_new, color_old, u_fade), 1.0);
    } else {
        f_color = vec4(color_new, 1.0);
    }
    
    // Vignette
    float d = length(v_uv - 0.5);
    f_color.rgb *= smoothstep(0.85, 0.35, d);
}
'''

def main():
    pygame.init()
    screen_info = pygame.display.Info()
    WIDTH, HEIGHT = screen_info.current_w, screen_info.current_h
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=1)
    
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
    vbo = ctx.buffer(vertices)
    vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])

    # State
    world_offset = [0.0, 0.0]
    old_offset = [0.0, 0.0]
    auto_pan_vel = [0.0, 0.0]
    current_scale = INITIAL_SCALE
    old_scale = INITIAL_SCALE
    
    is_dragging = False
    last_mouse_pos = (0, 0)
    auto_zoom = True
    zoom_rate = 1.0015
    
    current_seed = (random.uniform(-5000, 5000), random.uniform(-5000, 5000))
    old_seed = current_seed
    fade_value = 0.0
    mood_value = 0.0 
    mood_target = 0.0
    
    start_time = pygame.time.get_ticks()

    def reset_world(smooth=False):
        nonlocal current_scale, old_scale, world_offset, old_offset, current_seed, old_seed, fade_value, zoom_rate, mood_target, auto_pan_vel
        if smooth:
            old_seed = current_seed
            old_scale = current_scale
            old_offset = list(world_offset)
            fade_value = 1.0 
        else:
            fade_value = 0.0
            
        current_seed = (random.uniform(-9000, 9000), random.uniform(-9000, 9000))
        current_scale = INITIAL_SCALE
        world_offset = [random.uniform(-500, 500), random.uniform(-500, 500)]
        
        zoom_rate = random.uniform(1.0012, 1.0025)
        mood_target = random.choice([0.0, 1.0])
        auto_pan_vel = [random.uniform(-0.012, 0.012), random.uniform(-0.012, 0.012)]
        
        prog['u_seed'].value = current_seed
        prog['u_old_seed'].value = old_seed
        prog['u_old_scale'].value = old_scale
        prog['u_old_offset'].value = tuple(old_offset)

    prog['u_resolution'].value = (WIDTH, HEIGHT)
    prog['u_show_borders'].value = 1
    prog['u_show_caves'].value = 1
    reset_world()

    running = True
    clock = pygame.time.Clock()

    while running:
        t = (pygame.time.get_ticks() - start_time) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            
            if event.type == pygame.MOUSEWHEEL:
                current_scale *= (1.1 if event.y > 0 else 0.9)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_dragging = True
                    last_mouse_pos = event.pos
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: is_dragging = False

            if event.type == pygame.MOUSEMOTION and is_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                angle = t * ROTATION_SPEED
                gl_dy = -dy 
                rotated_dx = dx * math.cos(angle) - gl_dy * math.sin(angle)
                rotated_dy = dx * math.sin(angle) + gl_dy * math.cos(angle)
                world_offset[0] -= rotated_dx / current_scale
                world_offset[1] -= rotated_dy / current_scale
                last_mouse_pos = event.pos

        if auto_zoom:
            current_scale /= zoom_rate
            world_offset[0] += auto_pan_vel[0]
            world_offset[1] += auto_pan_vel[1]
            
            # Now resets much earlier (at 0.00005) so you never see the glitchy noise phase
            if current_scale < PRECISION_LIMIT:
                reset_world(smooth=True)

        if fade_value > 0:
            # Cinematic cross-fade speed
            fade_value -= 0.006 
            if fade_value < 0: fade_value = 0
            
        mood_value += (mood_target - mood_value) * 0.004

        # Update Uniforms
        prog['u_scale'].value = current_scale
        prog['u_mouse_offset'].value = tuple(world_offset)
        prog['u_time'].value = t
        prog['u_fade'].value = fade_value
        prog['u_mood'].value = mood_value

        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()