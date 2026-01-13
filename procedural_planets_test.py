import pygame
import moderngl
import struct
import numpy as np
import random

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
uniform vec2 u_offset;
uniform vec2 u_seed;

// --- Noise Functions ---

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
    if (elev < -0.25) col = vec3(15, 30, 80)/255.0;       // Deep Ocean
    else if (elev < 0.0) col = vec3(35, 70, 140)/255.0;   // Shallow Ocean
    else if (elev < 0.08) col = vec3(240, 220, 160)/255.0; // Beach
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
    return col;
}

void main() {
    vec2 uv_centered = (gl_FragCoord.xy - u_resolution.xy * 0.5);
    vec2 st = (uv_centered / u_scale) + u_offset;

    // Generate World Data
    vec3 v = voronoi(st * 0.1, u_seed); 
    float border_dist = v.x;
    
    float elev = fBm(st * 0.4);
    
    // I kept this line because it shapes the mountains, 
    // but I removed the line that drew the black outline on top of it.
    elev += smoothstep(0.2, 0.0, border_dist) * 0.6; 
    
    float moist = fBm(st * 0.2 + u_seed * 1.5);
    float temp = fBm(st * 0.1 - u_seed * 2.0);
    
    vec3 color = get_biome_color(elev, moist, temp);
    
    f_color = vec4(color, 1.0);
}
'''

def main():
    pygame.init()
    screen_info = pygame.display.Info()
    WIDTH, HEIGHT = screen_info.current_w, screen_info.current_h
    
    # --- CHANGE: Added vsync=1 to the display set_mode ---
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN, vsync=1)
    
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
    vbo = ctx.buffer(vertices)
    vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])

    scale = 300.0  
    offset = [0.0, 0.0] 
    seed = (random.uniform(-100, 100), random.uniform(-100, 100)) 

    is_dragging = False
    last_mouse_pos = (0, 0)

    prog['u_resolution'].value = (WIDTH, HEIGHT)
    prog['u_seed'].value = seed
    
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            
            if event.type == pygame.MOUSEWHEEL:
                scale *= (1.1 if event.y > 0 else 0.9)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    is_dragging = True
                    last_mouse_pos = event.pos
                if event.button == 3: 
                    seed = (random.uniform(-100, 100), random.uniform(-100, 100))
                    prog['u_seed'].value = seed
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: is_dragging = False

            if event.type == pygame.MOUSEMOTION and is_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                offset[0] -= dx / scale
                offset[1] += dy / scale 
                last_mouse_pos = event.pos

        prog['u_scale'].value = scale
        prog['u_offset'].value = tuple(offset)

        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()
        
        # Reduced tick rate slightly since vsync handles the cap usually, 
        # but 120 keeps physics smooth if we add it later.
        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()