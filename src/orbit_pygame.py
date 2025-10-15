# src/orbit_pygame.py
import math
import time
import pygame
import sys
import numpy as np
from collections import deque
from itertools import islice

# =======================
#   FYSIK & KONSTANTER
# =======================
G = 6.674e-11                 # gravitationskonstant (SI)
M = 5.972e24                  # Jordens massa (kg)
MU = G * M
EARTH_RADIUS = 6_371_000      # m
# Startvillkor
R0 = np.array([7_000_000.0, 0.0])  # m
V0 = np.array([0.0, 7_800.0])      # m/s

# =======================
#   SIMULATOR-SETTINGS
# =======================
DT_PHYS = 0.25                 # fysikens tidssteg (sekunder)
REAL_TIME_SPEED = 60.0        # sim-sek per real-sek (startvärde)
MAX_SUBSTEPS = 20             # skydd mot för många fysiksteg/frame
TRAIL_MAX = 99999              # punkter i spår
TRAIL_DRAW_MAX = 2000          # max punkter att rita per frame

# =======================
#   RIT- & KONTROLL-SETTINGS
# =======================
WIDTH, HEIGHT = 1000, 800
BG_COLOR = (10, 12, 18)
EARTH_COLOR = (70, 170, 255)
SAT_COLOR = (255, 230, 120)
TRAIL_COLOR = (120, 210, 180)
HUD_COLOR = (220, 230, 240)
VEL_COLOR = (255, 120, 120)

PIXELS_PER_METER = 0.5 * (min(WIDTH, HEIGHT) / (2.0 * np.linalg.norm(R0)))
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

def world_to_screen(x, y, ppm):
    sx = WIDTH//2 + int(x * ppm)
    sy = HEIGHT//2 - int(y * ppm)
    return sx, sy

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# =======================
#   HUVUDPROGRAM
# =======================
def main():
    pygame.init()
    pygame.display.set_caption("Omloppsbana i realtid (Pygame + RK4)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Simuleringsstate
    r = R0.copy()
    v = V0.copy()
    t_sim = 0.0
    paused = False
    show_trail = True
    trail = deque(maxlen=TRAIL_MAX)
    ppm = PIXELS_PER_METER
    real_time_speed = REAL_TIME_SPEED

    vel_vector_scale = 500.0  # endast visuellt

    # tidsackumulator för fast fysik
    accumulator = 0.0
    last_time = time.perf_counter()

    def reset():
        nonlocal r, v, t_sim, trail, paused, ppm, real_time_speed, accumulator, last_time
        r = R0.copy()
        v = V0.copy()
        t_sim = 0.0
        trail = deque(maxlen=TRAIL_MAX)
        paused = False
        ppm = PIXELS_PER_METER
        real_time_speed = REAL_TIME_SPEED
        accumulator = 0.0
        last_time = time.perf_counter()

    # ========= LOOP =========
    while True:
        # --- Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_t:
                    show_trail = not show_trail
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    ppm = clamp(ppm * 1.2, MIN_PPM, MAX_PPM)     # zoom in
                elif event.key == pygame.K_MINUS:
                    ppm = clamp(ppm / 1.2, MIN_PPM, MAX_PPM)     # zoom out

                # ---- Piltangenter styr simhastighet (ingen boost längre) ----
                elif event.key == pygame.K_RIGHT:
                    real_time_speed = min(real_time_speed * 1.5, 10_000.0)
                elif event.key == pygame.K_LEFT:
                    real_time_speed = max(real_time_speed / 1.5, 0.1)
                elif event.key == pygame.K_UP:
                    real_time_speed = min(real_time_speed * 2.0, 10_000.0)
                elif event.key == pygame.K_DOWN:
                    real_time_speed = max(real_time_speed / 2.0, 0.1)

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
                if show_trail:
                    trail.append((r[0], r[1]))

            accumulator = 0.0

        # --- Render ---
        screen.fill(BG_COLOR)

        # Jorden
        earth_screen_pos = world_to_screen(0.0, 0.0, ppm)
        earth_px = max(2, int(EARTH_RADIUS * ppm))
        pygame.draw.circle(screen, EARTH_COLOR, earth_screen_pos, earth_px)

        # Spår
        if show_trail and len(trail) >= 2:
            if len(trail) > TRAIL_DRAW_MAX:
                step = math.ceil(len(trail) / TRAIL_DRAW_MAX)
                sampled_trail = list(islice(trail, 0, len(trail), step))
                if sampled_trail[-1] != trail[-1]:
                    sampled_trail.append(trail[-1])
            else:
                sampled_trail = list(trail)

            pts = [world_to_screen(x, y, ppm) for (x, y) in sampled_trail]
            pygame.draw.lines(screen, TRAIL_COLOR, False, pts, 2)

        # Satellit
        sat_pos = world_to_screen(r[0], r[1], ppm)
        pygame.draw.circle(screen, SAT_COLOR, sat_pos, 5)

        # Hastighetsvektor (visuell)
        vx, vy = v
        v_end = world_to_screen(r[0] + vx*vel_vector_scale, r[1] + vy*vel_vector_scale, ppm)
        pygame.draw.line(screen, VEL_COLOR, sat_pos, v_end, 2)

        # HUD
        rmag = np.linalg.norm(r)
        vmag = np.linalg.norm(v)
        eps = energy_specific(r, v)
        e = eccentricity(r, v)
        hud_lines = [
            f"t = {t_sim:,.0f} s    (speed: {real_time_speed:.2f}x)",
            f"|r| = {rmag/1e6:.2f} Mm    |v| = {vmag:,.1f} m/s",
            f"energy = {eps: .3e} J/kg    e = {e:.4f}",
            "SPACE: paus | R: reset | T: trail | +/-: zoom",
            "←/→: -/+ 1.5x speed   ↑/↓: ×2/÷2 speed   ESC: quit",
        ]
        y = 10
        for line in hud_lines:
            surf = font.render(line, True, HUD_COLOR)
            screen.blit(surf, (10, y))
            y += 20

        pygame.display.flip()
        clock.tick()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()

