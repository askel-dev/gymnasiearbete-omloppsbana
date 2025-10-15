import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Fysik & startvillkor ---
G = 6.674e-11               # gravitationskonstant (SI)
M = 5.972e24                # central massa (Jorden) i kg

# Startposition (m): ~ jordradie + 600 km, på x-axeln
r0 = np.array([7.0e6, 0.0])
# Starthastighet (m/s): nästan låg omloppshastighet i y-led
v0 = np.array([0.0, 7800.0])

# Numerik
dt = 0.25                    # fysikens tidssteg (sekunder)
real_time_speed = 60        # simulera 60 s per verklig sekund (≈ snabbspolning)
trail_length = 2000         # antal punkter i spåret

# --- Hjälpfunktioner ---
def acceleration(r):
    rmag = np.linalg.norm(r)
    return -G * M * r / (rmag**3)

def rk4_step(r, v, dt):
    a1 = acceleration(r);              k1_r = v;              k1_v = a1
    a2 = acceleration(r + 0.5*dt*k1_r);k2_r = v + 0.5*dt*k1_v;k2_v = a2
    a3 = acceleration(r + 0.5*dt*k2_r);k3_r = v + 0.5*dt*k2_v;k3_v = a3
    a4 = acceleration(r + dt*k3_r);    k4_r = v + dt*k3_v;    k4_v = a4
    r_next = r + (dt/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_next, v_next

# --- Figur ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', 'box')
axis_scale = 1.2 * np.linalg.norm(r0)
ax.set_xlim(-axis_scale, axis_scale)
ax.set_ylim(-axis_scale, axis_scale)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Omloppsbana i realtid (RK4)')

earth,     = ax.plot([0], [0], 'o', markersize=8, label='Jorden')
satellite, = ax.plot([], [], 'o', markersize=5, label='Satellit')
trail,     = ax.plot([], [], lw=1, alpha=0.8, label='Bana (spår)')
vel_line,  = ax.plot([], [], lw=1, alpha=0.6, label='Hastighet')
time_text  = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
ax.legend(loc='upper right')

# --- Simuleringsstate ---
r = r0.copy()
v = v0.copy()
trail_x, trail_y = [], []
t = 0.0

def init():
    satellite.set_data([], [])
    trail.set_data([], [])
    vel_line.set_data([], [])
    time_text.set_text('')
    return satellite, trail, vel_line, time_text

def update(_frame):
    global r, v, t, trail_x, trail_y

    # Kör flera fysiksteg per bildruta för "realtidskänsla"
    # ~30 fps => n_steps ≈ real_time_speed/30
    n_steps = max(1, int(real_time_speed))
    for _ in range(n_steps):
        r, v = rk4_step(r, v, dt)
        t += dt

    # Uppdatera satellitpunkt
    satellite.set_data([r[0]], [r[1]])

    # Uppdatera spår
    trail_x.append(r[0]); trail_y.append(r[1])
    if len(trail_x) > trail_length:
        trail_x = trail_x[-trail_length:]
        trail_y = trail_y[-trail_length:]
    trail.set_data(trail_x, trail_y)

    # Hastighetsvektor (kort linje från satelliten)
    scale = 500.0
    vx, vy = v
    x2, y2 = r[0] + vx*scale, r[1] + vy*scale
    vel_line.set_data([r[0], x2], [r[1], y2])

    # Håll allt i bild (enkel auto-zoom)
    max_extent = max(np.abs(r[0]), np.abs(r[1]), axis_scale)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    # Info HUD
    rmag = np.linalg.norm(r); vmag = np.linalg.norm(v)
    time_text.set_text(f't = {t:,.0f} s\n|r| = {rmag/1e6:.2f} Mm\n|v| = {vmag:.1f} m/s')

    return satellite, trail, vel_line, time_text

anim = FuncAnimation(fig, update, init_func=init, interval=33, blit=True)  # ~30 fps
plt.show()