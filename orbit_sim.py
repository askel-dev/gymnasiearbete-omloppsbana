import numpy as np
import matplotlib.pyplot as plt

# Konstanter
G = 6.674e-11          # Gravitationskonstant
M = 5.972e24           # Jordens massa (kg)
r0 = np.array([7.0e6, 0.0])  # Startposition (m)
v0 = np.array([0.0, 7800.0]) # Start-hastighet (m/s), 7800 ≈ låg omloppshastighet

# Tidssteg
dt = 1.0               # sekunder
t_max = 6000           # total tid (s)

# Arrayer för lagring
positions = []
velocities = []

r = r0.copy()
v = v0.copy()

for _ in range(int(t_max / dt)):
    # Beräkna gravitationskraftens riktning
    r_mag = np.linalg.norm(r)
    a = -G * M * r / r_mag**3   # acceleration
    
    # Uppdatera med enkel Euler
    v += a * dt
    r += v * dt
    
    positions.append(r.copy())
    velocities.append(v.copy())

# Konvertera till numpy-arrays
positions = np.array(positions)

# Visualisera banan
plt.figure(figsize=(6,6))
plt.plot(positions[:,0], positions[:,1], label="Bana")
plt.plot(0, 0, 'yo', label="Jorden")
plt.xlabel("x-position (m)")
plt.ylabel("y-position (m)")
plt.title("Omloppsbana runt Jorden (enkel simulering)")
plt.legend()
plt.axis("equal")
plt.show()
