import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 7th harmonic demonstration
#
# Stator frame (αβ):
#   - Fundamental:  +ω
#   - 7th harmonic: +7ω   (positive sequence, forward rotating field)
#
# dq frame:
#   dq rotates with +ω
#   => 7th harmonic appears as:
#      (+7ω) − (+ω) = +6ω
# ============================================================

# -------------------------
# Parameters
# -------------------------
fps = 60
T = 10
t = np.linspace(0, T, int(T * fps))

w = 2 * np.pi * 0.5   # 1 Hz base electrical speed (slow and visible)

A1 = 1.0            # amplitude fundamental
A5 = 0.6            # amplitude 5th harmonic

# -------------------------
# Angles
# -------------------------
theta1 =  w * t       # fundamental, forward
theta5 =  -5 * w * t   # 5th harmonic, forward (positive sequence)

# -------------------------
# Space vectors in αβ
# -------------------------
x1 = A1 * np.cos(theta1)
y1 = A1 * np.sin(theta1)

x5 = A5 * np.cos(theta5)
y5 = A5 * np.sin(theta5)

# ============================================================
# Animation 1: Stator frame (αβ)
# ============================================================

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.set_aspect("equal")
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)
ax1.set_title("Stator frame (αβ)\nGrundschwingung (+ω) und 7. Harmonische (+7ω)")

line1, = ax1.plot([], [], linewidth=2, label="Grundschwingung (+ω)")
line5, = ax1.plot([], [], linewidth=2, linestyle="--",
                  label="5. Harmonische (+5ω)")
ax1.legend(loc="upper right")

def init1():
    line1.set_data([], [])
    line5.set_data([], [])
    return line1, line5

def update1(i):
    line1.set_data([0, x1[i]], [0, y1[i]])
    line5.set_data([0, x5[i]], [0, y5[i]])
    return line1, line5

ani1 = FuncAnimation(fig1, update1,
                     frames=len(t),
                     init_func=init1,
                     interval=1000 / fps,
                     blit=True)

plt.show()

# ============================================================
# Animation 2: dq frame
# dq frame rotates with +ω
# ============================================================

# Park transformation of the 5th harmonic into dq
xd5 =  x5 * np.cos(theta1) + y5 * np.sin(theta1)
yq5 = -x5 * np.sin(theta1) + y5 * np.cos(theta1)

# ------------------------------------------------------------
# Umdrehungen der GRUNDWELLE zählen
# ------------------------------------------------------------
theta_base = np.unwrap(theta1)

turn_index = np.floor(theta_base / (2*np.pi)).astype(int)
turn_event = np.zeros_like(turn_index, dtype=bool)
turn_event[1:] = turn_index[1:] != turn_index[:-1]

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_aspect("equal")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True)

ax2.set_title("dq frame\nroter Punkt = eine elektrische Umdrehung der Grundwelle")

line_dq, = ax2.plot([], [], linewidth=2, color="red",
                    label="5. Harmonische im dq-System")
dot, = ax2.plot([], [], "ro", markersize=8,
                label="Grundwelle: 2π")

ax2.legend(loc="upper right")

# Wie lange der rote Punkt sichtbar bleibt (in Frames)
flash_length = int(0.15 * fps)
flash_counter = 0

def init2():
    line_dq.set_data([], [])
    dot.set_data([], [])
    return line_dq, dot

def update2(i):
    global flash_counter

    # dq-Zeiger der Harmonischen
    line_dq.set_data([0, xd5[i]], [0, yq5[i]])

    # Prüfen: Grundwelle hat eine volle Umdrehung gemacht?
    if turn_event[i]:
        flash_counter = flash_length

    # Roter Punkt kurz anzeigen
    if flash_counter > 0:
        dot.set_data([0], [0])
        flash_counter -= 1
    else:
        dot.set_data([], [])

    return line_dq, dot

ani2 = FuncAnimation(fig2, update2,
                     frames=len(t),
                     init_func=init2,
                     interval=1000 / fps,
                     blit=True)

plt.show()

