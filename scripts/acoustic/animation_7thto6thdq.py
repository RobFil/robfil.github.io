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
A7 = 0.6            # amplitude 7th harmonic

# -------------------------
# Angles
# -------------------------
theta1 =  w * t       # fundamental, forward
theta7 =  7 * w * t   # 7th harmonic, forward (positive sequence)

# -------------------------
# Space vectors in αβ
# -------------------------
x1 = A1 * np.cos(theta1)
y1 = A1 * np.sin(theta1)

x7 = A7 * np.cos(theta7)
y7 = A7 * np.sin(theta7)

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
line7, = ax1.plot([], [], linewidth=2, linestyle="--",
                  label="7. Harmonische (+7ω)")
ax1.legend(loc="upper right")

def init1():
    line1.set_data([], [])
    line7.set_data([], [])
    return line1, line7

def update1(i):
    line1.set_data([0, x1[i]], [0, y1[i]])
    line7.set_data([0, x7[i]], [0, y7[i]])
    return line1, line7

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

# Park transformation of the 7th harmonic into dq:
# [d]   [ cosθ   sinθ ] [ α ]
# [q] = [−sinθ   cosθ ] [ β ]
# with θ = theta1

xd7 =  x7 * np.cos(theta1) + y7 * np.sin(theta1)
yq7 = -x7 * np.sin(theta1) + y7 * np.cos(theta1)

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_aspect("equal")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True)
ax2.set_title("dq frame\n7. Harmonische erscheint als +6ω")

line_dq, = ax2.plot([], [], linewidth=2, color="red",
                    label="7. Harmonische im dq-System (+6ω)")
ax2.legend(loc="upper right")

def init2():
    line_dq.set_data([], [])
    return line_dq,

def update2(i):
    line_dq.set_data([0, xd7[i]], [0, yq7[i]])
    return line_dq,

ani2 = FuncAnimation(fig2, update2,
                     frames=len(t),
                     init_func=init2,
                     interval=1000 / fps,
                     blit=True)

plt.show()
