import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Parameter
# ==========================================================
fs = 20000          # Abtastfrequenz [Hz]
T  = 0.05           # Simulationsdauer [s]
t  = np.arange(0, T, 1/fs)

f1 = 100            # Grundfrequenz [Hz]
w  = 2*np.pi*f1

# Harmonische
I1 = 10
I5 = 1    # 4 %
I7 = 0.4    # 3 %

theta1 = 0
theta5 = np.pi/6
theta7 = -np.pi/8

# ==========================================================
# 1. Dreiphasenströme nach Gl.(2)
# ==========================================================
iu = (I1*np.cos(w*t + theta1)
      + I5*np.cos(5*w*t + theta5)
      + I7*np.cos(7*w*t + theta7))

iv = (I1*np.cos(w*t - 2*np.pi/3 + theta1)
      + I5*np.cos(5*w*t - 2*5*np.pi/3 + theta5)
      + I7*np.cos(7*w*t - 2*7*np.pi/3 + theta7))

iw = (I1*np.cos(w*t + 2*np.pi/3 + theta1)
      + I5*np.cos(5*w*t + 2*5*np.pi/3 + theta5)
      + I7*np.cos(7*w*t + 2*7*np.pi/3 + theta7))

# ==========================================================
# 2. Clarke Transformation (abc → αβ)
# ==========================================================
ialpha = (2/3)*(iu - 0.5*iv - 0.5*iw)
ibeta  = (2/3)*(np.sqrt(3)/2*(iv - iw))

# ==========================================================
# 3. Park Transformation (αβ → dq, Fundamental-SRF)
# ==========================================================
theta = w*t
id_f =  ialpha*np.cos(theta) + ibeta*np.sin(theta)
iq_f = -ialpha*np.sin(theta) + ibeta*np.cos(theta)

# ==========================================================
# 4. MSRFT für 5. Ordnung (dq → dq5)
#   Rotationsgeschwindigkeit = -6ω
# ==========================================================
theta5s = -6*w*t
id5 =  id_f*np.cos(theta5s) + iq_f*np.sin(theta5s)
iq5 = -id_f*np.sin(theta5s) + iq_f*np.cos(theta5s)

# ==========================================================
# 5. MSRFT für 7. Ordnung (dq → dq7)
#   Rotationsgeschwindigkeit = +6ω
# ==========================================================
theta7s = 6*w*t
id7 =  id_f*np.cos(theta7s) + iq_f*np.sin(theta7s)
iq7 = -id_f*np.sin(theta7s) + iq_f*np.cos(theta7s)

# ==========================================================
# DC-Anteile berechnen
# ==========================================================
id5_dc = np.mean(id5)
iq5_dc = np.mean(iq5)

id7_dc = np.mean(id7)
iq7_dc = np.mean(iq7)

# ==========================================================
# Amplitude und Phase der 5. und 7. Harmonischen
# ==========================================================
I5_est = np.sqrt(id5_dc**2 + iq5_dc**2)
I7_est = np.sqrt(id7_dc**2 + iq7_dc**2)

# numerische Schwelle, ab wann wir "Amplitude = 0" annehmen
eps = 1e-6

# Phase zuerst in rad %mean entspricht DC Anteil, entspricht jeweiliger Harmonischen
phi5_est = np.arctan2(iq5_dc, id5_dc)
phi7_est = np.arctan2(iq7_dc, id7_dc)

# Wenn Amplitude ~ 0 → Phase = 0 setzen
if I5_est < eps:
    phi5_est = 0.0
if I7_est < eps:
    phi7_est = 0.0

# In Grad umrechnen
phi5_deg = np.degrees(phi5_est)
phi7_deg = np.degrees(phi7_est)



# ==========================================================
# Kombinierter Plot
# ==========================================================
plt.figure(figsize=(16,16))

# -------- abc --------
plt.subplot(3,2,1)
plt.plot(t, iu, label="i_u")
plt.plot(t, iv, label="i_v")
plt.plot(t, iw, label="i_w")
plt.title("Dreiphasenströme (abc)")
plt.legend()
plt.grid()

# -------- dq fundamental --------
plt.subplot(3,2,2)
plt.plot(t, id_f, label="i_d")
plt.plot(t, iq_f, label="i_q")
plt.title("Fundamentales dq-System")
plt.legend()
plt.grid()

# -------- dq5 (5th harmonic channel) --------
plt.subplot(3,2,3)
plt.plot(t, id5, label="i_d5(t)")
plt.plot(t, iq5, label="i_q5(t)")
plt.axhline(id5_dc, color="blue", linestyle="--",
            label=f"DC i_d5 = {id5_dc:.3f}")
plt.axhline(iq5_dc, color="orange", linestyle="--",
            label=f"DC i_q5 = {iq5_dc:.3f}")
plt.title("dq5-System (−6ω) → 5. Harmonische wird DC")
plt.legend()
plt.grid()

# -------- dq7 (7th harmonic channel) --------
plt.subplot(3,2,4)
plt.plot(t, id7, label="i_d7(t)")
plt.plot(t, iq7, label="i_q7(t)")
plt.axhline(id7_dc, color="blue", linestyle="--",
            label=f"DC i_d7 = {id7_dc:.3f}")
plt.axhline(iq7_dc, color="orange", linestyle="--",
            label=f"DC i_q7 = {iq7_dc:.3f}")
plt.title("dq7-System (+6ω) → 7. Harmonische (hier ≈ 0)")
plt.legend()
plt.grid()

# -------- 5th Harmonic: Amplitude & Phase (getrennte Achsen) --------
ax51 = plt.subplot(3,2,5)
ax52 = ax51.twinx()

# Amplitude (linke Achse)
ax51.bar([0], [I5_est], width=0.4, color="tab:blue", label="Amplitude I5")
ax51.set_ylabel("Amplitude")
ax51.set_xticks([0])
ax51.set_xticklabels(["5th"])
ax51.tick_params(axis='y', labelcolor="tab:blue")

# Phase (rechte Achse)
ax52.bar([0.5], [phi5_deg], width=0.4, color="tab:orange", label="Phase φ5 [deg]")
ax52.set_ylabel("Phase [deg]")
ax52.tick_params(axis='y', labelcolor="tab:orange")

ax51.set_title("5. Harmonische – Amplitude & Phase (dq5)")
ax51.grid(True, axis="y")

# kleine Legende bauen
lines_5 = [
    plt.Line2D([0],[0], color="tab:blue", lw=6, label="Amplitude I5"),
    plt.Line2D([0],[0], color="tab:orange", lw=6, label="Phase φ5 [deg]")
]
ax51.legend(handles=lines_5, loc="upper right")


# -------- 7th Harmonic: Amplitude & Phase (getrennte Achsen) --------
ax71 = plt.subplot(3,2,6)
ax72 = ax71.twinx()

# Amplitude (linke Achse)
ax71.bar([0], [I7_est], width=0.4, color="tab:blue", label="Amplitude I7")
ax71.set_ylabel("Amplitude")
ax71.set_xticks([0])
ax71.set_xticklabels(["7th"])
ax71.tick_params(axis='y', labelcolor="tab:blue")

# Phase (rechte Achse)
ax72.bar([0.5], [phi7_deg], width=0.4, color="tab:orange", label="Phase φ7 [deg]")
ax72.set_ylabel("Phase [deg]")
ax72.tick_params(axis='y', labelcolor="tab:orange")

ax71.set_title("7. Harmonische – Amplitude & Phase (dq7)")
ax71.grid(True, axis="y")

lines_7 = [
    plt.Line2D([0],[0], color="tab:blue", lw=6, label="Amplitude I7"),
    plt.Line2D([0],[0], color="tab:orange", lw=6, label="Phase φ7 [deg]")
]
ax71.legend(handles=lines_7, loc="upper right")


plt.tight_layout()
plt.show()


# ==========================================================
# Ideale Kompensation der 5. und 7. Harmonischen (ohne Regler)
# ==========================================================

# 1) DC-Anteile von oben

# 2) Gegenvektoren im jeweiligen MSRFT-System
# 5th
id5_comp = -id5_dc * np.ones_like(t)
iq5_comp = -iq5_dc * np.ones_like(t)

# 7th
id7_comp = -id7_dc * np.ones_like(t)
iq7_comp = -iq7_dc * np.ones_like(t)

# ==========================================================
# 3) Rücktransformation dq5 -> dq   (Frame rotiert mit −6ω)
#    inverse Rotation also mit +6ω
#    Drehung des Vektors
# ==========================================================
theta5s = -6*w*t
id5_to_dq =  id5_comp*np.cos(theta5s) - iq5_comp*np.sin(theta5s)
iq5_to_dq =  id5_comp*np.sin(theta5s) + iq5_comp*np.cos(theta5s)

# ==========================================================
# 4) Rücktransformation dq7 -> dq   (Frame rotiert mit +6ω)
#    inverse Rotation also mit −6ω
#    Drehung des Vektors
# ==========================================================
theta7s =  6*w*t
id7_to_dq =  id7_comp*np.cos(theta7s) - iq7_comp*np.sin(theta7s)
iq7_to_dq =  id7_comp*np.sin(theta7s) + iq7_comp*np.cos(theta7s)

# ==========================================================
# 5) Summieren der beiden Kompensationsanteile im dq-System
# ==========================================================
id_comp_dq = id5_to_dq + id7_to_dq
iq_comp_dq = iq5_to_dq + iq7_to_dq

# ==========================================================
# 6) Rücktransformation dq -> αβ
# ==========================================================
theta = w*t
ialpha_comp =  id_comp_dq*np.cos(theta) - iq_comp_dq*np.sin(theta)
ibeta_comp  =  id_comp_dq*np.sin(theta) + iq_comp_dq*np.cos(theta)

# ==========================================================
# 7) Rücktransformation αβ -> abc
# ==========================================================
iu_comp = ialpha_comp
iv_comp = -0.5*ialpha_comp + np.sqrt(3)/2 * ibeta_comp
iw_comp = -0.5*ialpha_comp - np.sqrt(3)/2 * ibeta_comp

# ==========================================================
# 8) Kompensierte Ströme
# ==========================================================
iu_c = iu + iu_comp
iv_c = iv + iv_comp
iw_c = iw + iw_comp



# ==========================================================
# Figure 2: Wirkung der Kompensation
# ==========================================================
plt.figure(figsize=(16,12))

# --- Subplot 1: Original abc ---
plt.subplot(4,1,1)
plt.plot(t, iu, label="i_u")
plt.plot(t, iv, label="i_v")
plt.plot(t, iw, label="i_w")
plt.title("Originale Dreiphasenströme (abc)")
plt.legend()
plt.grid()

# --- Subplot 2: Original dq ---
plt.subplot(4,1,2)
plt.plot(t, id_f, label="i_d")
plt.plot(t, iq_f, label="i_q")
plt.title("Originale Ströme im dq-System")
plt.legend()
plt.grid()

# --- Subplot 3: Kompensierte abc ---
plt.subplot(4,1,3)
plt.plot(t, iu_c, label="i_u komp.")
plt.plot(t, iv_c, label="i_v komp.")
plt.plot(t, iw_c, label="i_w komp.")
plt.title("Kompensierte Dreiphasenströme (abc)")
plt.legend()
plt.grid()

# --- Subplot 4: Kompensierte dq ---
# zuerst wieder Clarke + Park auf die kompensierten Ströme
ialpha_c = (2/3)*(iu_c - 0.5*iv_c - 0.5*iw_c)
ibeta_c  = (2/3)*(np.sqrt(3)/2*(iv_c - iw_c))

id_c =  ialpha_c*np.cos(theta) + ibeta_c*np.sin(theta)
iq_c = -ialpha_c*np.sin(theta) + ibeta_c*np.cos(theta)

plt.subplot(4,1,4)
plt.plot(t, id_c, label="i_d komp.")
plt.plot(t, iq_c, label="i_q komp.")
plt.title("Kompensierte Ströme im dq-System")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
