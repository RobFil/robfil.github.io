import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Physikalisch vereinfachtes Halte-Modell (stationaerer Zustand)
# ==========================================================
# Gegebene elektrische Parameter
R_rotor = 7.0                 # [Ohm]
L_rotor = 1e-3                # [H]
R_stator_d = 0.008            # [Ohm]
L_stator_d = 10e-6            # [H]

V_rotor_dc = 640.0            # [V] Rotor-Versorgung
V_stator_dc = 280.0           # [V] vereinfachte d-Achsen-Versorgung (aus AC-System abgeleitet)

# Bereits eingepraegte Sollstroeme, die nur gehalten werden
i_rotor_hold = 5.0            # [A]
i_d_hold = 50.0               # [A]

# Schaltfrequenzen (gekoppelt wie zuvor)
f_sw_stator = 10_000.0        # [Hz]
stator_to_rotor_factor = 4.0
f_sw_rotor = f_sw_stator / stator_to_rotor_factor

# Simulationsparameter
fs_sim = 2_000_000.0          # [Hz], hoch fuer Pulsdarstellung
T_total = 0.010               # [s]
dt = 1.0 / fs_sim
t = np.arange(0.0, T_total, dt)

# ==========================================================
# Duty-Cycle fuer Haltebetrieb (stationaer: <di/dt>=0 => V_avg = R*I)
# ==========================================================
V_rotor_req = R_rotor * i_rotor_hold
V_d_req = R_stator_d * i_d_hold

D_rotor = np.clip(V_rotor_req / V_rotor_dc, 0.0, 1.0)
D_stator = np.clip(V_d_req / V_stator_dc, 0.0, 1.0)

print("\n=== Haltebetrieb: benoetigte Duty-Cycles ===")
print(f"Rotor:  V_req = R*I = {V_rotor_req:.3f} V  ->  D_rotor  = {D_rotor*100:.3f} %")
print(f"d-Achse: V_req = R*I = {V_d_req:.3f} V  ->  D_stator = {D_stator*100:.3f} %")
print(f"f_sw,stator = {f_sw_stator:.1f} Hz, f_sw,rotor = {f_sw_rotor:.1f} Hz (Faktor {stator_to_rotor_factor:.1f})")

# ==========================================================
# PWM-Pulse (ideal, 0/1) und RL-Stromsimulation
# ==========================================================
T_sw_rotor = 1.0 / f_sw_rotor
T_sw_stator = 1.0 / f_sw_stator

phase_rotor = np.mod(t, T_sw_rotor) / T_sw_rotor
phase_stator = np.mod(t, T_sw_stator) / T_sw_stator

pwm_rotor = (phase_rotor < D_rotor).astype(float)
pwm_stator = (phase_stator < D_stator).astype(float)

v_rotor_applied = pwm_rotor * V_rotor_dc
v_d_applied = pwm_stator * V_stator_dc

# Start bereits im stationaeren Arbeitspunkt
i_rotor = np.zeros_like(t)
i_d = np.zeros_like(t)
i_rotor[0] = i_rotor_hold
i_d[0] = i_d_hold

for n in range(len(t) - 1):
    di_r = (v_rotor_applied[n] - R_rotor * i_rotor[n]) / L_rotor
    di_d = (v_d_applied[n] - R_stator_d * i_d[n]) / L_stator_d
    i_rotor[n + 1] = i_rotor[n] + dt * di_r
    i_d[n + 1] = i_d[n] + dt * di_d

# ==========================================================
# Ausschnitte im stationaeren Zustand (letzte 2 ms)
# ==========================================================
zoom_start = T_total - 0.002
m = t >= zoom_start

fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axs[0].step(t[m], pwm_rotor[m], where="post", color="tab:blue", label="PWM Rotor")
axs[0].set_title("Rotor-Pulse (stationaerer Ausschnitt)")
axs[0].set_ylabel("gate")
axs[0].set_ylim(-0.1, 1.1)
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].plot(t[m], i_rotor[m], color="tab:blue", label="i_rotor")
axs[1].axhline(i_rotor_hold, linestyle="--", color="gray", label="i_rotor,hold")
axs[1].set_title("Rotorstrom (stationaerer Ausschnitt)")
axs[1].set_ylabel("A")
axs[1].grid(True)
axs[1].legend(loc="upper right")

axs[2].step(t[m], pwm_stator[m], where="post", color="tab:orange", label="PWM d-Achse")
axs[2].set_title("Stator d-Achsen-Pulse (stationaerer Ausschnitt)")
axs[2].set_ylabel("gate")
axs[2].set_ylim(-0.1, 1.1)
axs[2].grid(True)
axs[2].legend(loc="upper right")

axs[3].plot(t[m], i_d[m], color="tab:red", label="i_d")
axs[3].axhline(i_d_hold, linestyle="--", color="gray", label="i_d,hold")
axs[3].set_title("d-Achsen-Strom (stationaerer Ausschnitt)")
axs[3].set_xlabel("Zeit [s]")
axs[3].set_ylabel("A")
axs[3].grid(True)
axs[3].legend(loc="upper right")

fig.suptitle(
    f"Haltebetrieb mit RL-Modell | D_rotor={D_rotor*100:.3f}% | D_stator={D_stator*100:.3f}%",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Kennwerte fuer den Ausschnitt
i_rotor_pp = np.max(i_rotor[m]) - np.min(i_rotor[m])
i_d_pp = np.max(i_d[m]) - np.min(i_d[m])

print("\n=== Stationaere Ripple-Kennwerte (Ausschnitt) ===")
print(f"Rotorstrom Ripple p-p: {i_rotor_pp:.6f} A")
print(f"d-Strom Ripple p-p:    {i_d_pp:.6f} A")
