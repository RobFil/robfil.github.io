import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Physikalisch vereinfachtes Halte-Modell
# Rotor: 3-Level aus interleaved HS/LS-Logik (wie beschrieben)
# Stator: 3-Phasen-Vollbruecke mit center-aligned SVPWM
# ==========================================================

# Elektrische Parameter
R_rotor = 7.0                  # [Ohm]
L_rotor = 8e-3                 # [H]
R_stator = 0.08                # [Ohm] (x10, Rotor bleibt unveraendert)
L_stator_d = 10e-5             # [H]
L_stator_q = 12e-5             # [H]

V_rotor_dc = 640.0             # [V]
V_stator_dc = 280.0            # [V]

# Sollwerte
i_rotor_hold = 5.0             # [A]
i_d_ref = -50.0                # [A]
i_q_ref = 0.0                  # [A]
k_couple = -20.0               # [A_d / A_r]

# Frequenzen
f_sw_stator = 10_000.0         # [Hz]
stator_to_rotor_factor = 4.0
f_sw_rotor = f_sw_stator / stator_to_rotor_factor
f_elec = 0.0                   # [Hz] elektrische Grundfrequenz fuer dq (0 = stationaere Achse)
omega_e = 2.0 * np.pi * f_elec

# Simulation
fs_sim = 2_000_000.0           # [Hz]
T_total = 0.050                # [s], laenger fuer klaren eingeschwungenen Zustand
dt = 1.0 / fs_sim
t = np.arange(0.0, T_total, dt)

# ==========================================================
# Sollspannungen fuer Haltebetrieb
# ==========================================================
V_rotor_req = R_rotor * i_rotor_hold

v_d_ff = R_stator * i_d_ref - omega_e * L_stator_q * i_q_ref
v_q_ff = R_stator * i_q_ref + omega_e * L_stator_d * i_d_ref

m_rotor = np.clip(V_rotor_req / V_rotor_dc, -1.0, 1.0)

print("\n=== Sollwerte / Haltebetrieb ===")
print(f"Rotor: V_req={V_rotor_req:.3f} V -> m_rotor={m_rotor*100:.3f}% von Udc")
print(f"Stator dq-Feedforward: v_d,ff={v_d_ff:.3f} V, v_q,ff={v_q_ff:.3f} V")
print(f"Stator-Sollstroeme: i_d*={i_d_ref:.2f} A, i_q*={i_q_ref:.2f} A")
print(f"f_sw,stator={f_sw_stator:.1f} Hz, f_sw,rotor={f_sw_rotor:.1f} Hz, f_elec={f_elec:.1f} Hz")

# ==========================================================
# PWM-Basen
# ==========================================================
T_sw_rotor = 1.0 / f_sw_rotor
T_sw_stator = 1.0 / f_sw_stator
phase_rotor = np.mod(t, T_sw_rotor) / T_sw_rotor
phase_stator = np.mod(t, T_sw_stator) / T_sw_stator
carrier_center = 1.0 - np.abs(2.0 * phase_stator - 1.0)  # 0..1..0

# ==========================================================
# Rotor: Schaltlogik gemaess Vorgabe
# 11 -> +Udc, 00 -> -Udc, 10/01 -> 0
# ==========================================================
pwm_rotor_hs_left = np.zeros_like(t)
pwm_rotor_ls_right = np.zeros_like(t)
a = abs(m_rotor)
for n in range(len(t)):
    p = phase_rotor[n]
    in_overlap = (0.5 - 0.5 * a) <= p < (0.5 + 0.5 * a)
    if m_rotor >= 0.0:
        if in_overlap:
            hs, ls = 1.0, 1.0
        elif p < (0.5 - 0.5 * a):
            hs, ls = 1.0, 0.0
        else:
            hs, ls = 0.0, 1.0
    else:
        if in_overlap:
            hs, ls = 0.0, 0.0
        elif p < (0.5 - 0.5 * a):
            hs, ls = 1.0, 0.0
        else:
            hs, ls = 0.0, 1.0
    pwm_rotor_hs_left[n] = hs
    pwm_rotor_ls_right[n] = ls

v_rotor_applied = np.zeros_like(t)
both_high = (pwm_rotor_hs_left > 0.5) & (pwm_rotor_ls_right > 0.5)
both_low = (pwm_rotor_hs_left < 0.5) & (pwm_rotor_ls_right < 0.5)
v_rotor_applied[both_high] = V_rotor_dc
v_rotor_applied[both_low] = -V_rotor_dc

# ==========================================================
# Stator: 3-Phasen-Vollbruecke mit center-aligned SVPWM
# ==========================================================
theta_e = omega_e * t

# Platzhalter-Arrays fuer Statorsignale (werden im Zeitschritt aufgebaut)
d_a = np.zeros_like(t)
d_b = np.zeros_like(t)
d_c = np.zeros_like(t)
s_a = np.zeros_like(t)
s_b = np.zeros_like(t)
s_c = np.zeros_like(t)
v_a = np.zeros_like(t)
v_b = np.zeros_like(t)
v_c = np.zeros_like(t)
v_d_applied = np.zeros_like(t)
v_q_applied = np.zeros_like(t)
v_d_ref_hist = np.zeros_like(t)
v_q_ref_hist = np.zeros_like(t)

# ==========================================================
# Stromsimulation
# ==========================================================
i_rotor = np.zeros_like(t)
i_d = np.zeros_like(t)
i_q = np.zeros_like(t)
i_rotor[0] = i_rotor_hold
i_d[0] = i_d_ref
i_q[0] = i_q_ref

for n in range(len(t) - 1):
    di_r = (v_rotor_applied[n] - R_rotor * i_rotor[n]) / L_rotor

    # Reine dq-Feedforward-Spannungen (ohne Kopplung, ohne Regler)
    v_d_ref_n = v_d_ff
    v_q_ref_n = v_q_ff

    # dq -> alpha/beta
    ce = np.cos(theta_e[n])
    se = np.sin(theta_e[n])
    v_alpha_ref_n = v_d_ref_n * ce - v_q_ref_n * se
    v_beta_ref_n = v_d_ref_n * se + v_q_ref_n * ce

    # alpha/beta -> abc (Referenz)
    v_a_ref_n = v_alpha_ref_n
    v_b_ref_n = -0.5 * v_alpha_ref_n + (np.sqrt(3.0) / 2.0) * v_beta_ref_n
    v_c_ref_n = -0.5 * v_alpha_ref_n - (np.sqrt(3.0) / 2.0) * v_beta_ref_n

    # SVPWM Offset-Injection
    v_max_n = max(v_a_ref_n, v_b_ref_n, v_c_ref_n)
    v_min_n = min(v_a_ref_n, v_b_ref_n, v_c_ref_n)
    v_offset_n = -0.5 * (v_max_n + v_min_n)

    v_a_mod_n = v_a_ref_n + v_offset_n
    v_b_mod_n = v_b_ref_n + v_offset_n
    v_c_mod_n = v_c_ref_n + v_offset_n

    # Duty + center-aligned comparator (Pulse mittig in der Statorperiode)
    d_a_n = np.clip(0.5 + v_a_mod_n / V_stator_dc, 0.0, 1.0)
    d_b_n = np.clip(0.5 + v_b_mod_n / V_stator_dc, 0.0, 1.0)
    d_c_n = np.clip(0.5 + v_c_mod_n / V_stator_dc, 0.0, 1.0)
    s_a_n = 1.0 if carrier_center[n] >= (1.0 - d_a_n) else 0.0
    s_b_n = 1.0 if carrier_center[n] >= (1.0 - d_b_n) else 0.0
    s_c_n = 1.0 if carrier_center[n] >= (1.0 - d_c_n) else 0.0

    # Pole -> Phase
    v_a_pole_n = (2.0 * s_a_n - 1.0) * (V_stator_dc / 2.0)
    v_b_pole_n = (2.0 * s_b_n - 1.0) * (V_stator_dc / 2.0)
    v_c_pole_n = (2.0 * s_c_n - 1.0) * (V_stator_dc / 2.0)
    v_n_n = (v_a_pole_n + v_b_pole_n + v_c_pole_n) / 3.0
    v_a_n = v_a_pole_n - v_n_n
    v_b_n = v_b_pole_n - v_n_n
    v_c_n = v_c_pole_n - v_n_n

    # abc -> alpha/beta -> dq (applied)
    v_alpha_n = (2.0 / 3.0) * (v_a_n - 0.5 * v_b_n - 0.5 * v_c_n)
    v_beta_n = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (v_b_n - v_c_n))
    v_d_n = ce * v_alpha_n + se * v_beta_n
    v_q_n = -se * v_alpha_n + ce * v_beta_n

    # dq-Stromdynamik mit den tatsaechlich gepulsten/applizierten Spannungen.
    # Rotor->Stator-Kopplung wird weiterhin erst fuer den finalen Mess-Plot addiert.
    di_d = (v_d_n - R_stator * i_d[n] + omega_e * L_stator_q * i_q[n]) / L_stator_d
    di_q = (v_q_n - R_stator * i_q[n] - omega_e * L_stator_d * i_d[n]) / L_stator_q

    i_rotor[n + 1] = i_rotor[n] + dt * di_r
    i_d[n + 1] = i_d[n] + dt * di_d
    i_q[n + 1] = i_q[n] + dt * di_q

    d_a[n] = d_a_n
    d_b[n] = d_b_n
    d_c[n] = d_c_n
    s_a[n] = s_a_n
    s_b[n] = s_b_n
    s_c[n] = s_c_n
    v_a[n] = v_a_n
    v_b[n] = v_b_n
    v_c[n] = v_c_n
    v_d_applied[n] = v_d_ref_n
    v_q_applied[n] = v_q_ref_n
    v_d_ref_hist[n] = v_d_ref_n
    v_q_ref_hist[n] = v_q_ref_n

# Letzten Wert fortschreiben fuer Plotkonsistenz
d_a[-1], d_b[-1], d_c[-1] = d_a[-2], d_b[-2], d_c[-2]
s_a[-1], s_b[-1], s_c[-1] = s_a[-2], s_b[-2], s_c[-2]
v_a[-1], v_b[-1], v_c[-1] = v_a[-2], v_b[-2], v_c[-2]
v_d_applied[-1], v_q_applied[-1] = v_d_applied[-2], v_q_applied[-2]
v_d_ref_hist[-1], v_q_ref_hist[-1] = v_d_ref_hist[-2], v_q_ref_hist[-2]

# Rotor->Stator-Kopplung auf d-Strom
i_d_coupling = k_couple * (i_rotor - i_rotor_hold)
i_d_meas = i_d + i_d_coupling

# ==========================================================
# Eingeschwungener Ausschnitt
# ==========================================================
num_rotor_periods_in_zoom = 6.25  # halbe Anzahl gegenüber bisher (~12.5)
zoom_window = num_rotor_periods_in_zoom / f_sw_rotor
zoom_start = T_total - zoom_window
m = t >= zoom_start

# Offsetfehler im eingeschwungenen Bereich bestimmen und fuer Plot kompensieren
i_d_offset_err = np.mean(i_d[m]) - i_d_ref
i_d_meas_offset_err = np.mean(i_d_meas[m]) - i_d_ref
i_d_plot = i_d - i_d_offset_err
i_d_meas_plot = i_d_meas - i_d_meas_offset_err

# Pruefung, ob zusaetzliche d-Achsen-Offsetspannung noch erforderlich ist
i_q_offset_err = np.mean(i_q[m]) - i_q_ref
v_d_offset_needed = R_stator * i_d_offset_err - omega_e * L_stator_q * i_q_offset_err
id_err_tol = 0.5  # [A]
needs_vd_offset = abs(i_d_offset_err) > id_err_tol

fig, axs = plt.subplots(8, 1, figsize=(14, 17), sharex=True)

axs[0].step(t[m], pwm_rotor_hs_left[m], where="post", color="tab:blue", label="Rotor HS links")
axs[0].step(t[m], pwm_rotor_ls_right[m], where="post", color="tab:cyan", label="Rotor LS rechts")
axs[0].set_title("Rotor-Schalterzustaende interleaved")
axs[0].set_ylabel("gate")
axs[0].set_ylim(-0.1, 1.1)
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].step(t[m], v_rotor_applied[m], where="post", color="tab:purple", label="v_rotor")
axs[1].set_title("Rotorspannung (-Udc / 0 / +Udc)")
axs[1].set_ylabel("V")
axs[1].grid(True)
axs[1].legend(loc="upper right")

axs[2].plot(t[m], i_rotor[m], color="tab:blue", label="i_rotor")
axs[2].axhline(i_rotor_hold, linestyle="--", color="gray", label="i_rotor,hold")
axs[2].set_title("Rotorstrom")
axs[2].set_ylabel("A")
axs[2].grid(True)
axs[2].legend(loc="upper right")

axs[3].step(t[m], s_a[m], where="post", color="tab:blue", label="S_a")
axs[3].step(t[m], s_b[m], where="post", color="tab:orange", label="S_b")
axs[3].step(t[m], s_c[m], where="post", color="tab:green", label="S_c")
axs[3].set_title("Stator-Schalter (Vollbruecke, center-aligned SVPWM)")
axs[3].set_ylabel("gate")
axs[3].set_ylim(-0.1, 1.1)
axs[3].grid(True)
axs[3].legend(loc="upper right")

axs[4].plot(t[m], v_a[m], color="tab:blue", label="v_a")
axs[4].plot(t[m], v_b[m], color="tab:orange", label="v_b")
axs[4].plot(t[m], v_c[m], color="tab:green", label="v_c")
axs[4].set_title("3-Phasenspannungen aus SVPWM")
axs[4].set_ylabel("V")
axs[4].grid(True)
axs[4].legend(loc="upper right")

axs[5].plot(t[m], v_d_applied[m], color="tab:brown", label="v_d")
axs[5].plot(t[m], v_q_applied[m], color="tab:olive", label="v_q")
axs[5].plot(t[m], v_d_ref_hist[m], linestyle="--", color="tab:brown", label="v_d*")
axs[5].plot(t[m], v_q_ref_hist[m], linestyle="--", color="tab:olive", label="v_q*")
axs[5].set_title("dq-Spannungen (Feedforward, ohne Regler)")
axs[5].set_ylabel("V")
axs[5].grid(True)
axs[5].legend(loc="upper right")

axs[6].plot(t[m], i_d_plot[m], color="tab:red", label="i_d (offsetkorrigiert)")
axs[6].plot(t[m], i_q[m], color="tab:purple", label="i_q")
axs[6].axhline(i_d_ref, linestyle="--", color="gray", label="i_d*")
axs[6].axhline(i_q_ref, linestyle="--", color="black", label="i_q*")
axs[6].set_title("dq-Stroeme ohne Rotor-Kopplung (offsetkorrigiert)")
axs[6].set_ylabel("A")
axs[6].grid(True)
axs[6].legend(loc="upper right")

axs[7].plot(t[m], i_d_meas_plot[m], color="tab:red", label="i_d,meas (offsetkorrigiert)")
axs[7].plot(t[m], i_d_coupling[m], color="tab:green", label="i_d,coupling")
axs[7].axhline(i_d_ref, linestyle="--", color="gray", label="i_d*")
axs[7].set_title("d-Strom mit Rotor-Kopplung (offsetkorrigiert)")
axs[7].set_xlabel("Zeit [s]")
axs[7].set_ylabel("A")
axs[7].grid(True)
axs[7].legend(loc="upper right")

fig.suptitle(
    f"Rotor + Stator Vollbruecke | SVPWM center-aligned | i_d*={i_d_ref:.1f} A, i_q*={i_q_ref:.1f} A",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Kennwerte
print("\n=== Stationaere Kennwerte (Ausschnitt) ===")
print(f"Offsetfehler i_d:      {i_d_offset_err:.6f} A")
print(f"Offsetfehler i_d_meas: {i_d_meas_offset_err:.6f} A")
print(f"Abgeleitete v_d-Offsetspannung: {v_d_offset_needed:.6f} V")
print(f"Zus. v_d-Offset notwendig (> {id_err_tol:.2f} A): {'JA' if needs_vd_offset else 'NEIN'}")
print(f"mean(v_rotor) = {np.mean(v_rotor_applied[m]):.6f} V")
print(f"mean(i_rotor) = {np.mean(i_rotor[m]):.6f} A")
print(f"mean(i_d)     = {np.mean(i_d[m]):.6f} A")
print(f"mean(i_q)     = {np.mean(i_q[m]):.6f} A")
print(f"mean(i_d_meas)= {np.mean(i_d_meas[m]):.6f} A")
print(f"mean(d_a,b,c) = {np.mean(d_a[m]):.4f}, {np.mean(d_b[m]):.4f}, {np.mean(d_c[m]):.4f}")
