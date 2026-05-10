import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Diskrete Statorpuls-Kompensation des rotorinduzierten d-Ripples
# Kopplungsmodell: NUR Rotorstromaenderung koppelt in d
# (periodische Delta-i_r pro Statorperiode -> periodische Delta-i_d)
# ==========================================================

# Parameter
R_rotor = 7.0
L_rotor = 8e-3
R_stator = 0.08
L_stator_d = 10e-5
L_stator_q = 12e-5

V_rotor_dc = 640.0
V_stator_ac = 280.0

i_rotor_hold = 5.0
i_d_ref = -50.0
i_q_ref = 0.0
k_couple_delta_i = 5.0  # Delta-i_d = k * Delta-i_r (nur Aenderung koppelt)
comp_gain = 1.0         # Skalierung der periodischen Vorsteuerung

f_sw_stator = 10_000.0
# Parametrierbarer Faktor: wie viele Statorperioden pro Rotorperiode
stator_periods_per_rotor_period = 8.0
f_sw_rotor = f_sw_stator / stator_periods_per_rotor_period
n_mech_rpm = 0.0
pole_pairs = 3
f_mech_hz = n_mech_rpm / 60.0
f_elec = pole_pairs * f_mech_hz
omega_e = 2.0 * np.pi * f_elec

fs_sim = 2_000_000.0
T_total = 0.05
dt = 1.0 / fs_sim
t = np.arange(0.0, T_total, dt)

T_sw_rotor = 1.0 / f_sw_rotor
T_sw_stator = 1.0 / f_sw_stator
phase_rotor = np.mod(t, T_sw_rotor) / T_sw_rotor
phase_stator = np.mod(t, T_sw_stator) / T_sw_stator
carrier_center = 1.0 - np.abs(2.0 * phase_stator - 1.0)  # 0..1..0
samples_per_stator_pwm = int(round(fs_sim / f_sw_stator))
samples_per_rotor_pwm = int(round(fs_sim / f_sw_rotor))
stator_slots_per_rotor = max(1, int(round(stator_periods_per_rotor_period)))


# ==========================================================
# Rotor: Vorgabe-Topologie
# 11 -> +Udc, 00 -> -Udc, 10/01 -> 0
# ==========================================================
V_rotor_req = R_rotor * i_rotor_hold
m_rotor_set = np.clip(V_rotor_req / V_rotor_dc, -1.0, 1.0)  # bleibt konstant


def circular_dist(x, c):
    d = abs(x - c)
    return min(d, 1.0 - d)


pwm_rotor_hs_left = np.zeros_like(t)
pwm_rotor_ls_right = np.zeros_like(t)
m_rotor_cmd = m_rotor_set
a = abs(m_rotor_cmd)

for n in range(len(t)):
    # Rotor-Duty nur zu Beginn einer Rotorperiode aktualisieren (hier konstant)
    if n % samples_per_rotor_pwm == 0:
        m_rotor_cmd = m_rotor_set
        a = abs(m_rotor_cmd)

    p = phase_rotor[n]

    # Grundzustand: Schalter invertiert -> v_rotor = 0
    if p < 0.5:
        hs, ls = 1.0, 0.0
    else:
        hs, ls = 0.0, 1.0

    # Zwei Ueberlappungsfenster pro Rotorperiode (bei 0 und 0.5)
    # -> doppelte Pulsfrequenz in v_rotor gg.ueber Einzelschalterfrequenz
    win_width = a / 2.0
    in_overlap = (circular_dist(p, 0.0) <= 0.5 * win_width) or (circular_dist(p, 0.5) <= 0.5 * win_width)

    if in_overlap:
        if m_rotor_cmd >= 0.0:
            hs, ls = 1.0, 1.0   # +Udc
        else:
            hs, ls = 0.0, 0.0   # -Udc

    pwm_rotor_hs_left[n] = hs
    pwm_rotor_ls_right[n] = ls

v_rotor_applied = np.zeros_like(t)
both_high = (pwm_rotor_hs_left > 0.5) & (pwm_rotor_ls_right > 0.5)
both_low = (pwm_rotor_hs_left < 0.5) & (pwm_rotor_ls_right < 0.5)
v_rotor_applied[both_high] = V_rotor_dc
v_rotor_applied[both_low] = -V_rotor_dc

i_rotor = np.zeros_like(t)
i_rotor[0] = i_rotor_hold
for n in range(len(t) - 1):
    di_r = (v_rotor_applied[n] - R_rotor * i_rotor[n]) / L_rotor
    i_rotor[n + 1] = i_rotor[n] + dt * di_r

# ==========================================================
# Statorperiodische Rotorauswertung und slot-basierte SchÃ¤tzung
# ==========================================================
sw_idx = np.arange(0, len(t), samples_per_stator_pwm, dtype=int)
if sw_idx[-1] != len(t) - 1:
    sw_idx = np.append(sw_idx, len(t) - 1)

i_rotor_sw = i_rotor[sw_idx]
delta_i_r_sw = np.diff(i_rotor_sw)  # Delta i_r je Statorperiode [A]
di_r_avg_sw = delta_i_r_sw / T_sw_stator  # [A/s]
di_r_avg_t = np.zeros_like(t)
for k in range(len(di_r_avg_sw)):
    di_r_avg_t[sw_idx[k]:sw_idx[k + 1]] = di_r_avg_sw[k]
di_r_avg_t[sw_idx[-1]:] = di_r_avg_sw[-1]
# Skalar je Statorperiode: Integral der Stromaenderung
# int(di_r/dt dt) ueber Ts = Delta i_r
delta_i_r_t = np.zeros_like(t)
for k in range(len(delta_i_r_sw)):
    delta_i_r_t[sw_idx[k]:sw_idx[k + 1]] = delta_i_r_sw[k]
delta_i_r_t[sw_idx[-1]:] = delta_i_r_sw[-1]

slot_sw = np.mod(np.arange(len(sw_idx) - 1), stator_slots_per_rotor)

# Rotor-Mittelspannung je Statorperiode aus realer PWM (nutzt Duty/Frequenz/Topologie)
v_rotor_avg_sw = np.zeros(len(sw_idx) - 1)
for k in range(len(sw_idx) - 1):
    i0 = sw_idx[k]
    i1 = sw_idx[k + 1]
    v_rotor_avg_sw[k] = np.mean(v_rotor_applied[i0:i1])

# Profil je Statorslot innerhalb der Rotorperiode im stationaeren Fenster
steady_sw = t[sw_idx[:-1]] >= (T_total - 0.02)
v_rotor_slot_profile = np.zeros(stator_slots_per_rotor)
for s in range(stator_slots_per_rotor):
    m_slot = steady_sw & (slot_sw == s)
    if np.any(m_slot):
        v_rotor_slot_profile[s] = np.mean(v_rotor_avg_sw[m_slot])
    else:
        v_rotor_slot_profile[s] = 0.0

# Repetitives Rotor-dI/dt-Profil aus R/L-Modell pro Statorslot:
# di_r/dt = (v_avg(slot) - R*i_r(slot_start))/L
di_r_slot_est = np.zeros(stator_slots_per_rotor)
i_r_slot_est = np.zeros(stator_slots_per_rotor + 1)
i_r_slot_est[0] = i_rotor_hold
for s in range(stator_slots_per_rotor):
    di_s = (v_rotor_slot_profile[s] - R_rotor * i_r_slot_est[s]) / L_rotor
    di_r_slot_est[s] = di_s
    i_r_slot_est[s + 1] = i_r_slot_est[s] + T_sw_stator * di_s

# Periodische Konsistenz: kein Drift ueber eine Rotorperiode
drift_per_rotor = i_r_slot_est[-1] - i_r_slot_est[0]
di_bias = drift_per_rotor / (stator_slots_per_rotor * T_sw_stator)
di_r_slot_est -= di_bias

# Reale d-Stoerung aus Rotor-Delta-i (nur Aenderung koppelt)
# Delta i_d = k * Delta i_r -> di_d,dist = Delta i_d / Ts
di_d_dist_sw = k_couple_delta_i * di_r_avg_sw
di_d_dist = np.zeros_like(t)
for k in range(len(di_d_dist_sw)):
    di_d_dist[sw_idx[k]:sw_idx[k + 1]] = di_d_dist_sw[k]
di_d_dist[sw_idx[-1]:] = di_d_dist_sw[-1]

# Repetitives Schaetzprofil fuer Kompensation (nur AC-Anteil)
di_r_slot_est_ac = di_r_slot_est - np.mean(di_r_slot_est)
di_d_dist_est_slot = k_couple_delta_i * di_r_slot_est_ac
di_r_slot_est_t = np.zeros_like(t)
for k in range(len(sw_idx) - 1):
    s = k % stator_slots_per_rotor
    di_r_slot_est_t[sw_idx[k]:sw_idx[k + 1]] = di_r_slot_est[s]
di_r_slot_est_t[sw_idx[-1]:] = di_r_slot_est[(len(sw_idx) - 1) % stator_slots_per_rotor]

# Anschaulicher gekoppelter d-Stromanteil (A), kontinuierlich aus di_d,dist integriert
# (vermeidet unphysikalische Stufenspruenge aus direkter Delta-Aufsummierung)
i_d_cpl = np.cumsum(di_d_dist * dt)
steady_time_mask = t >= (T_total - 0.02)
i_d_cpl -= np.mean(i_d_cpl[steady_time_mask]) if np.any(steady_time_mask) else np.mean(i_d_cpl)


# ==========================================================
# Statorsimulation: reiner d/q-Strom (ohne Kopplungsaddition im Modell)
# ==========================================================
def simulate_stator_pure():
    i_d = np.zeros_like(t)
    i_q = np.zeros_like(t)
    i_d[0] = i_d_ref
    i_q[0] = i_q_ref

    s_a = np.zeros_like(t)
    s_b = np.zeros_like(t)
    s_c = np.zeros_like(t)
    v_a = np.zeros_like(t)
    v_b = np.zeros_like(t)
    v_c = np.zeros_like(t)
    v_d_applied = np.zeros_like(t)
    v_q_applied = np.zeros_like(t)
    v_d_cmd = np.zeros_like(t)
    v_q_cmd = np.zeros_like(t)

    v_d_base = R_stator * i_d_ref - omega_e * L_stator_q * i_q_ref
    v_q_base = R_stator * i_q_ref + omega_e * L_stator_d * i_d_ref
    v_d_ref_hold = v_d_base
    v_q_ref_hold = v_q_base

    for n in range(len(t) - 1):
        # Diskrete Aktualisierung nur zu Beginn jeder Statorperiode (ohne Komp-Eingriff)
        if n % samples_per_stator_pwm == 0:
            v_d_ref_hold = v_d_base
            v_q_ref_hold = v_q_base

        v_d_ref_n = v_d_ref_hold
        v_q_ref_n = v_q_ref_hold
        v_d_cmd[n] = v_d_ref_n
        v_q_cmd[n] = v_q_ref_n

        # dq -> alpha/beta -> abc (winkelabhaengig)
        theta_e = omega_e * t[n]
        c = np.cos(theta_e)
        s = np.sin(theta_e)
        v_alpha_ref_n = c * v_d_ref_n - s * v_q_ref_n
        v_beta_ref_n = s * v_d_ref_n + c * v_q_ref_n
        v_a_ref_n = v_alpha_ref_n
        v_b_ref_n = -0.5 * v_alpha_ref_n + (np.sqrt(3.0) / 2.0) * v_beta_ref_n
        v_c_ref_n = -0.5 * v_alpha_ref_n - (np.sqrt(3.0) / 2.0) * v_beta_ref_n

        # SVPWM offset injection
        v_max_n = max(v_a_ref_n, v_b_ref_n, v_c_ref_n)
        v_min_n = min(v_a_ref_n, v_b_ref_n, v_c_ref_n)
        v_offset_n = -0.5 * (v_max_n + v_min_n)

        v_a_mod_n = v_a_ref_n + v_offset_n
        v_b_mod_n = v_b_ref_n + v_offset_n
        v_c_mod_n = v_c_ref_n + v_offset_n

        d_a_n = np.clip(0.5 + v_a_mod_n / V_stator_ac, 0.0, 1.0)
        d_b_n = np.clip(0.5 + v_b_mod_n / V_stator_ac, 0.0, 1.0)
        d_c_n = np.clip(0.5 + v_c_mod_n / V_stator_ac, 0.0, 1.0)

        # Center-aligned Pulse in der Periodenmitte
        s_a_n = 1.0 if carrier_center[n] >= (1.0 - d_a_n) else 0.0
        s_b_n = 1.0 if carrier_center[n] >= (1.0 - d_b_n) else 0.0
        s_c_n = 1.0 if carrier_center[n] >= (1.0 - d_c_n) else 0.0

        v_a_pole_n = (2.0 * s_a_n - 1.0) * (V_stator_ac / 2.0)
        v_b_pole_n = (2.0 * s_b_n - 1.0) * (V_stator_ac / 2.0)
        v_c_pole_n = (2.0 * s_c_n - 1.0) * (V_stator_ac / 2.0)
        v_n_n = (v_a_pole_n + v_b_pole_n + v_c_pole_n) / 3.0
        v_a_n = v_a_pole_n - v_n_n
        v_b_n = v_b_pole_n - v_n_n
        v_c_n = v_c_pole_n - v_n_n

        v_alpha_n = (2.0 / 3.0) * (v_a_n - 0.5 * v_b_n - 0.5 * v_c_n)
        v_beta_n = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (v_b_n - v_c_n))
        v_d_n = c * v_alpha_n + s * v_beta_n
        v_q_n = -s * v_alpha_n + c * v_beta_n

        # Reines d/q-Modell ohne Kopplungsaddition
        di_d = (v_d_n - R_stator * i_d[n] + omega_e * L_stator_q * i_q[n]) / L_stator_d
        di_q = (v_q_n - R_stator * i_q[n] - omega_e * L_stator_d * i_d[n]) / L_stator_q
        i_d[n + 1] = i_d[n] + dt * di_d
        i_q[n + 1] = i_q[n] + dt * di_q

        s_a[n], s_b[n], s_c[n] = s_a_n, s_b_n, s_c_n
        v_a[n], v_b[n], v_c[n] = v_a_n, v_b_n, v_c_n
        v_d_applied[n], v_q_applied[n] = v_d_n, v_q_n

    # letzter Wert
    s_a[-1], s_b[-1], s_c[-1] = s_a[-2], s_b[-2], s_c[-2]
    v_a[-1], v_b[-1], v_c[-1] = v_a[-2], v_b[-2], v_c[-2]
    v_d_applied[-1], v_q_applied[-1] = v_d_applied[-2], v_q_applied[-2]
    v_d_cmd[-1] = v_d_cmd[-2]
    v_q_cmd[-1] = v_q_cmd[-2]

    return {
        "i_d": i_d,
        "i_q": i_q,
        "v_d_cmd": v_d_cmd,
        "v_d_applied": v_d_applied,
        "v_q_cmd": v_q_cmd,
        "v_a": v_a,
        "v_b": v_b,
        "v_c": v_c,
        "s_a": s_a,
        "s_b": s_b,
        "s_c": s_c,
    }


pure = simulate_stator_pure()
theta_e_t = omega_e * t
c_t = np.cos(theta_e_t)
s_t = np.sin(theta_e_t)
v_alpha_from_abc = (2.0 / 3.0) * (pure["v_a"] - 0.5 * pure["v_b"] - 0.5 * pure["v_c"])
v_beta_from_abc = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (pure["v_b"] - pure["v_c"]))
v_d_from_abc = c_t * v_alpha_from_abc + s_t * v_beta_from_abc

# ==========================================================
# Kopplung/Kompensation nur fuer Darstellung addieren
# ==========================================================
di_d_dist_est_t = np.zeros_like(t)
for k in range(len(sw_idx) - 1):
    s = k % stator_slots_per_rotor
    di_d_dist_est_t[sw_idx[k]:sw_idx[k + 1]] = comp_gain * di_d_dist_est_slot[s]
di_d_dist_est_t[sw_idx[-1]:] = di_d_dist_est_t[sw_idx[-2]]
v_d_comp_t = -L_stator_d * di_d_dist_est_t
v_d_cmd_with = pure["v_d_cmd"] + v_d_comp_t

# Kompensationsanteil als additiver Strombeitrag (nur fuer Anzeige)
di_d_comp_t = v_d_comp_t / L_stator_d
i_d_comp_add = np.cumsum(di_d_comp_t * dt)
i_d_comp_add -= np.mean(i_d_comp_add[t >= (T_total - 0.02)])

# Reinen d-Strom fuer Anzeige um Sollwert zentrieren
i_d_pure = pure["i_d"].copy()
steady_mask = t >= (T_total - 0.02)
id_mean_steady = np.mean(i_d_pure[steady_mask])
i_d_pure += (i_d_ref - id_mean_steady)

# Messdarstellungen (nur additive Ueberlagerung)
i_d_meas_cpl = i_d_pure + i_d_cpl
i_d_meas_cpl_comp = i_d_pure + i_d_cpl + i_d_comp_add

# Eingeschwungener Ausschnitt
num_rotor_periods_in_zoom = 6.25
zoom_window = num_rotor_periods_in_zoom / f_sw_rotor
zoom_end = T_total - (1.0 / f_sw_rotor)  # letzte Rotorperiode ausblenden
zoom_start = zoom_end - zoom_window
m = (t >= zoom_start) & (t < zoom_end)

# Ripple-Vergleich
ripple_no = np.max(i_d_meas_cpl[m]) - np.min(i_d_meas_cpl[m])
ripple_yes = np.max(i_d_meas_cpl_comp[m]) - np.min(i_d_meas_cpl_comp[m])
red = 100.0 * (ripple_no - ripple_yes) / max(ripple_no, 1e-12)

print("\n=== Diskrete Ripple-Kompensation im Stator ===")
print(
    f"f_sw,stator={f_sw_stator:.1f} Hz, f_sw,rotor={f_sw_rotor:.1f} Hz, "
    f"Faktor(stator/rotor)={stator_periods_per_rotor_period:.2f}"
)
print(f"Statorslots pro Rotorperiode: {stator_slots_per_rotor}")
print(f"v_rotor_slot_profile [V]: {np.array2string(v_rotor_slot_profile, precision=3)}")
print(f"di_r_slot_est [A/s]:      {np.array2string(di_r_slot_est, precision=3)}")
print(f"di_r_slot_est_ac [A/s]:   {np.array2string(di_r_slot_est_ac, precision=3)}")
print(f"di_r_avg(real) min/max:   {np.min(di_r_avg_t[m]):.3f} / {np.max(di_r_avg_t[m]):.3f} A/s")
print(f"Delta i_r je Ts min/max:  {np.min(delta_i_r_sw):.6e} / {np.max(delta_i_r_sw):.6e} A")
print(f"i_d(pure) Mittelwert:   {np.mean(i_d_pure[m]):.6f} A")
print(f"i_d Ripple +Kopplung:   {ripple_no:.6f} A p-p")
print(f"i_d Ripple +Kopplung+K: {ripple_yes:.6f} A p-p")
print(f"Ripple-Reduktion:      {red:.2f} %")
print(
    f"di_d,dist_est p-p: "
    f"{(np.max(di_d_dist_est_t[m]) - np.min(di_d_dist_est_t[m])):.6f} A/s"
)

# Slotweise Mittelwerte im eingeschwungenen Zustand (pro Statorperiode innerhalb Rotorperiode)
delta_i_slot_mean = np.zeros(stator_slots_per_rotor)
di_r_slot_real_mean = np.zeros(stator_slots_per_rotor)
delta_i_slot_from_int_mean = np.zeros(stator_slots_per_rotor)
for s in range(stator_slots_per_rotor):
    ms = steady_sw & (slot_sw == s)
    if np.any(ms):
        delta_i_slot_mean[s] = np.mean(delta_i_r_sw[ms])
        di_r_slot_real_mean[s] = np.mean(di_r_avg_sw[ms])
        delta_i_slot_from_int_mean[s] = np.mean(delta_i_r_sw[ms])

print("Slot-Mittelwerte (steady): slot | Delta_i_r[A] | di_r/dt[A/s] | Delta_i_r_aus_di_dt[A]")
for s in range(stator_slots_per_rotor):
    print(
        f"  {s:2d} | {delta_i_slot_mean[s]:+.6e} | "
        f"{di_r_slot_real_mean[s]:+.6e} | {delta_i_slot_from_int_mean[s]:+.6e}"
    )

fig, axs = plt.subplots(7, 1, figsize=(14, 16), sharex=True)

axs[0].step(t[m], pwm_rotor_hs_left[m], where="post", color="tab:blue", label="Rotor HS")
axs[0].step(t[m], pwm_rotor_ls_right[m], where="post", color="tab:cyan", label="Rotor LS")
axs[0].set_title("Rotor-Schalterzustaende")
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
ax2b = axs[2].twinx()
ax2b.step(t[m], delta_i_r_t[m], where="post", color="tab:orange", label="Delta i_r je Ts")
axs[2].set_title("Rotorstrom und Delta i_r je Statorperiode")
axs[2].set_ylabel("A")
ax2b.set_ylabel("A")
axs[2].grid(True)
ln1, lb1 = axs[2].get_legend_handles_labels()
ln2, lb2 = ax2b.get_legend_handles_labels()
axs[2].legend(ln1 + ln2, lb1 + lb2, loc="upper right")

axs[3].plot(t[m], pure["v_d_cmd"][m], color="tab:red", label="v_d_cmd ohne")
axs[3].plot(t[m], v_d_cmd_with[m], color="tab:purple", label="v_d_cmd mit")
axs[3].plot(t[m], v_d_comp_t[m], color="tab:brown", linestyle="--", label="v_d_comp")
axs[3].plot(t[m], v_d_from_abc[m], color="tab:green", linestyle=":", label="v_d aus u_a/u_b/u_c")
axs[3].set_title("Diskrete d-Achsen-Spannungsvorgabe")
axs[3].set_ylabel("V")
axs[3].grid(True)
axs[3].legend(loc="upper right")

axs[4].plot(t[m], i_d_meas_cpl[m], color="tab:red", label="i_d + Kopplung")
axs[4].plot(t[m], i_d_meas_cpl_comp[m], color="tab:purple", label="i_d + Kopplung + Komp.")
axs[4].axhline(i_d_ref, linestyle="--", color="gray", label="i_d*")
axs[4].set_title("d-Strom: additive Darstellung")
axs[4].set_ylabel("A")
axs[4].grid(True)
axs[4].legend(loc="upper right")

axs[5].plot(t[m], i_d_pure[m], color="tab:blue", label="i_d rein (ohne Kopplung)")
axs[5].axhline(i_d_ref, linestyle="--", color="gray", label="i_d*")
axs[5].set_title("Reiner d-Strom")
axs[5].set_ylabel("A")
axs[5].grid(True)
axs[5].legend(loc="upper right")

axs[6].plot(t[m], i_d_meas_cpl[m] - np.mean(i_d_meas_cpl[m]), color="tab:red", label="Ripple +Kopplung")
axs[6].plot(t[m], i_d_meas_cpl_comp[m] - np.mean(i_d_meas_cpl_comp[m]), color="tab:purple", label="Ripple +Kopplung+K")
axs[6].set_title("Ripplevergleich (mittelwertfrei)")
axs[6].set_xlabel("Zeit [s]")
axs[6].set_ylabel("A")
axs[6].grid(True)
axs[6].legend(loc="upper right")

fig.suptitle("Diskrete Statorpuls-Kompensation: Kopplung nur ueber Rotorstromaenderung", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

