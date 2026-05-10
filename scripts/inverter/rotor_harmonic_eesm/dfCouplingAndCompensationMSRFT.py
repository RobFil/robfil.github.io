import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Vereinfachtes MSRFT-Beispiel:
# Rotor-Ripple als Sinus, Kopplung auf d-Strom, Identifikation
# (Amplitude/Phase) und Kompensation ohne R/L/Pulsmodell.
# ==========================================================

# Grundparameter
f_sw_stator = 10_000.0  # [Hz]
stator_to_rotor_factor = 4  # f_stator / f_rotor
f_rotor = f_sw_stator / stator_to_rotor_factor  # [Hz]
omega_r = 2.0 * np.pi * f_rotor
Ts = 1.0 / f_sw_stator

# Signale (vereinfachte Naeherung)
i_r_dc = 5.0          # [A]
i_r_ripple_amp = 0.50  # [A]
phi_r = np.deg2rad(35.0)

i_d_ref = -50.0       # [A]
k_couple = 5.0        # [A_d / A_r]

# Kompensation
comp_gain = 1.0
id_noise_std = 0.0

# MSRFT-Fenster:
# Faktor ist bekannt, daher Samples pro Rotorperiode = stator_to_rotor_factor
samples_per_rotor = int(round(f_sw_stator / f_rotor))
msrft_rotor_periods = 4
N_win = msrft_rotor_periods * samples_per_rotor
sim_rotor_periods = 12
T_total = sim_rotor_periods / f_rotor
t = np.arange(0.0, T_total, Ts)
n = len(t)


def msrft_identify(x, t_sig, omega, n_win):
    """
    Einfache gleitende Einfrequenz-Identifikation (MSRFT-artig):
    x(t) ~= A*sin(omega*t + phi)
    Rueckgabe: A_hat(t), phi_hat(t), x_hat(t)
    """
    a_hat = np.zeros_like(x)
    phi_hat = np.zeros_like(x)
    x_hat = np.zeros_like(x)

    for k in range(len(x)):
        if k < n_win - 1:
            continue

        idx0 = k - n_win + 1
        tw = t_sig[idx0:k + 1]
        xw = x[idx0:k + 1]

        s = np.sin(omega * tw)
        c = np.cos(omega * tw)

        b_sin = (2.0 / n_win) * np.sum(xw * s)
        b_cos = (2.0 / n_win) * np.sum(xw * c)

        amp = np.sqrt(b_sin * b_sin + b_cos * b_cos)
        ph = np.arctan2(b_cos, b_sin)

        a_hat[k] = amp
        phi_hat[k] = ph
        x_hat[k] = amp * np.sin(omega * t_sig[k] + ph)

    return a_hat, phi_hat, x_hat


# ==========================================================
# 1) "Wahre" Signale
# ==========================================================
i_r_ripple = i_r_ripple_amp * np.sin(omega_r * t + phi_r)
i_r = i_r_dc + i_r_ripple

i_d_dist_true = k_couple * i_r_ripple
i_d_meas = i_d_ref + i_d_dist_true + id_noise_std * np.random.randn(n)


# ==========================================================
# 2) MSRFT-Identifikation der d-Stoerung aus i_d - i_d_ref
# ==========================================================
x_id = i_d_meas - i_d_ref
amp_hat, phi_hat, i_d_dist_est = msrft_identify(x_id, t, omega_r, N_win)

# Kompensation
i_d_comp = -comp_gain * i_d_dist_est
i_d_after_comp = i_d_meas + i_d_comp


# ==========================================================
# 3) Auswertung
# ==========================================================
steady_eval_periods = 4
steady_mask = t >= (T_total - steady_eval_periods / f_rotor)

ripple_before = np.max(i_d_meas[steady_mask]) - np.min(i_d_meas[steady_mask])
ripple_after = np.max(i_d_after_comp[steady_mask]) - np.min(i_d_after_comp[steady_mask])
ripple_red = 100.0 * (ripple_before - ripple_after) / max(ripple_before, 1e-12)

amp_true = k_couple * i_r_ripple_amp
amp_hat_mean = np.mean(amp_hat[steady_mask])

phi_hat_mean = np.angle(np.mean(np.exp(1j * phi_hat[steady_mask])))
phi_err_deg = np.rad2deg(np.angle(np.exp(1j * (phi_hat_mean - phi_r))))

print("\n=== MSRFT: d-Ripple-Identifikation und Kompensation ===")
print(f"f_sw_stator = {f_sw_stator:.1f} Hz")
print(f"f_rotor     = {f_rotor:.1f} Hz (Faktor {stator_to_rotor_factor})")
print(f"N_win       = {N_win} Samples ({msrft_rotor_periods} Rotorperioden)")
print(f"A_true(d)   = {amp_true:.6f} A")
print(f"A_hat(d)    = {amp_hat_mean:.6f} A")
print(f"phi_err     = {phi_err_deg:.3f} deg")
print(f"Ripple vor  = {ripple_before:.6f} A p-p")
print(f"Ripple nach = {ripple_after:.6f} A p-p")
print(f"Reduktion   = {ripple_red:.2f} %")


# ==========================================================
# 4) Plot
# ==========================================================
fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

axs[0].plot(t, i_r, label="i_r")
axs[0].plot(t, i_d_dist_true, label="i_d_dist true")
axs[0].set_title("Rotorstrom und gekoppelte d-Stoerung (vereinfacht)")
axs[0].set_ylabel("A")
axs[0].grid(True)
axs[0].legend(loc="upper right")

axs[1].plot(t, amp_true * np.ones_like(t), "--", label="A_true")
axs[1].plot(t, amp_hat, label="A_hat (MSRFT)")
axs[1].set_title("MSRFT-Amplitudenschaetzung")
axs[1].set_ylabel("A")
axs[1].grid(True)
axs[1].legend(loc="upper right")

axs[2].plot(t, i_d_meas, label="i_d vor Komp.")
axs[2].plot(t, i_d_after_comp, label="i_d nach Komp.")
axs[2].axhline(i_d_ref, linestyle="--", color="gray", label="i_d_ref")
axs[2].set_title("d-Strom: vor / nach Kompensation")
axs[2].set_ylabel("A")
axs[2].grid(True)
axs[2].legend(loc="upper right")

axs[3].plot(t, i_d_meas - np.mean(i_d_meas[steady_mask]), label="Ripple vor")
axs[3].plot(t, i_d_after_comp - np.mean(i_d_after_comp[steady_mask]), label="Ripple nach")
axs[3].set_title("Ripplevergleich (mittelwertfrei)")
axs[3].set_xlabel("Zeit [s]")
axs[3].set_ylabel("A")
axs[3].grid(True)
axs[3].legend(loc="upper right")

fig.suptitle("MSRFT-basierte Kompensation (vereinfachtes Sinusmodell)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
