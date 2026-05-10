import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.ticker import FuncFormatter, MaxNLocator


def generate_signal(t, frequency, form, value_min, value_max):
    if form == "Sinus":
        return (value_max - value_min) / 2 * np.sin(2 * np.pi * frequency * t) + (value_max + value_min) / 2
    if form == "Dreieck":
        return (value_max - value_min) * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1) / 2 + (value_max + value_min) / 2
    if form == "Rechteck":
        phase = np.mod(t * frequency, 1.0)
        return np.where(phase < 0.5, value_min, value_max)
    raise ValueError(f"Unbekannte Signalform: {form}")


def generate_triangle_carrier(t, frequency):
    phase = t * frequency - np.floor(t * frequency)
    return 1.0 - 4.0 * np.abs(phase - 0.5)


def dq_to_abc(d_value, q_value, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    alpha = cos_theta * d_value - sin_theta * q_value
    beta = sin_theta * d_value + cos_theta * q_value

    phase_a = alpha
    phase_b = -0.5 * alpha + np.sqrt(3) / 2 * beta
    phase_c = -0.5 * alpha - np.sqrt(3) / 2 * beta
    return phase_a, phase_b, phase_c


def abc_to_dq(phase_a, phase_b, phase_c, theta):
    alpha = (2.0 / 3.0) * (phase_a - 0.5 * phase_b - 0.5 * phase_c)
    beta = (2.0 / 3.0) * ((np.sqrt(3) / 2.0) * (phase_b - phase_c))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    d_value = cos_theta * alpha + sin_theta * beta
    q_value = -sin_theta * alpha + cos_theta * beta
    return d_value, q_value


def switching_to_phase_voltages(state_a, state_b, state_c, vdc):
    phase_a = (2.0 * state_a - state_b - state_c) * vdc / 3.0
    phase_b = (2.0 * state_b - state_a - state_c) * vdc / 3.0
    phase_c = (2.0 * state_c - state_a - state_b) * vdc / 3.0
    return phase_a, phase_b, phase_c


def pwm(modulation, carrier):
    return int(modulation >= carrier)


def build_pwm_update_lookup(t, f_pwm):
    pwm_period_index = np.floor(t * f_pwm + 1e-12).astype(int)
    pwm_update_mask = np.ones_like(t, dtype=bool)
    pwm_update_mask[1:] = pwm_period_index[1:] != pwm_period_index[:-1]
    return pwm_update_mask


def sample_and_hold_signal(signal, update_mask):
    held_signal = np.array(signal, copy=True)
    current_value = held_signal[0]
    for idx in range(len(held_signal)):
        if update_mask[idx]:
            current_value = signal[idx]
        held_signal[idx] = current_value
    return held_signal


def advance_current_reference(current_state, target_current, resistance, inductance, voltage_limit, sample_time):
    if inductance <= 0.0:
        return target_current, float(np.clip(resistance * target_current, -voltage_limit, voltage_limit))

    if resistance <= 1e-12:
        required_voltage = inductance * (target_current - current_state) / sample_time
        command_voltage = float(np.clip(required_voltage, -voltage_limit, voltage_limit))
        next_current = current_state + command_voltage * sample_time / inductance
        return next_current, command_voltage

    alpha = np.exp(-(resistance / inductance) * sample_time)
    one_minus_alpha = max(1.0 - alpha, 1e-12)
    required_voltage = resistance * (target_current - current_state * alpha) / one_minus_alpha
    command_voltage = float(np.clip(required_voltage, -voltage_limit, voltage_limit))
    next_current = current_state * alpha + (command_voltage / resistance) * one_minus_alpha
    return next_current, command_voltage


def moving_rms(signal, window_size):
    if window_size <= 1:
        return np.abs(signal)

    kernel = np.ones(window_size) / window_size
    mean_square = np.convolve(signal ** 2, kernel, mode="same")
    return np.sqrt(mean_square)


def block_rms_trace(signal, block_size):
    if block_size <= 1:
        return np.abs(signal)

    rms_trace = np.zeros_like(signal)
    last_rms = 0.0
    for start in range(0, len(signal), block_size):
        stop = min(start + block_size, len(signal))
        if stop - start < block_size and start > 0:
            rms_trace[start:stop] = last_rms
            continue

        block = signal[start:stop]
        last_rms = np.sqrt(np.mean(block ** 2))
        rms_trace[start:stop] = last_rms
    return rms_trace


def rms_to_instantaneous_factor(form):
    if form == "Sinus":
        return np.sqrt(2.0)
    if form == "Dreieck":
        return np.sqrt(3.0)
    if form == "Rechteck":
        return 1.0
    raise ValueError(f"Unbekannte Signalform: {form}")


def simulate_machine(params):
    f_sig = params["f_sig"]
    f_pwm = params["f_pwm"]
    form = params["form"]
    periods = max(int(params.get("periods", 4)), 1)
    rms_margin_periods = max(int(params.get("rms_margin_periods", 1)), 0)
    rotor_angle_deg = params.get("theta_deg", params.get("theta", 0.0))
    theta = np.deg2rad(rotor_angle_deg)

    inductance_d = params["Ld"]
    inductance_q = params["Lq"]
    psi_pm = params["psi"]
    resistance_s = params["Rs"]
    vdc_nominal = params["Udc"]
    capacitance_dc = params["Cdc_uF"] * 1e-6
    source_resistance = max(params.get("Rsrc_mOhm", 20.0) * 1e-3, 1e-6)
    source_inductance = max(params.get("Lsrc_uH", 0.0) * 1e-6, 0.0)

    if capacitance_dc <= 0:
        raise ValueError("Cdc_uF must be greater than zero.")
    if f_pwm <= 0:
        raise ValueError("f_pwm must be greater than zero.")

    visible_time = periods / f_sig if f_sig > 0 else 0.05
    if f_sig > 0:
        pre_roll_time = rms_margin_periods / f_sig
        post_roll_time = rms_margin_periods / f_sig
    else:
        pre_roll_time = rms_margin_periods / f_pwm
        post_roll_time = rms_margin_periods / f_pwm

    dt = 1.0 / (f_pwm * 40.0)
    t = np.arange(-pre_roll_time, visible_time + post_roll_time, dt)
    carrier = generate_triangle_carrier(t, f_pwm)
    pwm_update_mask = build_pwm_update_lookup(t, f_pwm)

    id_rms_min = params.get("id_rms_min", params.get("i_min", 0.0))
    id_rms_max = params.get("id_rms_max", params.get("i_max", 0.0))
    current_scale = rms_to_instantaneous_factor(form)
    initial_id = params.get("id_init", 0.0) * current_scale
    if f_sig > 0 and pre_roll_time > 0.0:
        signal_time = t
        id_ref_cont = generate_signal(signal_time, f_sig, form, id_rms_min * current_scale, id_rms_max * current_scale)
    else:
        signal_time = np.clip(t, 0.0, visible_time)
        id_ref_cont = generate_signal(signal_time, f_sig, form, id_rms_min * current_scale, id_rms_max * current_scale)
        id_ref_cont[t < 0.0] = initial_id
    id_ref = sample_and_hold_signal(id_ref_cont, pwm_update_mask)
    iq_ref = np.zeros_like(id_ref)
    psi_d_ref = inductance_d * id_ref + psi_pm
    psi_q_ref = inductance_q * iq_ref

    id_arr = np.zeros_like(t)
    iq_arr = np.zeros_like(t)
    psi_d_arr = np.zeros_like(t)
    psi_q_arr = np.zeros_like(t)
    psi_d_arr[0] = inductance_d * initial_id + psi_pm
    psi_q_arr[0] = 0.0
    id_arr[0] = (psi_d_arr[0] - psi_pm) / inductance_d
    iq_arr[0] = psi_q_arr[0] / inductance_q

    ud_cmd = np.zeros_like(t)
    uq_cmd = np.zeros_like(t)
    ud_act = np.zeros_like(t)
    uq_act = np.zeros_like(t)
    id_ref_ff = np.zeros_like(t)
    iq_ref_ff = np.zeros_like(t)

    state_a = np.zeros_like(t, dtype=int)
    state_b = np.zeros_like(t, dtype=int)
    state_c = np.zeros_like(t, dtype=int)
    duty_a = np.zeros_like(t)
    duty_b = np.zeros_like(t)
    duty_c = np.zeros_like(t)

    ia = np.zeros_like(t)
    ib = np.zeros_like(t)
    ic = np.zeros_like(t)
    idc = np.zeros_like(t)
    i_source = np.zeros_like(t)
    i_cap = np.zeros_like(t)
    vdc = np.zeros_like(t)
    vdc[0] = vdc_nominal

    modulation_a_hold = 0.0
    modulation_b_hold = 0.0
    modulation_c_hold = 0.0
    duty_a_hold = 0.5
    duty_b_hold = 0.5
    duty_c_hold = 0.5
    ud_cmd_hold = 0.0
    uq_cmd_hold = 0.0
    id_ref_ff_state = initial_id
    iq_ref_ff_state = 0.0

    id_ref_ff[0] = id_ref_ff_state
    iq_ref_ff[0] = iq_ref_ff_state

    for k in range(len(t) - 1):
        id_arr[k] = (psi_d_arr[k] - psi_pm) / inductance_d
        iq_arr[k] = psi_q_arr[k] / inductance_q

        voltage_limit = max(vdc[k] / 2.0, 1e-6)

        if pwm_update_mask[k]:
            target_dt = max(1.0 / f_pwm, dt)
            id_ref_ff_state, ud_cmd_hold = advance_current_reference(
                id_ref_ff_state,
                id_ref[k],
                resistance_s,
                inductance_d,
                voltage_limit,
                target_dt,
            )
            iq_ref_ff_state, uq_cmd_hold = advance_current_reference(
                iq_ref_ff_state,
                iq_ref[k],
                resistance_s,
                inductance_q,
                voltage_limit,
                target_dt,
            )

            ua_ref, ub_ref, uc_ref = dq_to_abc(ud_cmd_hold, uq_cmd_hold, theta)
            modulation_a_hold = np.clip(ua_ref / voltage_limit, -1.0, 1.0)
            modulation_b_hold = np.clip(ub_ref / voltage_limit, -1.0, 1.0)
            modulation_c_hold = np.clip(uc_ref / voltage_limit, -1.0, 1.0)
            duty_a_hold = 0.5 * (modulation_a_hold + 1.0)
            duty_b_hold = 0.5 * (modulation_b_hold + 1.0)
            duty_c_hold = 0.5 * (modulation_c_hold + 1.0)

        ud_cmd[k] = ud_cmd_hold
        uq_cmd[k] = uq_cmd_hold
        id_ref_ff[k] = id_ref_ff_state
        iq_ref_ff[k] = iq_ref_ff_state
        duty_a[k] = duty_a_hold
        duty_b[k] = duty_b_hold
        duty_c[k] = duty_c_hold

        state_a[k] = pwm(modulation_a_hold, carrier[k])
        state_b[k] = pwm(modulation_b_hold, carrier[k])
        state_c[k] = pwm(modulation_c_hold, carrier[k])

        ia[k], ib[k], ic[k] = dq_to_abc(id_arr[k], iq_arr[k], theta)
        idc[k] = state_a[k] * ia[k] + state_b[k] * ib[k] + state_c[k] * ic[k]

        if source_inductance > 0.0:
            source_voltage = vdc_nominal - source_resistance * i_source[k] - vdc[k]
            di_source = source_voltage / source_inductance
            i_source[k + 1] = i_source[k] + di_source * dt
        else:
            i_source[k] = (vdc_nominal - vdc[k]) / source_resistance

        i_cap[k] = i_source[k] - idc[k]

        vdc[k + 1] = max(vdc[k] + (i_cap[k] / capacitance_dc) * dt, 1.0)

        if source_inductance == 0.0:
            i_source[k + 1] = (vdc_nominal - vdc[k + 1]) / source_resistance

        phase_a, phase_b, phase_c = switching_to_phase_voltages(duty_a[k], duty_b[k], duty_c[k], vdc[k])
        ud_act[k], uq_act[k] = abc_to_dq(phase_a, phase_b, phase_c, theta)

        psi_d_arr[k + 1] = psi_d_arr[k] + (ud_act[k] - resistance_s * id_arr[k]) * dt
        psi_q_arr[k + 1] = psi_q_arr[k] + (uq_act[k] - resistance_s * iq_arr[k]) * dt
        id_arr[k + 1] = (psi_d_arr[k + 1] - psi_pm) / inductance_d
        iq_arr[k + 1] = psi_q_arr[k + 1] / inductance_q

    if len(t) > 1:
        ud_cmd[-1] = ud_cmd[-2]
        uq_cmd[-1] = uq_cmd[-2]
        id_ref_ff[-1] = id_ref_ff[-2]
        iq_ref_ff[-1] = iq_ref_ff[-2]
        state_a[-1] = state_a[-2]
        state_b[-1] = state_b[-2]
        state_c[-1] = state_c[-2]
        duty_a[-1] = duty_a[-2]
        duty_b[-1] = duty_b[-2]
        duty_c[-1] = duty_c[-2]
        i_source[-1] = i_source[-2]

    ia[-1], ib[-1], ic[-1] = dq_to_abc(id_arr[-1], iq_arr[-1], theta)
    idc[-1] = state_a[-1] * ia[-1] + state_b[-1] * ib[-1] + state_c[-1] * ic[-1]
    i_cap[-1] = i_source[-1] - idc[-1]
    phase_a, phase_b, phase_c = switching_to_phase_voltages(duty_a[-1], duty_b[-1], duty_c[-1], vdc[-1])
    ud_act[-1], uq_act[-1] = abc_to_dq(phase_a, phase_b, phase_c, theta)
    vdc_ripple = vdc - np.mean(vdc)

    return {
        "t": t,
        "visible_time": visible_time,
        "visible_start_time": 0.0,
        "id_arr": id_arr,
        "iq_arr": iq_arr,
        "id_ref": id_ref,
        "id_ref_ff": id_ref_ff,
        "iq_ref": iq_ref,
        "iq_ref_ff": iq_ref_ff,
        "psi_d_ref": psi_d_ref,
        "psi_q_ref": psi_q_ref,
        "psi_d_arr": psi_d_arr,
        "psi_q_arr": psi_q_arr,
        "ia": ia,
        "ib": ib,
        "ic": ic,
        "state_a": state_a,
        "state_b": state_b,
        "state_c": state_c,
        "idc": idc,
        "i_source": i_source,
        "i_cap": i_cap,
        "vdc": vdc,
        "vdc_ripple": vdc_ripple,
        "ud_cmd": ud_cmd,
        "uq_cmd": uq_cmd,
        "ud_act": ud_act,
        "uq_act": uq_act,
    }


def run_sim(params):
    sim = simulate_machine(params)
    rotor_angle_deg = params.get("theta_deg", params.get("theta", 0.0))

    t = sim["t"]
    visible_time = sim["visible_time"]
    visible_start_time = sim["visible_start_time"]
    id_arr = sim["id_arr"]
    id_ref = sim["id_ref"]
    ia = sim["ia"]
    ib = sim["ib"]
    ic = sim["ic"]
    state_a = sim["state_a"]
    state_b = sim["state_b"]
    state_c = sim["state_c"]
    idc = sim["idc"]
    i_source = sim["i_source"]
    i_cap = sim["i_cap"]
    vdc = sim["vdc"]
    vdc_ripple = sim["vdc_ripple"]
    ud_cmd = sim["ud_cmd"]
    ud_act = sim["ud_act"]

    idc_rms = np.sqrt(np.mean(idc ** 2))
    i_cap_rms = np.sqrt(np.mean(i_cap ** 2))
    vdc_ripple_pp = np.max(vdc) - np.min(vdc)

    if len(t) > 1:
        visible_end_time = visible_start_time + visible_time - 0.5 * (t[1] - t[0])
        visible_mask = (t >= visible_start_time) & (t < visible_end_time)
    else:
        visible_mask = np.ones_like(t, dtype=bool)
    t_plot = t[visible_mask]
    id_arr_plot = id_arr[visible_mask]
    id_ref_plot = id_ref[visible_mask]
    ia_plot = ia[visible_mask]
    ib_plot = ib[visible_mask]
    ic_plot = ic[visible_mask]
    state_a_plot = state_a[visible_mask]
    state_b_plot = state_b[visible_mask]
    state_c_plot = state_c[visible_mask]
    idc_plot = idc[visible_mask]
    i_source_plot = i_source[visible_mask]
    ud_cmd_plot = ud_cmd[visible_mask]
    ud_act_plot = ud_act[visible_mask]
    vdc_plot = vdc[visible_mask]

    if len(t_plot) > 1 and params["f_sig"] > 0:
        samples_per_period = max(int(round((1.0 / params["f_sig"]) / (t_plot[1] - t_plot[0]))), 1)
        idc_rms_trace_plot = block_rms_trace(idc_plot, samples_per_period)
    elif len(t_plot) > 1:
        samples_per_pwm = max(int(round((1.0 / params["f_pwm"]) / (t_plot[1] - t_plot[0]))), 1)
        idc_rms_trace_plot = block_rms_trace(idc_plot, samples_per_pwm)
    else:
        idc_rms_trace_plot = np.abs(idc_plot)

    x_axis = t_plot - visible_start_time
    x_label = "Zeit [s]"

    idc_rms = np.sqrt(np.mean(idc_plot ** 2))
    i_cap_rms = np.sqrt(np.mean(i_cap[visible_mask] ** 2))
    vdc_ripple_pp = np.max(vdc[visible_mask]) - np.min(vdc[visible_mask])
    ud_scale = np.sqrt(2.0)
    u_d_limit_plot = vdc_plot / (2.0 * ud_scale)

    print(f"DC RMS Strom: {idc_rms:.2f} A")
    print(f"Kondensator RMS Strom: {i_cap_rms:.2f} A")
    print(f"Zwischenkreis Ripple pp: {vdc_ripple_pp:.2f} V")

    fig, axs = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"Anzeige: {visible_time:.4f} s | Simuliert: {t[-1] - t[0]:.4f} s | Rotorwinkel Stillstand: {rotor_angle_deg:.1f} deg"
    )

    axs[0].plot(x_axis, id_arr_plot, label="i_d")
    if params["form"] == "Rechteck":
        axs[0].step(x_axis, id_ref_plot, where="post", linestyle="--", label="i_d Soll")
    else:
        axs[0].plot(x_axis, id_ref_plot, "--", label="i_d Soll")
    axs[0].legend(loc="upper right")
    axs[0].set_title("i_d Verlauf")

    axs[1].step(x_axis, ud_cmd_plot / ud_scale, where="post", color="C3", linewidth=1.0, label="u_d Soll / sqrt(2)")
    axs[1].step(x_axis, ud_act_plot / ud_scale, where="post", color="C4", linewidth=1.0, linestyle="--", label="u_d Ist / sqrt(2)")
    axs[1].plot(x_axis, u_d_limit_plot, color="0.5", linewidth=1.0, linestyle=":", label="+u_d max / sqrt(2)")
    axs[1].plot(x_axis, -u_d_limit_plot, color="0.5", linewidth=1.0, linestyle=":", label="-u_d max / sqrt(2)")
    axs[1].legend()
    axs[1].set_title("u_d Verlauf PWM-diskret skaliert mit sqrt(2)")
    axs[1].set_ylabel("u_d / sqrt(2) [V]")

    axs[2].plot(x_axis, ia_plot, label="ia")
    axs[2].plot(x_axis, ib_plot, label="ib")
    axs[2].plot(x_axis, ic_plot, label="ic")
    axs[2].legend()
    axs[2].set_title("Phasenströme")

    axs[3].plot(x_axis, state_a_plot, label="HS_a")
    axs[3].plot(x_axis, state_b_plot, label="HS_b")
    axs[3].plot(x_axis, state_c_plot, label="HS_c")
    axs[3].legend()
    axs[3].set_title("Schaltzustände HS")

    axs[4].plot(x_axis, idc_plot, color="0.75", linewidth=0.8, label="DC-Link Strom")
    axs[4].plot(x_axis, i_source_plot, color="C3", linewidth=1.5, label="Quellstrom")
    axs[4].legend()
    axs[4].set_title("DC-Link Strom")

    axs[5].plot(x_axis, idc_rms_trace_plot, color="C1", label="DC-Link RMS")
    axs[5].legend()
    axs[5].set_title("RMS Verlauf DC-Link Strom")
    axs[5].set_xlabel(x_label)
    axs[5].set_ylim(0.0, float(np.max(idc_rms_trace_plot)) + 4.0)
    axs[5].yaxis.set_major_locator(MaxNLocator())
    axs[5].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}".rstrip("0").rstrip(".")))
    axs[5].yaxis.offsetText.set_visible(False)

    for ax in axs:
        ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def start_gui():
    root = tk.Tk()
    root.title("PSM DC Ripple Tool (Advanced)")

    entries = {}

    def add_field(key, label, default):
        row = tk.Frame(root)
        lab = tk.Label(row, width=20, text=label)
        ent = tk.Entry(row)
        ent.insert(0, str(default))
        row.pack()
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT)
        entries[key] = ent

    add_field("id_rms_min", "id_rms_min", 0)
    add_field("id_rms_max", "id_rms_max", 100)
    add_field("f_sig", "f_sig", 1000)
    add_field("f_pwm", "f_pwm", 10000)
    add_field("periods", "periods", 4)
    add_field("rms_margin_periods", "RMS Vor/Nachlauf", 5)
    add_field("theta_deg", "Rotorwinkel [deg]", 0)

    add_field("Ld", "Ld", 0.0002)
    add_field("Lq", "Lq", 0.0004)
    add_field("psi", "psi", 0.08)
    add_field("Rs", "Rs", 0.015)
    add_field("Udc", "Udc", 400)
    add_field("Cdc_uF", "Cdc_uF", 1000)
    add_field("Rsrc_mOhm", "Rsrc_mOhm", 20)
    add_field("Lsrc_uH", "Lsrc_uH", 0)

    form_var = tk.StringVar(value="Sinus")
    ttk.Combobox(root, textvariable=form_var, values=["Sinus", "Dreieck", "Rechteck"]).pack()

    def run():
        params = {key: float(widget.get()) for key, widget in entries.items()}
        params["form"] = form_var.get()
        run_sim(params)

    tk.Button(root, text="Start Simulation", command=run).pack()

    root.mainloop()


if __name__ == "__main__":
    start_gui()