from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox


SQRT3 = np.sqrt(3.0)
SQRT2 = np.sqrt(2.0)
DEFAULT_VDC = 640.0
DEFAULT_UD_RMS = -261.0
DEFAULT_UQ_RMS = 0.0
DEFAULT_MODULATION_INDEX = abs(DEFAULT_UD_RMS) * np.sqrt(6.0) / DEFAULT_VDC
MIN_MODULATION_INDEX = 1e-6
DISPLAY_ELECTRICAL_REVOLUTIONS = 6


@dataclass(frozen=True)
class SimulationConfig:
    vdc: float
    mechanical_speed_rpm: float
    pole_pairs: int
    switching_frequency_hz: float
    ud_rms: float
    uq_rms: float
    modulation_index: float
    electrical_angle_offset_deg: float
    display_seconds_per_mechanical_revolution: float
    trail_samples: int

    @property
    def mechanical_frequency_hz(self) -> float:
        return self.mechanical_speed_rpm / 60.0

    @property
    def electrical_frequency_hz(self) -> float:
        return self.pole_pairs * self.mechanical_frequency_hz

    @property
    def active_vector_radius(self) -> float:
        return (2.0 / 3.0) * self.vdc

    @property
    def linear_svm_radius(self) -> float:
        return self.vdc / SQRT3

    @property
    def max_modulation_index(self) -> float:
        return self.active_vector_radius / self.linear_svm_radius

    @property
    def electrical_angle_step_deg(self) -> float:
        return 360.0 * self.electrical_frequency_hz / self.switching_frequency_hz

    @property
    def steps_per_mechanical_revolution(self) -> float:
        return self.switching_frequency_hz / self.mechanical_frequency_hz


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Visualisiert das Spannungshexagon im alpha-beta-System und die "
            "dazugehoerigen drei Phasenspannungen in einer synchronen Animation."
        )
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=8000.0,
        help="Mechanische Drehzahl in rpm",
    )
    parser.add_argument(
        "--pole-pairs",
        type=int,
        default=3,
        help="Polpaarzahl der Maschine",
    )
    parser.add_argument(
        "--fsw",
        type=float,
        default=10000.0,
        help="Taktfrequenz bzw. PWM-Frequenz in Hz",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=DEFAULT_MODULATION_INDEX,
        help="Modulationsgrad bezogen auf den linearen SVM-Kreis",
    )
    parser.add_argument(
        "--ud-rms",
        type=float,
        default=DEFAULT_UD_RMS,
        help="u_d-Vorgabe als Effektivwert in V",
    )
    parser.add_argument(
        "--uq-rms",
        type=float,
        default=DEFAULT_UQ_RMS,
        help="u_q-Vorgabe als Effektivwert in V",
    )
    parser.add_argument(
        "--angle-offset-deg",
        type=float,
        default=0.0,
        help="Elektrischer Startwinkel-Offset in Grad",
    )
    parser.add_argument(
        "--display-seconds-per-rev",
        type=float,
        default=120.0,
        help="Anzeigezeit in Sekunden pro mechanischer Umdrehung",
    )
    parser.add_argument(
        "--trail-samples",
        type=int,
        default=40,
        help="Laenge der Spur des rotierenden Zeigers in Samples",
    )
    args = parser.parse_args()

    modulation_index = max(args.m, 0.0)
    vdc = calculate_vdc_from_dq(args.ud_rms, args.uq_rms, modulation_index)

    return SimulationConfig(
        vdc=vdc,
        mechanical_speed_rpm=max(args.rpm, 1e-9),
        pole_pairs=max(args.pole_pairs, 1),
        switching_frequency_hz=max(args.fsw, 1.0),
        ud_rms=args.ud_rms,
        uq_rms=args.uq_rms,
        modulation_index=max(args.m, 0.0),
        electrical_angle_offset_deg=args.angle_offset_deg,
        display_seconds_per_mechanical_revolution=max(args.display_seconds_per_rev, 1.0),
        trail_samples=max(args.trail_samples, 2),
    )


def calculate_vdc_from_dq(ud_rms: float, uq_rms: float, modulation_index: float) -> float:
    dq_radius_rms = float(np.hypot(ud_rms, uq_rms))
    if dq_radius_rms <= 1e-12:
        return 0.0
    return dq_radius_rms * np.sqrt(6.0) / max(modulation_index, MIN_MODULATION_INDEX)


def calculate_lead_angle_deg(ud_rms: float, uq_rms: float) -> float:
    if np.hypot(ud_rms, uq_rms) <= 1e-12:
        return 0.0
    return float(np.rad2deg(np.arctan2(-uq_rms, ud_rms)))


def create_hexagon(radius: float) -> np.ndarray:
    angles = np.deg2rad(np.arange(0.0, 361.0, 60.0))
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def inverse_clarke(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_a = alpha
    phase_b = -0.5 * alpha + (SQRT3 / 2.0) * beta
    phase_c = -0.5 * alpha - (SQRT3 / 2.0) * beta
    return phase_a, phase_b, phase_c


def hexagon_radius_limit(electrical_angle: np.ndarray, linear_svm_radius: float) -> np.ndarray:
    sector_angle = np.mod(electrical_angle, np.pi / 3.0) - (np.pi / 6.0)
    return linear_svm_radius / np.cos(sector_angle)


def build_signals(config: SimulationConfig) -> dict[str, np.ndarray]:
    mechanical_angle_step = 2.0 * np.pi * config.mechanical_frequency_hz / config.switching_frequency_hz
    total_mechanical_angle = DISPLAY_ELECTRICAL_REVOLUTIONS * 2.0 * np.pi / config.pole_pairs
    step_count = max(int(np.ceil(config.steps_per_mechanical_revolution * total_mechanical_angle / (2.0 * np.pi))), 2)
    mechanical_angle = np.minimum(np.arange(step_count + 1) * mechanical_angle_step, total_mechanical_angle)
    electrical_angle_offset = np.deg2rad(config.electrical_angle_offset_deg)
    electrical_angle = config.pole_pairs * mechanical_angle + electrical_angle_offset
    real_time = mechanical_angle / (2.0 * np.pi * config.mechanical_frequency_hz)
    display_time = mechanical_angle / (2.0 * np.pi) * config.display_seconds_per_mechanical_revolution

    requested_ud_peak = config.ud_rms * SQRT2
    requested_uq_peak = config.uq_rms * SQRT2
    requested_radius = float(np.hypot(requested_ud_peak, requested_uq_peak))
    lead_angle_deg = calculate_lead_angle_deg(config.ud_rms, config.uq_rms)
    lead_angle = np.deg2rad(lead_angle_deg)
    requested_modulation_index = float(np.clip(config.modulation_index, 0.0, config.max_modulation_index))

    available_radius = hexagon_radius_limit(electrical_angle, config.linear_svm_radius)
    applied_radius = np.minimum(requested_radius, available_radius)

    alpha = applied_radius * np.cos(electrical_angle)
    beta = applied_radius * np.sin(electrical_angle)
    phase_a, phase_b, phase_c = inverse_clarke(alpha, beta)
    rotor_angle = electrical_angle + lead_angle

    if requested_radius > 1e-12:
        ud_peak = applied_radius * (requested_ud_peak / requested_radius)
        uq_peak = applied_radius * (requested_uq_peak / requested_radius)
    else:
        ud_peak = np.zeros_like(applied_radius)
        uq_peak = np.zeros_like(applied_radius)

    requested_modulation_index_array = np.full_like(display_time, requested_modulation_index)
    applied_modulation_index = np.divide(
        applied_radius,
        config.linear_svm_radius,
        out=np.zeros_like(applied_radius),
        where=config.linear_svm_radius > 1e-12,
    )

    return {
        "real_time": real_time,
        "display_time": display_time,
        "mechanical_angle": mechanical_angle,
        "angle": electrical_angle,
        "electrical_angle_deg": np.rad2deg(electrical_angle),
        "rotor_angle": rotor_angle,
        "lead_angle_deg": np.full_like(display_time, lead_angle_deg),
        "requested_radius": np.full_like(display_time, requested_radius),
        "requested_radius_rms": np.full_like(display_time, requested_radius / SQRT2),
        "available_radius": available_radius,
        "applied_radius": applied_radius,
        "applied_radius_rms": applied_radius / SQRT2,
        "requested_modulation_index": requested_modulation_index_array,
        "applied_modulation_index": applied_modulation_index,
        "alpha": alpha,
        "beta": beta,
        "ua": phase_a,
        "ub": phase_b,
        "uc": phase_c,
        "ud": ud_peak / SQRT2,
        "uq": uq_peak / SQRT2,
    }


def build_phase_axis_ticks(max_electrical_angle_deg: float) -> tuple[np.ndarray, list[str]]:
    tick_step_deg = 180.0
    tick_values = np.arange(0.0, max_electrical_angle_deg + 0.5 * tick_step_deg, tick_step_deg)
    tick_labels = []
    for tick_value in tick_values:
        multiplier = int(round(tick_value / tick_step_deg))
        tick_labels.append("0" if multiplier == 0 else f"{multiplier}*180")
    return tick_values, tick_labels


def build_phase_axis_ticks_for_range(
    min_electrical_angle_deg: float,
    max_electrical_angle_deg: float,
) -> tuple[np.ndarray, list[str]]:
    tick_step_deg = 180.0
    start_tick = np.floor(min_electrical_angle_deg / tick_step_deg) * tick_step_deg
    tick_values = np.arange(start_tick, max_electrical_angle_deg + 0.5 * tick_step_deg, tick_step_deg)
    tick_labels = []
    for tick_value in tick_values:
        multiplier = int(round(tick_value / tick_step_deg))
        tick_labels.append("0" if multiplier == 0 else f"{multiplier}*180")
    return tick_values, tick_labels


def build_time_axis_ticks(
    min_electrical_angle_deg: float,
    max_electrical_angle_deg: float,
    electrical_frequency_hz: float,
) -> np.ndarray:
    phase_tick_values, _ = build_phase_axis_ticks_for_range(min_electrical_angle_deg, max_electrical_angle_deg)
    if electrical_frequency_hz <= 1e-12:
        return np.zeros_like(phase_tick_values)
    return (phase_tick_values - min_electrical_angle_deg) / (360.0 * electrical_frequency_hz)


def main() -> None:
    config = parse_args()
    signals = build_signals(config)

    real_time_s = signals["real_time"]
    electrical_angle_deg = signals["electrical_angle_deg"]
    circle_angle = np.linspace(0.0, 2.0 * np.pi, 400)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(13, 11.8))
    figure.subplots_adjust(left=0.08, right=0.82, top=0.93, bottom=0.17)
    grid = figure.add_gridspec(4, 1, height_ratios=[1.15, 0.95, 0.72, 0.72], hspace=0.08)

    ax_ab = figure.add_subplot(grid[0])
    ax_abc = figure.add_subplot(grid[1])
    ax_ud_plot = figure.add_subplot(grid[2])
    ax_uq_plot = figure.add_subplot(grid[3], sharex=ax_ud_plot)
    ax_vdc = figure.add_axes([0.08, 0.070, 0.14, 0.042])
    ax_rpm = figure.add_axes([0.24, 0.070, 0.14, 0.036])
    ax_fsw = figure.add_axes([0.40, 0.070, 0.14, 0.036])
    ax_angle_offset = figure.add_axes([0.56, 0.070, 0.14, 0.036])
    ax_pole_pairs = figure.add_axes([0.08, 0.020, 0.14, 0.036])
    ax_ud = figure.add_axes([0.24, 0.020, 0.14, 0.036])
    ax_uq = figure.add_axes([0.40, 0.020, 0.14, 0.036])
    ax_modulation = figure.add_axes([0.58, 0.028, 0.22, 0.030])
    figure.suptitle("Spannungshexagon und Phasenspannungen")
    figure.text(0.08, 0.125, "Eingaben", ha="left", va="center")

    hexagon_line, = ax_ab.plot([], [], color="black", linewidth=2.0, label="Hexagon")
    circle_line, = ax_ab.plot([], [], linestyle="--", color="0.45", label="linearer SVM-Kreis")
    ax_ab.axhline(0.0, color="0.75", linewidth=1.0)
    ax_ab.axvline(0.0, color="0.75", linewidth=1.0)
    ax_ab.set_aspect("equal", adjustable="box")
    ax_ab.set_xlabel(r"$u_{\alpha}$ [V]")
    ax_ab.set_ylabel(r"$u_{\beta}$ [V]")
    ax_ab.set_title("alpha-beta-Ebene mit PWM-Schritten")

    vector_line, = ax_ab.plot([], [], color="tab:red", linewidth=2.5, label="Referenzzeiger")
    vector_tip, = ax_ab.plot([], [], marker="o", color="tab:red", markersize=7)
    vector_trail, = ax_ab.plot([], [], color="tab:orange", linewidth=1.5, alpha=0.85, label="Spur")
    rotor_line, = ax_ab.plot([], [], color="0.35", linewidth=1.0, linestyle="--", label="Rotorzeiger")
    rotor_tip, = ax_ab.plot([], [], marker="o", color="0.35", markersize=4)
    status_text = figure.text(
        0.835,
        0.80,
        "",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax_ab.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    phase_colors = {"ua": "tab:blue", "ub": "tab:green", "uc": "tab:purple"}
    phase_labels = {"ua": r"$u_a$", "ub": r"$u_b$", "uc": r"$u_c$"}
    full_lines = {}
    active_lines = {}
    markers = {}
    for key in ("ua", "ub", "uc"):
        full_lines[key], = ax_abc.plot(
            electrical_angle_deg,
            signals[key],
            color=phase_colors[key],
            linewidth=1.0,
            alpha=0.25,
            drawstyle="steps-post",
        )
        active_lines[key], = ax_abc.plot(
            [],
            [],
            color=phase_colors[key],
            linewidth=2.0,
            label=phase_labels[key],
            drawstyle="steps-post",
        )
        markers[key], = ax_abc.plot([], [], marker="o", color=phase_colors[key], markersize=6)

    cursor = ax_abc.axvline(electrical_angle_deg[0], color="0.2", linestyle="--", linewidth=1.2)
    ax_abc.set_xlim(electrical_angle_deg[0], electrical_angle_deg[-1])
    phase_tick_values, phase_tick_labels = build_phase_axis_ticks_for_range(
        electrical_angle_deg[0],
        electrical_angle_deg[-1],
    )
    time_tick_values = build_time_axis_ticks(
        electrical_angle_deg[0],
        electrical_angle_deg[-1],
        config.electrical_frequency_hz,
    )
    ax_abc.set_xticks(phase_tick_values)
    ax_abc.set_xticklabels(phase_tick_labels)
    ax_abc.set_xlabel("gradEl")
    ax_abc.set_ylabel("Phasenspannung [V]")
    ax_abc.set_title("Drei Phasenspannungen ueber den elektrischen Winkel")
    ax_abc.legend(loc="upper right")

    dq_colors = {"ud": "tab:green", "uq": "tab:brown"}
    dq_labels = {"ud": r"$u_d$", "uq": r"$u_q$"}
    dq_axes = {"ud": ax_ud_plot, "uq": ax_uq_plot}
    dq_titles = {
        "ud": "$u_d$ im synchron rotierenden System",
        "uq": "$u_q$ im synchron rotierenden System",
    }
    dq_full_lines = {}
    dq_active_lines = {}
    dq_markers = {}
    dq_cursors = {}
    for key in ("ud", "uq"):
        dq_full_lines[key], = dq_axes[key].plot(
            real_time_s,
            signals[key],
            color=dq_colors[key],
            linewidth=1.0,
            alpha=0.25,
            drawstyle="steps-post",
        )
        dq_active_lines[key], = dq_axes[key].plot(
            [],
            [],
            color=dq_colors[key],
            linewidth=2.0,
            label=dq_labels[key],
            drawstyle="steps-post",
        )
        dq_markers[key], = dq_axes[key].plot([], [], marker="o", color=dq_colors[key], markersize=6)
        dq_cursors[key] = dq_axes[key].axvline(real_time_s[0], color="0.2", linestyle="--", linewidth=1.2)
        dq_axes[key].set_xlim(real_time_s[0], real_time_s[-1])
        dq_axes[key].set_xticks(time_tick_values)
        dq_axes[key].set_ylabel(f"{key} [Veff]")
        dq_axes[key].set_title(dq_titles[key])
        dq_axes[key].legend(loc="upper right")

    ax_ud_plot.tick_params(labelbottom=False)
    ax_uq_plot.set_xlabel("t in s")

    slider_values = np.round(np.arange(0.0, config.max_modulation_index + 0.001, 0.01), 2)
    if slider_values[-1] < config.max_modulation_index:
        slider_values = np.append(slider_values, config.max_modulation_index)
    modulation_slider = Slider(
        ax=ax_modulation,
        label="m",
        valmin=float(slider_values[0]),
        valmax=float(slider_values[-1]),
        valinit=float(np.clip(config.modulation_index, slider_values[0], slider_values[-1])),
        valstep=slider_values,
        valfmt="%1.2f",
    )
    ax_modulation.set_title("Modulationsgrad", fontsize=10, pad=2.0)

    ax_vdc.set_xticks([])
    ax_vdc.set_yticks([])
    ax_vdc.set_facecolor("white")
    for spine in ax_vdc.spines.values():
        spine.set_edgecolor("0.5")
    ax_vdc.set_title("Udc [V]", pad=4.0)
    vdc_value_text = ax_vdc.text(0.5, 0.5, "", ha="center", va="center", fontsize=11, transform=ax_vdc.transAxes)

    rpm_box = TextBox(ax_rpm, "Drehzahl [rpm]", initial=f"{config.mechanical_speed_rpm:.0f}")
    fsw_box = TextBox(ax_fsw, "Taktfreq. [Hz]", initial=f"{config.switching_frequency_hz:.0f}")
    angle_offset_box = TextBox(ax_angle_offset, "Offset [gradEl]", initial=f"{config.electrical_angle_offset_deg:.1f}")
    pole_pairs_box = TextBox(ax_pole_pairs, "Polpaare", initial=f"{config.pole_pairs:d}")
    ud_box = TextBox(ax_ud, "u_d [Veff]", initial=f"{config.ud_rms:.1f}")
    uq_box = TextBox(ax_uq, "u_q [Veff]", initial=f"{config.uq_rms:.1f}")

    state = {
        "frame_index": 0,
        "signals": signals,
        "config": config,
    }

    def compute_frame_interval_ms(active_config: SimulationConfig, active_signals: dict[str, np.ndarray]) -> float:
        return active_config.display_seconds_per_mechanical_revolution * 1000.0 / max(
            len(active_signals["display_time"]) - 1,
            1,
        )

    frame_interval_ms = compute_frame_interval_ms(config, signals)

    def refresh_computed_displays() -> None:
        current_config = state["config"]
        vdc_value_text.set_text(f"{current_config.vdc:6.1f}")

    def refresh_reference_geometry() -> None:
        current_config = state["config"]
        current_hexagon = create_hexagon(current_config.active_vector_radius)
        circle_x = current_config.linear_svm_radius * np.cos(circle_angle)
        circle_y = current_config.linear_svm_radius * np.sin(circle_angle)
        ab_limit = max(current_config.active_vector_radius * 1.15, 1.0)
        max_phase = max(current_config.active_vector_radius * 1.05, 1.0)

        hexagon_line.set_data(current_hexagon[:, 0], current_hexagon[:, 1])
        circle_line.set_data(circle_x, circle_y)
        ax_ab.set_xlim(-ab_limit, ab_limit)
        ax_ab.set_ylim(-ab_limit, ab_limit)
        ax_abc.set_ylim(-max_phase, max_phase)

    def refresh_static_lines() -> None:
        current_signals = state["signals"]
        ax_abc.set_xlim(current_signals["electrical_angle_deg"][0], current_signals["electrical_angle_deg"][-1])
        phase_tick_values, phase_tick_labels = build_phase_axis_ticks_for_range(
            current_signals["electrical_angle_deg"][0],
            current_signals["electrical_angle_deg"][-1],
        )
        time_tick_values = build_time_axis_ticks(
            current_signals["electrical_angle_deg"][0],
            current_signals["electrical_angle_deg"][-1],
            state["config"].electrical_frequency_hz,
        )
        ax_abc.set_xticks(phase_tick_values)
        ax_abc.set_xticklabels(phase_tick_labels)
        ax_ud_plot.set_xlim(current_signals["real_time"][0], current_signals["real_time"][-1])
        ax_uq_plot.set_xlim(current_signals["real_time"][0], current_signals["real_time"][-1])
        ax_ud_plot.set_xticks(time_tick_values)
        ax_uq_plot.set_xticks(time_tick_values)
        for key in ("ua", "ub", "uc"):
            full_lines[key].set_data(current_signals["electrical_angle_deg"], current_signals[key])
        for key in ("ud", "uq"):
            dq_full_lines[key].set_data(current_signals["real_time"], current_signals[key])
        update_dq_axis_limits(current_signals)

    def update_dq_axis_limits(current_signals: dict[str, np.ndarray]) -> None:
        for key in ("ud", "uq"):
            dq_min = float(np.min(current_signals[key]))
            dq_max = float(np.max(current_signals[key]))
            dq_span = max(dq_max - dq_min, 1.0)
            dq_margin = 0.08 * dq_span
            dq_axes[key].set_ylim(dq_min - dq_margin, dq_max + dq_margin)

    refresh_reference_geometry()
    refresh_static_lines()
    refresh_computed_displays()

    def init() -> tuple:
        vector_line.set_data([], [])
        vector_tip.set_data([], [])
        vector_trail.set_data([], [])
        rotor_line.set_data([], [])
        rotor_tip.set_data([], [])
        status_text.set_text("")
        cursor.set_xdata([state["signals"]["electrical_angle_deg"][0], state["signals"]["electrical_angle_deg"][0]])
        artists = [vector_line, vector_tip, vector_trail, rotor_line, rotor_tip, status_text, cursor]
        for key in ("ua", "ub", "uc"):
            active_lines[key].set_data([], [])
            markers[key].set_data([], [])
            artists.extend((active_lines[key], markers[key]))
        for key in ("ud", "uq"):
            dq_active_lines[key].set_data([], [])
            dq_markers[key].set_data([], [])
            dq_cursors[key].set_xdata([state["signals"]["real_time"][0], state["signals"]["real_time"][0]])
            artists.extend((dq_active_lines[key], dq_markers[key], dq_cursors[key]))
        return tuple(artists)

    def update(frame_index: int) -> tuple:
        current_signals = state["signals"]
        current_config = state["config"]
        frame_index = frame_index % len(current_signals["display_time"])
        state["frame_index"] = frame_index
        alpha_value = current_signals["alpha"][frame_index]
        beta_value = current_signals["beta"][frame_index]
        vector_line.set_data([0.0, alpha_value], [0.0, beta_value])
        vector_tip.set_data([alpha_value], [beta_value])

        rotor_radius = current_signals["applied_radius"][frame_index]
        rotor_angle = current_signals["rotor_angle"][frame_index]
        rotor_alpha = rotor_radius * np.cos(rotor_angle)
        rotor_beta = rotor_radius * np.sin(rotor_angle)
        rotor_line.set_data([0.0, rotor_alpha], [0.0, rotor_beta])
        rotor_tip.set_data([rotor_alpha], [rotor_beta])

        trail_start = max(0, frame_index - current_config.trail_samples)
        vector_trail.set_data(
            current_signals["alpha"][trail_start : frame_index + 1],
            current_signals["beta"][trail_start : frame_index + 1],
        )

        cursor_time = current_signals["real_time"][frame_index]
        cursor_angle = current_signals["electrical_angle_deg"][frame_index]
        cursor.set_xdata([cursor_angle, cursor_angle])

        artists = [vector_line, vector_tip, vector_trail, rotor_line, rotor_tip, status_text, cursor]
        for key in ("ua", "ub", "uc"):
            active_lines[key].set_data(
                current_signals["electrical_angle_deg"][: frame_index + 1],
                current_signals[key][: frame_index + 1],
            )
            markers[key].set_data([cursor_angle], [current_signals[key][frame_index]])
            artists.extend((active_lines[key], markers[key]))
        for key in ("ud", "uq"):
            dq_active_lines[key].set_data(
                current_signals["real_time"][: frame_index + 1],
                current_signals[key][: frame_index + 1],
            )
            dq_markers[key].set_data([cursor_time], [current_signals[key][frame_index]])
            dq_cursors[key].set_xdata([cursor_time, cursor_time])
            artists.extend((dq_active_lines[key], dq_markers[key], dq_cursors[key]))

        electrical_angle_deg = np.rad2deg(current_signals["angle"][frame_index]) % 360.0
        mechanical_angle_deg = np.rad2deg(current_signals["mechanical_angle"][frame_index]) % 360.0
        requested_radius = current_signals["requested_radius"][frame_index]
        applied_radius = current_signals["applied_radius"][frame_index]
        is_limited = requested_radius > applied_radius + 1e-9
        status_text.set_text(
            "\n".join(
                (
                    f"t_real = {cursor_time * 1e3:6.3f} ms",
                    f"theta_mech = {mechanical_angle_deg:6.1f} deg",
                    f"theta_el = {electrical_angle_deg:6.1f} deg",
                    f"u_d = {current_signals['ud'][frame_index]:6.1f} Veff, u_q = {current_signals['uq'][frame_index]:6.1f} Veff",
                    f"Udc = {current_config.vdc:6.1f} V, rpm = {current_config.mechanical_speed_rpm:7.1f}, p = {current_config.pole_pairs:d}",
                    f"fsw = {current_config.switching_frequency_hz:7.1f} Hz",
                    f"f_el = {current_config.electrical_frequency_hz:6.1f} Hz, dtheta_el/PWM = {current_config.electrical_angle_step_deg:5.2f} deg",
                    f"|u_dq|_eff = {current_signals['applied_radius_rms'][frame_index]:6.1f} Veff",
                    f"m = {current_signals['requested_modulation_index'][frame_index]:4.2f}, m_appl = {current_signals['applied_modulation_index'][frame_index]:4.2f}",
                    f"Begrenzung = {'ja' if is_limited else 'nein'}",
                )
            )
        )
        return tuple(artists)

    def rebuild_signals(reset_frame: bool = False) -> None:
        if reset_frame:
            state["frame_index"] = 0
        state["signals"] = build_signals(state["config"])
        refresh_reference_geometry()
        refresh_static_lines()
        refresh_computed_displays()
        anim.event_source.interval = compute_frame_interval_ms(state["config"], state["signals"])
        update(state["frame_index"])
        figure.canvas.draw_idle()

    def apply_runtime_config(reset_frame: bool = False, **changes: float | int) -> None:
        updated_config = replace(state["config"], **changes)
        if any(key in changes for key in ("ud_rms", "uq_rms", "modulation_index")):
            updated_config = replace(
                updated_config,
                vdc=calculate_vdc_from_dq(
                    updated_config.ud_rms,
                    updated_config.uq_rms,
                    updated_config.modulation_index,
                ),
            )
        state["config"] = updated_config
        rebuild_signals(reset_frame=reset_frame)

    def on_rpm_submit(text: str) -> None:
        try:
            rpm_value = max(float(text), 1e-9)
        except ValueError:
            return
        apply_runtime_config(mechanical_speed_rpm=rpm_value)

    def on_fsw_submit(text: str) -> None:
        try:
            fsw_value = max(float(text), 1.0)
        except ValueError:
            return
        apply_runtime_config(switching_frequency_hz=fsw_value)

    def on_pole_pairs_submit(text: str) -> None:
        try:
            pole_pairs_value = max(int(float(text)), 1)
        except ValueError:
            return
        apply_runtime_config(pole_pairs=pole_pairs_value)

    def on_angle_offset_submit(text: str) -> None:
        try:
            angle_offset_value = float(text)
        except ValueError:
            return
        apply_runtime_config(reset_frame=True, electrical_angle_offset_deg=angle_offset_value)

    def on_modulation_change(value: float) -> None:
        apply_runtime_config(modulation_index=max(float(value), 0.0))

    def on_ud_submit(text: str) -> None:
        try:
            ud_value = float(text)
        except ValueError:
            return
        apply_runtime_config(ud_rms=ud_value)

    def on_uq_submit(text: str) -> None:
        try:
            uq_value = float(text)
        except ValueError:
            return
        apply_runtime_config(uq_rms=uq_value)

    modulation_slider.on_changed(on_modulation_change)
    rpm_box.on_submit(on_rpm_submit)
    fsw_box.on_submit(on_fsw_submit)
    angle_offset_box.on_submit(on_angle_offset_submit)
    pole_pairs_box.on_submit(on_pole_pairs_submit)
    ud_box.on_submit(on_ud_submit)
    uq_box.on_submit(on_uq_submit)

    anim = animation.FuncAnimation(
        figure,
        update,
        init_func=init,
        frames=None,
        interval=frame_interval_ms,
        blit=False,
        repeat=True,
    )
    figure._animation = anim
    figure.canvas.draw_idle()

    plt.show(block=True)


if __name__ == "__main__":
    main()