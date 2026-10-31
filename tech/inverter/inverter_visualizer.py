"""Einfache, interaktive Visualisierung eines dreiphasigen Inverters.

Start:
    py tech/inverter/inverter_visualizer.py

Es ist bewusst kein Motorregler. Eine idealisierte SPMSM berechnet aus
Stromamplitude und Stromwinkel ein elektromagnetisches Moment. Die
Spannungsamplitude wird nur fuer die Zeichnung aus Drehzahl und Strom
abgeleitet.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from tkinter import font as tkfont


PHASES = ("a", "b", "c")
PHASE_OFFSETS = (0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
PHASE_COLOURS = ("#1d4ed8", "#059669", "#c2410c")


@dataclass(frozen=True)
class MachineValues:
    """Momentaufnahme der absichtlich einfachen Visualisierungsformel."""

    current_peak: float
    voltage_peak: float
    electrical_frequency: float
    iq_peak: float
    torque_nm: float
    stator_field_peak_t: float


class InverterVisualizer(tk.Tk):
    """GUI mit drei Strom- und Spannungsverlaeufen sowie einem einfachen Motorbild."""

    canvas_width = 1050
    canvas_height = 660
    pole_pairs = 1
    flux_linkage_wb = 0.08
    stator_field_per_amp_t = 0.01
    update_ms = 25

    def __init__(self) -> None:
        super().__init__()
        self.title("Dreiphasen-Inverter – einfache Visualisierung")
        self.minsize(920, 720)
        self.configure(padx=14, pady=12)

        self.speed_rpm = tk.DoubleVar(value=0.0)
        self.current_amplitude_a = tk.DoubleVar(value=50.0)
        self.current_phase_deg = tk.DoubleVar(value=90.0)
        self.speed_input = tk.StringVar(value="0")
        self.amplitude_input = tk.StringVar(value="50")
        self.phase_input = tk.StringVar(value="90")
        self.coil_voltage_v = tk.DoubleVar(value=0.0)
        self.coil_voltage_input = tk.StringVar(value="0")
        self.coil_current_a = 0.0
        # Feste Modellwerte; sie werden nicht als Bedienwerte angezeigt.
        self.coil_resistance_ohm = 1.0
        self.coil_inductance_h = 0.25
        self.coil_field_per_amp_t = 0.05
        self.coil_elapsed_s = 0.0
        # Kurzer Verlauf fuer dieselbe Diagramm-Sprache wie auf der Motorseite.
        self.coil_history: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
        # Seite 2: ein absichtlich vereinfachter Inverter. Jede Phase ist
        # entweder spannungsfrei oder direkt mit +Udc verbunden.
        self.test_phase_active = [False, False, False]
        self.test_rotor_angle = 0.0
        self.test_rotor_speed_rad_s = 0.0
        self.test_torque = 0.0
        # Normierte Modellwerte fuer die sichtbare, träge Rotorbewegung.
        self.test_inertia = 0.35
        self.test_damping = 1.10
        self.test_torque_constant = 2.20
        self.test_rotor_history: list[float] = [0.0]
        self.running = True
        self.electrical_angle = 0.0
        self.last_time_ms = self.tk.call("clock", "milliseconds")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Reihenfolge der Lernseiten: erst die Einzelspule, dann der noch
        # vorzubereitende Versuch und zuletzt die bestehende Motoransicht.
        self.coil_tab = ttk.Frame(self.notebook)
        self.motor_test_tab = ttk.Frame(self.notebook, padding=18)
        self.motor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.coil_tab, text="1  Spule")
        self.notebook.add(self.motor_test_tab, text="2  Motor-versuch")
        self.notebook.add(self.motor_tab, text="3  Motor")
        self._build_coil_page()
        self._build_motor_test_page()

        self._build_controls(self.motor_tab)
        self.canvas = tk.Canvas(
            self.motor_tab,
            width=self.canvas_width,
            height=self.canvas_height,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.motor_tab.columnconfigure(0, weight=1)
        self.motor_tab.rowconfigure(1, weight=1)

        self.bind("<space>", lambda _event: self._toggle_animation())
        for key, phase_index in (("q", 1), ("s", 2), ("e", 0), ("Q", 1), ("S", 2), ("E", 0)):
            self.bind(
                f"<KeyPress-{key}>",
                lambda event, index=phase_index: self._set_test_phase_from_key(event, index, True),
            )
            self.bind(
                f"<KeyRelease-{key}>",
                lambda event, index=phase_index: self._set_test_phase_from_key(event, index, False),
            )
        self.state("zoomed")
        self.after(self.update_ms, self._animate)

    @staticmethod
    def _build_placeholder_page(parent: ttk.Frame, *, title: str, description: str) -> None:
        """Erstellt eine vorbereitete, noch nicht mit Inhalt gefuellte Lernseite."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        content = ttk.Frame(parent, padding=24)
        content.grid(row=0, column=0, sticky="nsew")
        ttk.Label(content, text=title, font=("Segoe UI", 22, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(content, text=description, foreground="#475569").grid(row=1, column=0, sticky="w", pady=(10, 0))

    def _build_motor_test_page(self) -> None:
        """Erstellt den manuellen, bewusst stark vereinfachten Inverter-Versuch."""
        controls = ttk.LabelFrame(self.motor_test_tab, text="Phasenschalter: 0 V oder +Udc", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        self.test_phase_buttons: list[tk.Button] = []
        for button_index, (phase_index, key) in enumerate(zip((1, 2, 0), ("Q", "S", "E"))):
            button = tk.Button(
                controls,
                font=("Segoe UI", 10, "bold"),
                relief="raised",
                padx=12,
                pady=5,
            )
            button.bind("<ButtonPress-1>", lambda _event, index=phase_index: self._set_test_phase(index, True))
            button.bind("<ButtonRelease-1>", lambda _event, index=phase_index: self._set_test_phase(index, False))
            button.grid(row=0, column=button_index, padx=(0, 8), sticky="w")
            self.test_phase_buttons.append(button)
        ttk.Button(controls, text="Rotor zurücksetzen", command=self._reset_test_rotor).grid(row=0, column=3, padx=(12, 18))
        ttk.Label(
            controls,
            text="Tastatur: Q = Phase V, S = Phase W, E = Phase U. +Udc liegt nur an, solange Taste oder Taster gedrückt ist.",
            foreground="#475569",
        ).grid(row=0, column=4, sticky="w")
        controls.columnconfigure(4, weight=1)

        self.test_canvas = tk.Canvas(
            self.motor_test_tab,
            width=self.canvas_width,
            height=self.canvas_height,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.test_canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.motor_test_tab.columnconfigure(0, weight=1)
        self.motor_test_tab.rowconfigure(1, weight=1)
        self._refresh_test_phase_buttons()
        self._draw_motor_test_page()

    def _set_test_phase_from_key(self, event: tk.Event, phase_index: int, active: bool) -> str | None:
        """Nimmt Q/S/E nur auf der Versuchsseite an, damit Eingabefelder frei bleiben."""
        if self.notebook.select() != str(self.motor_test_tab):
            return None
        self._set_test_phase(phase_index, active)
        return "break"

    def _set_test_phase(self, phase_index: int, active: bool) -> None:
        if self.test_phase_active[phase_index] == active:
            return
        self.test_phase_active[phase_index] = active
        self._refresh_test_phase_buttons()
        self._draw_motor_test_page()

    def _refresh_test_phase_buttons(self) -> None:
        for button, phase_index, key in zip(
            self.test_phase_buttons,
            (1, 2, 0),
            ("Q", "S", "E"),
        ):
            phase = ("U", "V", "W")[phase_index]
            active = self.test_phase_active[phase_index]
            if active:
                button.config(
                    text=f"{key}: {phase} = +Udc",
                    background="#16a34a",
                    foreground="white",
                    activebackground="#15803d",
                    activeforeground="white",
                )
            else:
                button.config(
                    text=f"{key}: {phase} = 0 V",
                    background="#e2e8f0",
                    foreground="#334155",
                    activebackground="#cbd5e1",
                    activeforeground="#334155",
                )

    def _reset_test_rotor(self) -> None:
        self.test_rotor_angle = 0.0
        self.test_rotor_speed_rad_s = 0.0
        self.test_torque = 0.0
        self.test_rotor_history = [0.0]
        self._draw_motor_test_page()

    def _test_stator_field(self) -> tuple[float, float, float]:
        """Resultierender Südpol aus drei gleich grossen, geschalteten Phasen."""
        field_x = 0.0
        field_y = 0.0
        for active, phase_angle in zip(self.test_phase_active, (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)):
            if active:
                field_x += math.cos(phase_angle)
                field_y += math.sin(phase_angle)
        field_strength = math.hypot(field_x, field_y)
        field_angle = math.atan2(field_y, field_x) if field_strength > 1e-9 else 0.0
        return field_angle, field_strength, field_x

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _update_test_rotor(self, elapsed_s: float) -> None:
        """Integriert ein kleines Drehmomentmodell mit Trägheit und Reibung."""
        step_s = min(max(elapsed_s, 0.0), 0.05)
        field_angle, field_strength, _field_x = self._test_stator_field()
        # Der geschaltete Stator-Südpol zieht den Rotor-Nordpol an.
        error = self._wrap_angle(field_angle - self.test_rotor_angle)
        self.test_torque = self.test_torque_constant * field_strength * math.sin(error)
        acceleration = (self.test_torque - self.test_damping * self.test_rotor_speed_rad_s) / self.test_inertia
        self.test_rotor_speed_rad_s += acceleration * step_s
        self.test_rotor_angle += self.test_rotor_speed_rad_s * step_s
        self.test_rotor_angle %= 2.0 * math.pi
        self.test_rotor_history.append(self.test_rotor_angle)
        if len(self.test_rotor_history) > 240:
            del self.test_rotor_history[:-240]

    def _draw_motor_test_page(self) -> None:
        """Zeichnet den manuellen Schaltversuch im bekannten Motorseiten-Raster."""
        canvas = self.test_canvas
        canvas.delete("all")
        field_angle, field_strength, _field_x = self._test_stator_field()
        active_phases = [phase for phase, active in zip(("U", "V", "W"), self.test_phase_active) if active]
        active_text = ", ".join(active_phases) if active_phases else "keine"
        if field_strength <= 1e-9:
            result_text = "kein resultierender Statorpol"
        else:
            result_text = f"S_res bei {math.degrees(field_angle) % 360.0:.0f} Grad"
        test_speed_rpm = self.test_rotor_speed_rad_s * 60.0 / (2.0 * math.pi)

        canvas.create_text(
            25,
            25,
            anchor="w",
            text="Motor-Versuch: Statorpol manuell schalten",
            font=("Segoe UI", 18, "bold"),
            fill="#0f172a",
        )
        canvas.create_text(
            25,
            55,
            anchor="w",
            text=f"Aktiv: {active_text}   |   {result_text}   |   T_em = {self.test_torque:+.2f}   |   n = {test_speed_rpm:+.1f} rpm",
            font=("Segoe UI", 12, "bold"),
            fill="#166534" if field_strength > 1e-9 else "#475569",
        )

        self._draw_test_phase_panel()
        self._draw_test_field_panel(field_angle, field_strength)
        self._draw_test_rotor_panel()
        self._draw_test_machine(field_angle, field_strength)
        self._draw_test_esb()
        self._scale_canvas_to_view(canvas)

    def _draw_test_phase_panel(self) -> None:
        canvas = self.test_canvas
        left, top, width, height = 25, 95, 610, 195
        canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        canvas.create_text(left + 12, top + 13, text="Phasen: 0 V oder +Udc", anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_text(
            left + width - 12,
            top + 13,
            text="Q / S / E",
            anchor="e",
            font=("Segoe UI", 9, "bold"),
            fill="#475569",
        )
        for phase_index, (phase, key, angle_deg, active) in enumerate(
            zip(("U", "V", "W"), ("E", "Q", "S"), (0, 120, 240), self.test_phase_active)
        ):
            row_y = top + 55 + phase_index * 42
            fill = "#dcfce7" if active else "#f8fafc"
            outline = "#16a34a" if active else "#94a3b8"
            text_colour = "#166534" if active else "#475569"
            canvas.create_rectangle(left + 24, row_y - 14, left + 118, row_y + 14, outline=outline, fill=fill, width=2)
            canvas.create_text(left + 71, row_y, text=f"{key}: {phase}", font=("Segoe UI", 10, "bold"), fill="#0f172a")
            canvas.create_line(left + 140, row_y, left + 455, row_y, fill="#cbd5e1", width=2)
            if active:
                canvas.create_line(left + 140, row_y, left + 455, row_y, fill="#16a34a", width=5)
                canvas.create_text(left + 500, row_y, text="+Udc  ->  S", font=("Segoe UI", 11, "bold"), fill="#166534")
            else:
                canvas.create_text(left + 500, row_y, text="0 V", font=("Segoe UI", 11, "bold"), fill=text_colour)
            canvas.create_text(left + 585, row_y, text=f"{angle_deg}°", anchor="e", font=("Segoe UI", 9), fill="#64748b")

    def _draw_test_field_panel(self, field_angle: float, field_strength: float) -> None:
        canvas = self.test_canvas
        left, top, width, height = 25, 310, 610, 195
        canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        canvas.create_text(left + 12, top + 13, text="Resultierender Statorpol", anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        centre_x, centre_y, radius = 170, 410, 68
        canvas.create_oval(centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius, outline="#94a3b8", width=2, fill="#f8fafc")
        for angle_deg in (0, 120, 240):
            angle = math.radians(angle_deg)
            x = centre_x + radius * math.cos(angle)
            y = centre_y - radius * math.sin(angle)
            canvas.create_text(x, y, text=str(angle_deg) + "°", font=("Segoe UI", 8), fill="#64748b")
        if field_strength > 1e-9:
            field_length = 50 + 18 * min(field_strength, 1.0)
            canvas.create_line(
                centre_x,
                centre_y,
                centre_x + field_length * math.cos(field_angle),
                centre_y - field_length * math.sin(field_angle),
                fill="#16a34a",
                width=5,
                arrow=tk.LAST,
            )
            canvas.create_text(
                290,
                382,
                anchor="w",
                text=f"S_res = {math.degrees(field_angle) % 360.0:.0f}°",
                font=("Segoe UI", 14, "bold"),
                fill="#166534",
            )
            canvas.create_text(
                290,
                413,
                anchor="w",
                text="Der N-Pol des Rotors richtet sich dorthin aus.",
                font=("Segoe UI", 10),
                fill="#475569",
            )
        else:
            canvas.create_text(290, 393, anchor="w", text="Kein resultierender Pol", font=("Segoe UI", 14, "bold"), fill="#64748b")
            canvas.create_text(
                290,
                423,
                anchor="w",
                text="Bei U + V + W heben sich die drei Felder auf.",
                font=("Segoe UI", 10),
                fill="#475569",
            )
        canvas.create_text(
            290,
            460,
            anchor="w",
            text=f"Feldstärke (normiert): {field_strength:.1f}",
            font=("Segoe UI", 10, "bold"),
            fill="#475569",
        )

    def _draw_test_rotor_panel(self) -> None:
        canvas = self.test_canvas
        left, top, width, height = 25, 525, 610, 90
        canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        rotor_angle_deg = math.degrees(self.test_rotor_angle) % 360.0
        canvas.create_text(left + 12, top + 13, text="Rotorwinkel", anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_text(left + width - 12, top + 13, text=f"{rotor_angle_deg:6.1f} Grad", anchor="e", font=("Segoe UI", 9), fill="#b45309")
        plot_left, plot_right = left + 55, left + width - 18
        plot_top, plot_bottom = top + 30, top + height - 13
        canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="#94a3b8", dash=(3, 3))
        canvas.create_text(left + 8, plot_top, text="360", anchor="w", font=("Segoe UI", 8), fill="#64748b")
        canvas.create_text(left + 18, plot_bottom, text="0", anchor="w", font=("Segoe UI", 8), fill="#64748b")
        points: list[float] = []
        sample_count = max(1, len(self.test_rotor_history) - 1)
        for point_index, angle in enumerate(self.test_rotor_history):
            x = plot_left + (plot_right - plot_left) * point_index / sample_count
            y = plot_bottom - (plot_bottom - plot_top) * (math.degrees(angle) % 360.0) / 360.0
            points.extend((x, y))
        if len(points) == 2:
            points.extend(points)
        canvas.create_line(*points, fill="#d97706", width=2)
        marker_y = points[-1]
        canvas.create_oval(plot_right - 4, marker_y - 4, plot_right + 4, marker_y + 4, outline="#d97706", fill="#d97706")

    def _draw_test_machine(self, field_angle: float, field_strength: float) -> None:
        canvas = self.test_canvas
        centre_x, centre_y = 830, 280
        outer_radius, rotor_radius = 182, 78
        canvas.create_text(centre_x, 66, text="Dreiphasen-Maschine", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_oval(
            centre_x - outer_radius,
            centre_y - outer_radius,
            centre_x + outer_radius,
            centre_y + outer_radius,
            outline="#334155",
            width=3,
            fill="#e2e8f0",
        )

        phase_angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
        for phase, phase_angle, active in zip(("U", "V", "W"), phase_angles, self.test_phase_active):
            radial_x, radial_y = math.cos(phase_angle), -math.sin(phase_angle)
            tangent_x, tangent_y = -math.sin(phase_angle), -math.cos(phase_angle)
            coil_x = centre_x + 150 * radial_x
            coil_y = centre_y + 150 * radial_y
            coil_colour = "#16a34a" if active else "#64748b"
            coil_points: list[float] = []
            for point_index in range(25):
                progress = point_index / 24.0
                along = -23.0 + 46.0 * progress
                offset = 7.0 * math.sin(3.0 * 2.0 * math.pi * progress)
                coil_points.extend((coil_x + along * radial_x + offset * tangent_x, coil_y + along * radial_y + offset * tangent_y))
            canvas.create_line(*coil_points, fill=coil_colour, width=3, smooth=True, splinesteps=12)
            canvas.create_line(coil_x - 33 * radial_x, coil_y - 33 * radial_y, coil_x - 23 * radial_x, coil_y - 23 * radial_y, fill=coil_colour, width=3)
            canvas.create_line(coil_x + 23 * radial_x, coil_y + 23 * radial_y, coil_x + 33 * radial_x, coil_y + 33 * radial_y, fill=coil_colour, width=3)
            canvas.create_text(
                centre_x + 174 * radial_x,
                centre_y + 174 * radial_y,
                text=f"{phase} {'+Udc' if active else '0 V'}",
                font=("Segoe UI", 8, "bold"),
                fill=coil_colour,
            )
            if active:
                pole_x = centre_x + 118 * radial_x
                pole_y = centre_y + 118 * radial_y
                canvas.create_oval(pole_x - 16, pole_y - 16, pole_x + 16, pole_y + 16, fill="#2563eb", outline="#1d4ed8", width=2)
                canvas.create_text(pole_x, pole_y, text="S", font=("Segoe UI", 12, "bold"), fill="white")

        if field_strength > 1e-9:
            field_length = 62 + 80 * min(field_strength, 1.0)
            canvas.create_line(
                centre_x,
                centre_y,
                centre_x + field_length * math.cos(field_angle),
                centre_y - field_length * math.sin(field_angle),
                fill="#16a34a",
                width=4,
                arrow=tk.LAST,
            )
            canvas.create_text(
                centre_x + 105 * math.cos(field_angle),
                centre_y - 105 * math.sin(field_angle) - 15,
                text="S_res",
                font=("Segoe UI", 9, "bold"),
                fill="#166534",
            )

        direction_x, direction_y = math.cos(self.test_rotor_angle), -math.sin(self.test_rotor_angle)
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        corners: list[float] = []
        for longitudinal, lateral in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            corners.extend(
                (
                    centre_x + longitudinal * 67 * direction_x + lateral * 29 * perpendicular_x,
                    centre_y + longitudinal * 67 * direction_y + lateral * 29 * perpendicular_y,
                )
            )
        canvas.create_polygon(*corners, fill="#fef3c7", outline="#334155", width=2)
        canvas.create_line(
            centre_x,
            centre_y,
            centre_x + 67 * direction_x,
            centre_y + 67 * direction_y,
            fill="#dc2626",
            width=58,
        )
        canvas.create_line(
            centre_x,
            centre_y,
            centre_x - 67 * direction_x,
            centre_y - 67 * direction_y,
            fill="#2563eb",
            width=58,
        )
        canvas.create_text(centre_x + 40 * direction_x, centre_y + 40 * direction_y, text="N", font=("Segoe UI", 13, "bold"), fill="white")
        canvas.create_text(centre_x - 40 * direction_x, centre_y - 40 * direction_y, text="S", font=("Segoe UI", 13, "bold"), fill="white")
        canvas.create_oval(centre_x - rotor_radius, centre_y - rotor_radius, centre_x + rotor_radius, centre_y + rotor_radius, outline="#334155", width=2)
        rlg_radius = 110
        canvas.create_line(
            centre_x,
            centre_y,
            centre_x + rlg_radius * direction_x,
            centre_y + rlg_radius * direction_y,
            fill="#000000",
            width=2,
            arrow=tk.LAST,
        )
        canvas.create_text(
            centre_x + 138 * direction_x,
            centre_y + 138 * direction_y,
            text="RLG 0°",
            font=("Segoe UI", 9, "bold"),
            fill="#000000",
        )
        canvas.create_text(
            centre_x,
            468,
            text="T_em aus Winkelfehler bewegt den Rotor",
            font=("Segoe UI", 9),
            fill="#475569",
        )

    def _draw_test_esb(self) -> None:
        canvas = self.test_canvas
        left, top, right, bottom = 655, 493, 1025, 646
        canvas.create_rectangle(left, top, right, bottom, outline="#94a3b8", fill="#ffffff")
        canvas.create_text(left + 12, top + 13, text="Vereinfachter Inverter", anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_text(right - 12, top + 13, text="nur 0 V / +Udc", anchor="e", font=("Segoe UI", 9), fill="#475569")
        plus_y, zero_y = 527, 582
        rail_left, rail_right = 694, 895
        canvas.create_line(rail_left, plus_y, rail_right, plus_y, fill="#dc2626", width=2)
        canvas.create_line(rail_left, zero_y, rail_right, zero_y, fill="#64748b", width=2)
        canvas.create_text(680, plus_y, text="+Udc", anchor="e", font=("Segoe UI", 8, "bold"), fill="#dc2626")
        canvas.create_text(680, zero_y, text="0 V", anchor="e", font=("Segoe UI", 8, "bold"), fill="#475569")
        phase_x_values = (746, 803, 860)
        for phase, phase_x, active, colour in zip(("U", "V", "W"), phase_x_values, self.test_phase_active, PHASE_COLOURS):
            switch_fill = "#16a34a" if active else "#ffffff"
            switch_text = "+Udc" if active else "0 V"
            switch_text_colour = "white" if active else "#475569"
            canvas.create_line(phase_x, plus_y, phase_x, 541, fill="#475569", width=1)
            canvas.create_rectangle(phase_x - 17, 541, phase_x + 17, 557, outline="#166534" if active else "#64748b", fill=switch_fill, width=2)
            canvas.create_text(phase_x, 549, text=switch_text, font=("Segoe UI", 7, "bold"), fill=switch_text_colour)
            canvas.create_line(phase_x, 557, phase_x, 567, fill=colour, width=2)
            canvas.create_oval(phase_x - 3, 564, phase_x + 3, 570, fill=colour, outline=colour)
            canvas.create_line(phase_x, 567, 935, 567, fill=colour, width=2)
            canvas.create_text(phase_x, 603, text=phase, font=("Segoe UI", 9, "bold"), fill=colour)
        motor_x, motor_y, motor_radius = 970, 565, 30
        canvas.create_oval(motor_x - motor_radius, motor_y - motor_radius, motor_x + motor_radius, motor_y + motor_radius, outline="#334155", width=2, fill="#f8fafc")
        canvas.create_text(motor_x, motor_y - 5, text="M", font=("Segoe UI", 14, "bold"), fill="#0f172a")
        canvas.create_text(motor_x, motor_y + 12, text="PSM", font=("Segoe UI", 8), fill="#475569")

    def _build_coil_page(self) -> None:
        """Erstellt die einfache RL-Lernseite mit regelbarer Batteriespannung."""
        controls = ttk.LabelFrame(self.coil_tab, text="Batterie", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Batteriespannung").grid(row=0, column=0, sticky="w")
        voltage_slider = ttk.Scale(
            controls,
            from_=-10,
            to=10,
            orient="horizontal",
            variable=self.coil_voltage_v,
            command=lambda _value: self._draw_coil_page(),
        )
        voltage_slider.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.coil_voltage_entry = ttk.Entry(controls, width=10, justify="right", textvariable=self.coil_voltage_input)
        self.coil_voltage_entry.grid(row=0, column=2, sticky="e")
        ttk.Label(controls, text="V").grid(row=0, column=3, sticky="w", padx=(4, 18))
        ttk.Button(controls, text="Spannung zurücksetzen", command=self._reset_coil_voltage).grid(row=0, column=4)
        ttk.Label(
            controls,
            text="Die festen R- und L-Werte wirken nur intern auf den Stromverlauf.",
            foreground="#475569",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        self.coil_canvas = tk.Canvas(
            self.coil_tab,
            width=self.canvas_width,
            height=self.canvas_height,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.coil_canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.coil_tab.columnconfigure(0, weight=1)
        self.coil_tab.rowconfigure(1, weight=1)

        def apply_voltage(_event: tk.Event | None = None) -> None:
            try:
                voltage = float(self.coil_voltage_input.get().replace(",", "."))
            except ValueError:
                voltage = self.coil_voltage_v.get()
            voltage = min(10.0, max(-10.0, voltage))
            self.coil_voltage_v.set(voltage)
            self.coil_voltage_input.set(f"{voltage:.1f}")
            self._draw_coil_page()

        self.coil_voltage_entry.bind("<Return>", apply_voltage)
        self.coil_voltage_entry.bind("<FocusOut>", apply_voltage)

    def _reset_coil_voltage(self) -> None:
        """Setzt nur die Batteriespannung auf 0 V; der RL-Strom klingt ab."""
        self.coil_voltage_v.set(0.0)
        self.coil_voltage_input.set("0.0")
        self._draw_coil_page()

    def _update_coil_current(self, elapsed_s: float) -> None:
        """Integriert das einfache RL-Modell: U = R*i + L*di/dt."""
        step_s = min(max(elapsed_s, 0.0), 0.05)
        voltage = self.coil_voltage_v.get()
        current_slope = (voltage - self.coil_resistance_ohm * self.coil_current_a) / self.coil_inductance_h
        self.coil_current_a += current_slope * step_s
        self.coil_elapsed_s += step_s
        self.coil_history.append(
            (
                self.coil_elapsed_s,
                voltage,
                self.coil_current_a,
                self.coil_field_per_amp_t * self.coil_current_a,
            )
        )
        if len(self.coil_history) > 240:
            del self.coil_history[:-240]

    def _draw_coil_page(self) -> None:
        canvas = self.coil_canvas
        canvas.delete("all")
        focus = self.focus_get()
        if focus is not self.coil_voltage_entry:
            self.coil_voltage_input.set(f"{self.coil_voltage_v.get():.1f}")

        voltage = self.coil_voltage_v.get()
        current = self.coil_current_a
        magnetic_field_t = self.coil_field_per_amp_t * current
        canvas.create_text(25, 25, text="Spule mit Batterie", anchor="w", font=("Segoe UI", 18, "bold"), fill="#0f172a")
        canvas.create_text(
            25,
            55,
            text=f"U_bat = {voltage:+.1f} V   |   i = {current:+.2f} A   |   B = {magnetic_field_t:+.3f} T",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            fill="#166534",
        )
        self._draw_coil_trace_panel(
            left=25,
            top=95,
            width=610,
            height=195,
            title="Strom i durch die Spule",
            unit="A",
            values=[sample[2] for sample in self.coil_history],
            axis_limit=10.0,
            colour="#d97706",
        )
        self._draw_coil_trace_panel(
            left=25,
            top=310,
            width=610,
            height=195,
            title="Magnetfeld B in der Spule",
            unit="T",
            values=[sample[3] for sample in self.coil_history],
            axis_limit=0.5,
            colour="#16a34a",
        )
        self._draw_coil_trace_panel(
            left=25,
            top=525,
            width=610,
            height=90,
            title="Batteriespannung U_bat",
            unit="V",
            values=[sample[1] for sample in self.coil_history],
            axis_limit=10.0,
            colour="#dc2626",
        )
        self._draw_coil_visual(voltage, current, magnetic_field_t)
        self._draw_coil_esb(voltage, current)
        self._scale_canvas_to_view(canvas)

    def _draw_coil_trace_panel(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        title: str,
        unit: str,
        values: list[float],
        axis_limit: float,
        colour: str,
    ) -> None:
        """Eine einzelne Verlaufskurve im selben Kartenraster wie iabc und uabc."""
        canvas = self.coil_canvas
        canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        canvas.create_text(left + 12, top + 13, text=title, anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        plot_left, plot_right = left + 55, left + width - 18
        plot_top, plot_bottom = top + 32, top + height - 18
        centre_y = (plot_top + plot_bottom) / 2.0
        plot_height = plot_bottom - plot_top
        canvas.create_line(plot_left, centre_y, plot_right, centre_y, fill="#94a3b8", dash=(3, 3))
        canvas.create_text(left + 8, plot_top, text=f"+{axis_limit:g}", anchor="w", font=("Segoe UI", 9), fill="#64748b")
        canvas.create_text(left + 8, plot_bottom, text=f"-{axis_limit:g}", anchor="w", font=("Segoe UI", 9), fill="#64748b")
        canvas.create_text(left + width - 18, top + height - 7, text=unit, anchor="e", font=("Segoe UI", 9), fill="#64748b")

        if not values:
            return
        points: list[float] = []
        number_of_values = max(1, len(values) - 1)
        for point_index, value in enumerate(values):
            x = plot_left + (plot_right - plot_left) * point_index / number_of_values
            clipped_value = min(axis_limit, max(-axis_limit, value))
            y = centre_y - 0.43 * plot_height * clipped_value / axis_limit
            points.extend((x, y))
        if len(points) == 2:
            points.extend(points)
        canvas.create_line(*points, fill=colour, width=2, smooth=True)
        marker_y = points[-1]
        canvas.create_oval(plot_right - 4, marker_y - 4, plot_right + 4, marker_y + 4, outline=colour, fill=colour)
        canvas.create_text(
            plot_right - 4,
            top + 14,
            text=f"{values[-1]:+.3g} {unit}",
            anchor="e",
            font=("Segoe UI", 9, "bold"),
            fill=colour,
        )

    def _draw_coil_visual(self, voltage: float, current: float, magnetic_field_t: float) -> None:
        """Große physikalische Einzelspule an der Position der Motoransicht."""
        canvas = self.coil_canvas
        left, top, right, bottom = 655, 75, 1025, 475
        canvas.create_rectangle(left, top, right, bottom, outline="#cbd5e1", fill="#ffffff")
        canvas.create_text((left + right) / 2, top + 20, text="Einzelspule", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_text(
            (left + right) / 2,
            top + 42,
            text="Strom erzeugt ein Magnetfeld",
            font=("Segoe UI", 9),
            fill="#475569",
        )

        coil_left, coil_right, coil_y = 710, 970, 276
        winding_top, winding_bottom = 212, 340
        former_left, former_right, former_top, former_bottom = 722, 958, 229, 323
        # Neutraler Wicklungstraeger: Die Darstellung behauptet bewusst
        # keinen Eisenkern und bleibt damit fuer die RL-Lehrspule allgemein.
        canvas.create_rectangle(former_left, former_top, former_right, former_bottom, outline="#64748b", width=2, fill="#e2e8f0")
        canvas.create_rectangle(former_left, coil_y - 28, former_right, coil_y + 28, outline="#94a3b8", fill="#f8fafc")
        for flange_x in (coil_left, coil_right - 14):
            canvas.create_rectangle(flange_x, winding_top, flange_x + 14, winding_bottom, outline="#475569", width=2, fill="#cbd5e1")
        for turn_index in range(9):
            turn_x = coil_left + 20 + turn_index * 28
            canvas.create_oval(turn_x - 11, coil_y - 64, turn_x + 11, coil_y + 64, outline="#9a3412", width=5)
            canvas.create_oval(turn_x - 11, coil_y - 64, turn_x + 11, coil_y + 64, outline="#fb923c", width=2)
        canvas.create_line(687, winding_top, coil_left, winding_top, fill="#c2410c", width=3)
        canvas.create_line(687, winding_bottom, coil_left, winding_bottom, fill="#c2410c", width=3)
        for terminal_y, terminal_name, label_y in ((winding_top, "A1", winding_top - 14), (winding_bottom, "A2", winding_bottom + 14)):
            canvas.create_oval(676, terminal_y - 5, 686, terminal_y + 5, outline="#c2410c", width=2, fill="#ffffff")
            canvas.create_text(681, label_y, text=terminal_name, font=("Segoe UI", 8, "bold"), fill="#9a3412")

        current_scale = min(abs(current) / 10.0, 1.0)
        arrow_length = 12.0 + 112.0 * current_scale
        arrow_y, arrow_centre_x = 154, 830
        if abs(current) < 0.02:
            canvas.create_line(arrow_centre_x - 6, arrow_y, arrow_centre_x + 6, arrow_y, fill="#64748b", width=2)
        else:
            sign = 1.0 if current > 0.0 else -1.0
            canvas.create_line(
                arrow_centre_x - sign * arrow_length / 2.0,
                arrow_y,
                arrow_centre_x + sign * arrow_length / 2.0,
                arrow_y,
                fill="#d97706",
                width=4,
                arrow=tk.LAST,
            )
        canvas.create_text(830, 178, text=f"i = {current:+.2f} A", font=("Segoe UI", 11, "bold"), fill="#b45309")

        field_scale = min(abs(magnetic_field_t) / 0.5, 1.0)
        field_length = 18.0 + 205.0 * field_scale
        field_centre_x = (coil_left + coil_right) / 2.0
        if abs(magnetic_field_t) < 0.002:
            canvas.create_line(field_centre_x - 8, coil_y, field_centre_x + 8, coil_y, fill="#64748b", width=2)
        else:
            field_sign = 1.0 if magnetic_field_t > 0.0 else -1.0
            canvas.create_line(
                field_centre_x - field_sign * field_length / 2.0,
                coil_y,
                field_centre_x + field_sign * field_length / 2.0,
                coil_y,
                fill="#16a34a",
                width=max(3, round(3 + 5 * field_scale)),
                arrow=tk.LAST,
            )
            left_pole, right_pole = ("S", "N") if field_sign > 0.0 else ("N", "S")
            pole_colours = {"N": "#dc2626", "S": "#2563eb"}
            for pole_name, pole_x in ((left_pole, coil_left - 18), (right_pole, coil_right + 18)):
                canvas.create_oval(pole_x - 15, coil_y - 15, pole_x + 15, coil_y + 15, outline="#334155", fill=pole_colours[pole_name])
                canvas.create_text(pole_x, coil_y, text=pole_name, font=("Segoe UI", 12, "bold"), fill="white")
        canvas.create_text(830, 384, text=f"B = {magnetic_field_t:+.3f} T", font=("Segoe UI", 12, "bold"), fill="#15803d")

    def _draw_coil_esb(self, voltage: float, current: float) -> None:
        """Kleines, bewusst einfaches Ersatzschaltbild im Motor-ESB-Raster."""
        canvas = self.coil_canvas
        left, top, right, bottom = 655, 493, 1025, 646
        canvas.create_rectangle(left, top, right, bottom, outline="#94a3b8", fill="#ffffff")
        canvas.create_text(left + 12, top + 13, text="Einfaches ESB", anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        canvas.create_text(right - 12, top + 13, text="Batterie - R - L", anchor="e", font=("Segoe UI", 9), fill="#475569")

        top_y, bottom_y = 548, 603
        battery_x, resistor_left, resistor_right = 700, 765, 824
        coil_left, coil_right, return_x = 856, 937, 978
        wire_colour = "#334155"
        canvas.create_line(battery_x, top_y, resistor_left, top_y, fill=wire_colour, width=2)
        canvas.create_line(resistor_right, top_y, coil_left, top_y, fill=wire_colour, width=2)
        canvas.create_line(coil_right, top_y, return_x, top_y, fill=wire_colour, width=2)
        canvas.create_line(return_x, top_y, return_x, bottom_y, fill=wire_colour, width=2)
        canvas.create_line(return_x, bottom_y, battery_x, bottom_y, fill=wire_colour, width=2)
        canvas.create_line(battery_x, bottom_y, battery_x, 574, fill=wire_colour, width=2)
        canvas.create_line(battery_x, 556, battery_x, top_y, fill=wire_colour, width=2)

        positive_at_top = voltage >= 0.0
        top_mark, bottom_mark = ("+", "-") if positive_at_top else ("-", "+")
        top_colour = "#dc2626" if positive_at_top else "#2563eb"
        bottom_colour = "#2563eb" if positive_at_top else "#dc2626"
        canvas.create_line(battery_x - 16, 556, battery_x + 16, 556, fill=top_colour, width=3)
        canvas.create_line(battery_x - 10, 574, battery_x + 10, 574, fill=bottom_colour, width=3)
        canvas.create_text(battery_x - 28, 556, text=top_mark, font=("Segoe UI", 11, "bold"), fill=top_colour)
        canvas.create_text(battery_x - 28, 574, text=bottom_mark, font=("Segoe UI", 11, "bold"), fill=bottom_colour)
        canvas.create_text(battery_x, 622, text=f"U_bat {voltage:+.1f} V", font=("Segoe UI", 8, "bold"), fill="#0f172a")

        resistor_points: list[float] = [resistor_left, top_y]
        for point_index in range(1, 7):
            x = resistor_left + point_index * (resistor_right - resistor_left) / 7.0
            y = top_y - 8 if point_index % 2 else top_y + 8
            resistor_points.extend((x, y))
        # Der letzte Punkt liegt wieder exakt auf der Leiter-Mittellinie.
        # So schliesst der Widerstand ohne sichtbaren Versatz an die Leitung an.
        resistor_points.extend((resistor_right, top_y))
        canvas.create_line(*resistor_points, fill="#475569", width=2)
        canvas.create_text((resistor_left + resistor_right) / 2, 531, text="R", font=("Segoe UI", 10, "bold"), fill="#475569")

        coil_points: list[float] = []
        for point_index in range(33):
            progress = point_index / 32.0
            x = coil_left + (coil_right - coil_left) * progress
            y = top_y - 11.0 * math.sin(4.0 * 2.0 * math.pi * progress)
            coil_points.extend((x, y))
        canvas.create_line(*coil_points, fill="#7c3aed", width=2, smooth=True, splinesteps=12)
        canvas.create_text((coil_left + coil_right) / 2, 531, text="L", font=("Segoe UI", 10, "bold"), fill="#6d28d9")

        current_scale = min(abs(current) / 10.0, 1.0)
        arrow_length = 10.0 + 36.0 * current_scale
        arrow_centre_x = 736
        if abs(current) < 0.02:
            canvas.create_line(arrow_centre_x - 5, 531, arrow_centre_x + 5, 531, fill="#64748b", width=2)
        else:
            sign = 1.0 if current > 0.0 else -1.0
            canvas.create_line(
                arrow_centre_x - sign * arrow_length / 2.0,
                531,
                arrow_centre_x + sign * arrow_length / 2.0,
                531,
                fill="#d97706",
                width=3,
                arrow=tk.LAST,
            )
        canvas.create_text(736, 517, text=f"i {current:+.2f} A", font=("Segoe UI", 8, "bold"), fill="#b45309")

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Eingaben", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Drehzahl").grid(row=0, column=0, sticky="w")
        speed_slider = ttk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.speed_rpm,
            command=lambda _value: self._draw(),
        )
        speed_slider.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.speed_entry = ttk.Entry(controls, width=13, justify="right", textvariable=self.speed_input)
        self.speed_entry.grid(row=0, column=2, sticky="e")
        ttk.Label(controls, text="rpm").grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(controls, text="Stromamplitude I Spitze").grid(row=0, column=4, sticky="w", padx=(24, 0))
        amplitude_slider = ttk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.current_amplitude_a,
            command=lambda _value: self._draw(),
        )
        amplitude_slider.grid(row=0, column=5, sticky="ew", padx=(8, 8))
        controls.columnconfigure(5, weight=1)
        self.amplitude_entry = ttk.Entry(controls, width=10, justify="right", textvariable=self.amplitude_input)
        self.amplitude_entry.grid(row=0, column=6, sticky="e")
        ttk.Label(controls, text="A").grid(row=0, column=7, sticky="w", padx=(4, 0))

        self.start_stop_button = ttk.Button(controls, command=self._toggle_animation)
        self.start_stop_button.grid(row=0, column=8, padx=(22, 0))

        ttk.Label(controls, text="Stromwinkel phi zu Rotor").grid(row=1, column=0, sticky="w", pady=(9, 0))
        phase_slider = ttk.Scale(
            controls,
            from_=-180,
            to=180,
            orient="horizontal",
            variable=self.current_phase_deg,
            command=lambda _value: self._draw(),
        )
        phase_slider.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(9, 0))
        self.phase_entry = ttk.Entry(controls, width=13, justify="right", textvariable=self.phase_input)
        self.phase_entry.grid(row=1, column=2, sticky="e", pady=(9, 0))
        ttk.Label(controls, text="Grad").grid(row=1, column=3, sticky="w", padx=(4, 0), pady=(9, 0))
        ttk.Label(
            controls,
            text="0 Grad: Feld und Rotor ausgerichtet | +90 Grad: maximales positives Moment",
            foreground="#475569",
        ).grid(row=1, column=4, columnspan=5, sticky="w", padx=(24, 0), pady=(9, 0))
        ttk.Label(controls, text="Leertaste: Start/Stopp", foreground="#475569").grid(
            row=2, column=0, columnspan=9, sticky="w", pady=(8, 0)
        )

        self._bind_numeric_entry(self.speed_entry, self.speed_input, self.speed_rpm, 0.0, 100.0)
        self._bind_numeric_entry(self.amplitude_entry, self.amplitude_input, self.current_amplitude_a, 0.0, 100.0)
        self._bind_numeric_entry(self.phase_entry, self.phase_input, self.current_phase_deg, -180.0, 180.0)

    def _bind_numeric_entry(
        self,
        entry: ttk.Entry,
        input_value: tk.StringVar,
        target_value: tk.DoubleVar,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        """Verbindet ein Freitextfeld sicher mit einem Slider-Wert."""

        def apply_value(_event: tk.Event | None = None) -> None:
            try:
                value = float(input_value.get().replace(",", "."))
            except ValueError:
                value = target_value.get()
            value = min(upper_bound, max(lower_bound, value))
            target_value.set(value)
            input_value.set(f"{value:.0f}")
            self._draw()

        entry.bind("<Return>", apply_value)
        entry.bind("<FocusOut>", apply_value)

    def _sync_numeric_inputs(self) -> None:
        """Aktualisiert Textfelder nach Slider-Bewegung, ohne Eingaben zu ueberschreiben."""
        focus = self.focus_get()
        for entry, input_value, numeric_value in (
            (self.speed_entry, self.speed_input, self.speed_rpm),
            (self.amplitude_entry, self.amplitude_input, self.current_amplitude_a),
            (self.phase_entry, self.phase_input, self.current_phase_deg),
        ):
            if focus is not entry:
                input_value.set(f"{numeric_value.get():.0f}")

    def _values(self) -> MachineValues:
        """Idealisierte SPMSM: Te = 3/2 * p * psi_PM * iq."""
        current_peak = self.current_amplitude_a.get()
        iq_peak = current_peak * math.sin(math.radians(self.current_phase_deg.get()))
        # Bewusst einfache Spannungsdarstellung: Die Stromamplitude bestimmt
        # direkt die sichtbare Spannungsamplitude (100 A entsprechen 50 V).
        voltage_peak = 0.5 * current_peak
        electrical_frequency = self.pole_pairs * self.speed_rpm.get() / 60.0
        torque_nm = 1.5 * self.pole_pairs * self.flux_linkage_wb * iq_peak
        stator_field_peak_t = self.stator_field_per_amp_t * current_peak
        return MachineValues(current_peak, voltage_peak, electrical_frequency, iq_peak, torque_nm, stator_field_peak_t)

    def _current_angle(self) -> float:
        """Elektrischer Winkel des Stromsystems relativ zum Rotorwinkel."""
        return self.electrical_angle + math.radians(self.current_phase_deg.get())

    def _toggle_animation(self) -> None:
        self.running = not self.running
        self.last_time_ms = self.tk.call("clock", "milliseconds")
        self._draw()
        self._draw_coil_page()
        self._draw_motor_test_page()

    def _animate(self) -> None:
        now_ms = self.tk.call("clock", "milliseconds")
        elapsed_s = max(0.0, (now_ms - self.last_time_ms) / 1000.0)
        self.last_time_ms = now_ms
        if self.running:
            # Die Animation wird bei hohen Drehzahlen auf eine gut sichtbare
            # Geschwindigkeit begrenzt; der Messwert bleibt der echte Sliderwert.
            shown_frequency = min(abs(self._values().electrical_frequency), 8.0)
            direction = 1.0 if self.speed_rpm.get() >= 0 else -1.0
            self.electrical_angle = (self.electrical_angle + direction * 2.0 * math.pi * shown_frequency * elapsed_s) % (
                2.0 * math.pi
            )
        self._update_coil_current(elapsed_s)
        self._update_test_rotor(elapsed_s)
        self._draw()
        self._draw_coil_page()
        self._draw_motor_test_page()
        self.after(self.update_ms, self._animate)

    def _draw(self) -> None:
        self.canvas.delete("all")
        values = self._values()
        self._sync_numeric_inputs()
        self.start_stop_button.config(text="Stopp" if self.running else "Start")

        self._draw_header(values)
        self._draw_wave_panel(
            left=25,
            top=95,
            width=610,
            height=195,
            title="Phasenströme iabc",
            unit="A",
            amplitude=values.current_peak,
            axis_limit=100.0,
        )
        self._draw_wave_panel(
            left=25,
            top=310,
            width=610,
            height=195,
            title="Phasenspannungen uabc",
            unit="V",
            amplitude=values.voltage_peak,
            axis_limit=50.0,
        )
        self._draw_rotor_angle_panel(left=25, top=525, width=610, height=90)
        self._draw_machine(values)
        self._draw_esb(values)
        self._scale_canvas_to_view()

    def _scale_canvas_to_view(self, canvas: tk.Canvas | None = None) -> None:
        """Skaliert die feste Lehrgrafik passend zur aktuellen Fensterflaeche."""
        target_canvas = self.canvas if canvas is None else canvas
        view_width = target_canvas.winfo_width()
        view_height = target_canvas.winfo_height()
        if view_width <= 1 or view_height <= 1:
            return

        scale = min(view_width / self.canvas_width, view_height / self.canvas_height)
        offset_x = (view_width - self.canvas_width * scale) / 2.0
        offset_y = (view_height - self.canvas_height * scale) / 2.0
        target_canvas.scale("all", 0.0, 0.0, scale, scale)
        target_canvas.move("all", offset_x, offset_y)

        # Canvas skaliert Positionen, aber keine Schriftgroessen oder
        # Linienbreiten. Beides wird deshalb proportional nachgezogen.
        for item_id in target_canvas.find_all():
            item_type = target_canvas.type(item_id)
            if item_type == "text":
                try:
                    font_info = tkfont.Font(font=target_canvas.itemcget(item_id, "font")).actual()
                    font_style: list[str | int] = [font_info["family"], max(7, round(abs(font_info["size"]) * scale))]
                    if font_info["weight"] == "bold":
                        font_style.append("bold")
                    if font_info["slant"] == "italic":
                        font_style.append("italic")
                    target_canvas.itemconfigure(item_id, font=tuple(font_style))
                except tk.TclError:
                    pass
            elif item_type in {"line", "rectangle", "oval", "polygon", "arc"}:
                try:
                    line_width = float(target_canvas.itemcget(item_id, "width"))
                    if line_width > 0.0:
                        target_canvas.itemconfigure(item_id, width=max(1.0, line_width * scale))
                except (tk.TclError, ValueError):
                    pass

    def _draw_header(self, values: MachineValues) -> None:
        self.canvas.create_text(
            25,
            25,
            anchor="w",
            text="Dreiphasensystem mit rotierendem Magnetfeld",
            font=("Segoe UI", 18, "bold"),
            fill="#0f172a",
        )
        self.canvas.create_text(
            25,
            55,
            anchor="w",
            text=f"Drehmoment T_e = {values.torque_nm:+.2f} Nm",
            font=("Segoe UI", 13, "bold"),
            fill="#166534",
        )

    def _draw_wave_panel(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        title: str,
        unit: str,
        amplitude: float,
        axis_limit: float,
    ) -> None:
        self.canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        self.canvas.create_text(left + 12, top + 13, text=title, anchor="w", font=("Segoe UI", 12, "bold"), fill="#0f172a")
        plot_left, plot_right = left + 55, left + width - 18
        plot_top, plot_bottom = top + 42, top + height - 28
        centre_y = (plot_top + plot_bottom) / 2
        plot_height = plot_bottom - plot_top

        self.canvas.create_line(plot_left, centre_y, plot_right, centre_y, fill="#94a3b8", dash=(3, 3))
        self.canvas.create_text(left + 9, plot_top, text=f"+{axis_limit:.0f}", anchor="w", font=("Segoe UI", 9), fill="#64748b")
        self.canvas.create_text(left + 9, plot_bottom, text=f"−{axis_limit:.0f}", anchor="w", font=("Segoe UI", 9), fill="#64748b")
        self.canvas.create_text(left + width - 18, top + height - 9, text=unit, anchor="e", font=("Segoe UI", 9), fill="#64748b")

        waveform_scale = min(amplitude / axis_limit, 1.0)
        for phase_offset, colour, phase in zip(PHASE_OFFSETS, PHASE_COLOURS, PHASES):
            points: list[float] = []
            for pixel_x in range(plot_left, plot_right + 1, 3):
                history_angle = self._current_angle() - 2.0 * math.pi * (plot_right - pixel_x) / (plot_right - plot_left)
                y = centre_y - 0.43 * plot_height * waveform_scale * math.cos(history_angle + phase_offset)
                points.extend((pixel_x, y))
            self.canvas.create_line(*points, fill=colour, width=2, smooth=True)
            instantaneous = amplitude * math.cos(self._current_angle() + phase_offset)
            y = centre_y - 0.43 * plot_height * waveform_scale * math.cos(self._current_angle() + phase_offset)
            self.canvas.create_oval(plot_right - 4, y - 4, plot_right + 4, y + 4, outline=colour, fill=colour)
            self.canvas.create_text(
                plot_right - 2,
                plot_top + 12 + PHASES.index(phase) * 18,
                text=f"{phase}: {instantaneous:+.1f} {unit}",
                anchor="e",
                font=("Segoe UI", 9),
                fill=colour,
            )

    def _draw_machine(self, values: MachineValues) -> None:
        centre_x, centre_y = 830, 280
        outer_radius, rotor_radius = 205, 84
        self.canvas.create_text(
            centre_x,
            66,
            text="Dreiphasen-Maschine",
            font=("Segoe UI", 12, "bold"),
            fill="#0f172a",
        )
        self.canvas.create_oval(
            centre_x - outer_radius,
            centre_y - outer_radius,
            centre_x + outer_radius,
            centre_y + outer_radius,
            outline="#334155",
            width=3,
            fill="#e2e8f0",
        )

        # Genau drei einfache Spulensymbole A/B/C. Der Pfeil daneben ist
        # jeweils der momentane Phasenstrom.
        for phase_index, phase_angle_deg in enumerate((0.0, 120.0, 240.0)):
            stator_angle = math.radians(phase_angle_deg)
            x = centre_x + 157 * math.cos(stator_angle)
            y = centre_y - 157 * math.sin(stator_angle)
            current = values.current_peak * math.cos(self._current_angle() + PHASE_OFFSETS[phase_index])
            colour = PHASE_COLOURS[phase_index]

            # Eine einzelne, einfache Spule: ihre Enden liegen radial, damit
            # ein Ende klar zum Luftspalt und eines nach aussen zeigt.
            radial_x = math.cos(stator_angle)
            radial_y = -math.sin(stator_angle)
            tangent_x = -math.sin(stator_angle)
            tangent_y = -math.cos(stator_angle)
            coil_points: list[float] = []
            coil_half_width = 24.0
            for point_index in range(25):
                progress = point_index / 24.0
                along_coil = -coil_half_width + 2.0 * coil_half_width * progress
                coil_offset = 8.0 * math.sin(3.0 * 2.0 * math.pi * progress)
                coil_points.extend(
                    (
                        x + along_coil * radial_x + coil_offset * tangent_x,
                        y + along_coil * radial_y + coil_offset * tangent_y,
                    )
                )
            self.canvas.create_line(*coil_points, fill=colour, width=3, smooth=True, splinesteps=12)
            self.canvas.create_line(
                x - 33 * radial_x,
                y - 33 * radial_y,
                x - 24 * radial_x,
                y - 24 * radial_y,
                fill=colour,
                width=3,
            )
            self.canvas.create_line(
                x + 24 * radial_x,
                y + 24 * radial_y,
                x + 33 * radial_x,
                y + 33 * radial_y,
                fill=colour,
                width=3,
            )

            # Ein klarer Phasenstromzeiger neben der Wicklung. Er ist eine
            # Referenzrichtung fuer i_a, i_b bzw. i_c (wie im ESB), nicht
            # eine separate Leiterseiten-Darstellung.
            relative_current = abs(current) / max(values.current_peak, 1.0)
            arrow_length = 10.0 + 46.0 * relative_current
            arrow_centre = (x - 25.0 * tangent_x, y - 25.0 * tangent_y)
            if abs(current) < 0.05:
                self.canvas.create_line(
                    arrow_centre[0] - 5 * radial_x,
                    arrow_centre[1] - 5 * radial_y,
                    arrow_centre[0] + 5 * radial_x,
                    arrow_centre[1] + 5 * radial_y,
                    fill="#64748b",
                    width=2,
                )
            else:
                current_direction = 1.0 if current > 0.0 else -1.0
                arrow_tail_x = arrow_centre[0] - current_direction * arrow_length / 2.0 * radial_x
                arrow_tail_y = arrow_centre[1] - current_direction * arrow_length / 2.0 * radial_y
                arrow_head_x = arrow_centre[0] + current_direction * arrow_length / 2.0 * radial_x
                arrow_head_y = arrow_centre[1] + current_direction * arrow_length / 2.0 * radial_y
                self.canvas.create_line(arrow_tail_x, arrow_tail_y, arrow_head_x, arrow_head_y, fill=colour, width=3, arrow=tk.LAST)

            label_x = centre_x + 104 * radial_x
            label_y = centre_y + 104 * radial_y
            self.canvas.create_text(
                label_x,
                label_y,
                text=f"{PHASES[phase_index]}  {current:+.0f} A",
                font=("Segoe UI", 9, "bold"),
                fill=colour,
            )

        # Statorfeldachse im Luftspalt. Der Richtungsvektor B zeigt von
        # Nord- nach Suedpol; deshalb liegt N am negativen und S am positiven
        # Ende der Achse. Die gestrichelte Linie erleichtert das Ablesen des
        # Stromwinkelversatzes zum Rotor.
        stator_field_x = math.cos(self._current_angle())
        stator_field_y = -math.sin(self._current_angle())
        stator_perpendicular_x = -stator_field_y
        stator_perpendicular_y = stator_field_x
        field_axis_radius = 148.0
        self.canvas.create_line(
            centre_x - field_axis_radius * stator_field_x,
            centre_y - field_axis_radius * stator_field_y,
            centre_x + field_axis_radius * stator_field_x,
            centre_y + field_axis_radius * stator_field_y,
            fill="#94a3b8",
            width=1,
            dash=(4, 4),
        )
        field_strength = min(values.current_peak / 100.0, 1.0)
        if field_strength > 0.001:
            # Die Polschuhe wachsen mit der Stromamplitude. Bei I = 0 gibt es
            # kein eingeprägtes Statorfeld; bei I = 100 A sind sie maximal.
            pole_half_radial = 6.0 + 18.0 * field_strength
            pole_half_lateral = 8.0 + 34.0 * field_strength
            for pole_name, pole_sign, pole_colour in (("N", -1.0, "#dc2626"), ("S", 1.0, "#2563eb")):
                pole_centre_x = centre_x + pole_sign * 122.0 * stator_field_x
                pole_centre_y = centre_y + pole_sign * 122.0 * stator_field_y
                pole_corners: list[float] = []
                for radial_sign, lateral_sign in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)):
                    pole_corners.extend(
                        (
                            pole_centre_x + radial_sign * pole_half_radial * stator_field_x + lateral_sign * pole_half_lateral * stator_perpendicular_x,
                            pole_centre_y + radial_sign * pole_half_radial * stator_field_y + lateral_sign * pole_half_lateral * stator_perpendicular_y,
                        )
                    )
                self.canvas.create_polygon(*pole_corners, fill=pole_colour, outline="#334155", width=1)
                self.canvas.create_text(
                    pole_centre_x,
                    pole_centre_y,
                    text=pole_name,
                    font=("Segoe UI", max(7, round(8 + 6 * field_strength)), "bold"),
                    fill="white",
                )

        # Mathematische Standardkonvention auf dem Canvas: theta = 0 Grad
        # zeigt nach +x; steigender Winkel dreht gegen den Uhrzeigersinn.
        # Das Minus bei y kompensiert die nach unten wachsende Canvas-y-Achse.
        direction_x = math.cos(self.electrical_angle)
        direction_y = -math.sin(self.electrical_angle)
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        magnet_half_length, magnet_half_width = 72, 32
        corners = []
        for longitudinal, lateral in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            corners.extend(
                (
                    centre_x + longitudinal * magnet_half_length * direction_x + lateral * magnet_half_width * perpendicular_x,
                    centre_y + longitudinal * magnet_half_length * direction_y + lateral * magnet_half_width * perpendicular_y,
                )
            )
        self.canvas.create_polygon(*corners, fill="#fef3c7", outline="#334155", width=2)
        self.canvas.create_line(
            centre_x,
            centre_y,
            centre_x + magnet_half_length * direction_x,
            centre_y + magnet_half_length * direction_y,
            fill="#dc2626",
            width=magnet_half_width * 2,
        )
        self.canvas.create_line(
            centre_x,
            centre_y,
            centre_x - magnet_half_length * direction_x,
            centre_y - magnet_half_length * direction_y,
            fill="#2563eb",
            width=magnet_half_width * 2,
        )
        self.canvas.create_text(
            centre_x + 43 * direction_x,
            centre_y + 43 * direction_y,
            text="N",
            font=("Segoe UI", 13, "bold"),
            fill="white",
        )
        self.canvas.create_text(
            centre_x - 43 * direction_x,
            centre_y - 43 * direction_y,
            text="S",
            font=("Segoe UI", 13, "bold"),
            fill="white",
        )
        self.canvas.create_oval(centre_x - rotor_radius, centre_y - rotor_radius, centre_x + rotor_radius, centre_y + rotor_radius, outline="#334155", width=2)

        # Schwarze, am Rotor feste Referenz: RLG-Nullwinkel. Sie zeigt die
        # Nullposition des Rotorlagegebers und dreht mit dem Rotor mit.
        rlg_radius = 116
        rlg_x = centre_x + rlg_radius * direction_x
        rlg_y = centre_y + rlg_radius * direction_y
        self.canvas.create_line(centre_x, centre_y, rlg_x, rlg_y, fill="#000000", width=2, arrow=tk.LAST)
        self.canvas.create_text(
            centre_x + 142 * direction_x,
            centre_y + 142 * direction_y,
            text="RLG 0°",
            font=("Segoe UI", 9, "bold"),
            fill="#000000",
        )

    def _draw_rotor_angle_panel(self, *, left: int, top: int, width: int, height: int) -> None:
        """Zeichnet den mechanischen Rotorwinkel unter iabc und uabc."""
        self.canvas.create_rectangle(left, top, left + width, top + height, outline="#cbd5e1", fill="#ffffff")
        mechanical_angle_deg = math.degrees(self.electrical_angle / self.pole_pairs) % 360.0
        self.canvas.create_text(
            left + 12,
            top + 13,
            text="Rotorwinkel",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            fill="#0f172a",
        )
        self.canvas.create_text(
            left + width - 12,
            top + 13,
            text=f"{mechanical_angle_deg:6.1f} Grad",
            anchor="e",
            font=("Segoe UI", 9),
            fill="#b45309",
        )

        plot_left, plot_right = left + 55, left + width - 18
        plot_top, plot_bottom = top + 30, top + height - 13
        self.canvas.create_line(plot_left, plot_bottom, plot_right, plot_bottom, fill="#94a3b8", dash=(3, 3))
        self.canvas.create_text(left + 8, plot_top, text="360", anchor="w", font=("Segoe UI", 8), fill="#64748b")
        self.canvas.create_text(left + 18, plot_bottom, text="0", anchor="w", font=("Segoe UI", 8), fill="#64748b")

        points: list[float] = []
        for pixel_x in range(plot_left, plot_right + 1, 3):
            history_electrical_angle = self.electrical_angle - 2.0 * math.pi * (plot_right - pixel_x) / (plot_right - plot_left)
            history_mechanical_deg = math.degrees(history_electrical_angle / self.pole_pairs) % 360.0
            y = plot_bottom - (plot_bottom - plot_top) * history_mechanical_deg / 360.0
            points.extend((pixel_x, y))
        self.canvas.create_line(*points, fill="#d97706", width=2, smooth=False)
        marker_y = plot_bottom - (plot_bottom - plot_top) * mechanical_angle_deg / 360.0
        self.canvas.create_oval(plot_right - 4, marker_y - 4, plot_right + 4, marker_y + 4, outline="#d97706", fill="#d97706")

    def _draw_esb(self, values: MachineValues) -> None:
        """Vereinfachtes ESB: Steuerung, Leistungspfad und Rueckfuehrungen."""
        left, top, right, bottom = 655, 493, 1025, 646
        self.canvas.create_rectangle(left, top, right, bottom, outline="#94a3b8", fill="#ffffff")
        # Steuerung: ein klarer PWM-Pfad zum Inverter sowie zwei eigene
        # Rueckfuehrungen fuer Strommessung und Rotorlage.
        micro_left, micro_right = 670, 738
        self.canvas.create_rectangle(micro_left, 522, micro_right, 575, outline="#2563eb", width=2, fill="#eff6ff")
        self.canvas.create_text((micro_left + micro_right) / 2, 540, text="µC", font=("Segoe UI", 12, "bold"), fill="#1d4ed8")
        self.canvas.create_text((micro_left + micro_right) / 2, 555, text="PWM", font=("Segoe UI", 8, "bold"), fill="#1d4ed8")

        # Drei getrennte Halbbruecken mit gemeinsamem Zwischenkreis.
        inverter_left, inverter_right = 775, 871
        inverter_top, inverter_bottom = 515, 581
        self.canvas.create_rectangle(inverter_left, inverter_top, inverter_right, inverter_bottom, outline="#7c3aed", width=2, fill="#f5f3ff")
        self.canvas.create_line(micro_right, 548, inverter_left, 548, fill="#2563eb", width=2, arrow=tk.LAST)
        self.canvas.create_text((micro_right + inverter_left) / 2, 539, text="PWM", font=("Segoe UI", 7), fill="#2563eb")

        # Die Batterie speist die beiden Zwischenkreis-Schienen. Das macht
        # die Spannungsquelle ohne schwer lesbare Udc-Beschriftung sichtbar.
        battery_x = 758
        rail_left, rail_right = battery_x, 863
        plus_rail_y, minus_rail_y = 521, 576
        self.canvas.create_line(rail_left, plus_rail_y, rail_right, plus_rail_y, fill="#dc2626", width=2)
        self.canvas.create_line(rail_left, minus_rail_y, rail_right, minus_rail_y, fill="#2563eb", width=2)
        self.canvas.create_line(battery_x, plus_rail_y, battery_x, 533, fill="#dc2626", width=2)
        self.canvas.create_line(battery_x - 14, 533, battery_x + 14, 533, fill="#dc2626", width=3)
        self.canvas.create_line(battery_x - 8, 563, battery_x + 8, 563, fill="#2563eb", width=3)
        self.canvas.create_line(battery_x, 563, battery_x, minus_rail_y, fill="#2563eb", width=2)
        self.canvas.create_text(battery_x - 20, 533, text="+", font=("Segoe UI", 10, "bold"), fill="#dc2626")
        self.canvas.create_text(battery_x - 20, 563, text="−", font=("Segoe UI", 12, "bold"), fill="#2563eb")

        # Der mittlere Knoten jeder Halbbruecke ist der zugehoerige
        # Phasenausgang. HS/LS werden passend zum Vorzeichen von uabc markiert.
        phase_y_values = (535, 550, 565)
        leg_x_values = (799, 827, 855)
        sensor_centres: list[tuple[float, float]] = []
        for phase, phase_offset, phase_y, leg_x, colour in zip(("u", "v", "w"), PHASE_OFFSETS, phase_y_values, leg_x_values, PHASE_COLOURS):
            phase_voltage = values.voltage_peak * math.cos(self._current_angle() + phase_offset)
            high_side_active = phase_voltage >= 0.0
            hs_fill = "#16a34a" if high_side_active else "#ffffff"
            ls_fill = "#dc2626" if not high_side_active else "#ffffff"
            hs_text_colour = "#ffffff" if high_side_active else "#166534"
            ls_text_colour = "#ffffff" if not high_side_active else "#991b1b"
            switch_half_height = 4
            hs_centre_y = phase_y - 6
            ls_centre_y = phase_y + 6
            self.canvas.create_line(leg_x, plus_rail_y, leg_x, hs_centre_y - switch_half_height, fill="#475569", width=1)
            self.canvas.create_rectangle(leg_x - 8, hs_centre_y - switch_half_height, leg_x + 8, hs_centre_y + switch_half_height, outline="#166534", fill=hs_fill)
            self.canvas.create_text(leg_x, hs_centre_y, text="HS", font=("Segoe UI", 6, "bold"), fill=hs_text_colour)
            self.canvas.create_line(leg_x, hs_centre_y + switch_half_height, leg_x, phase_y, fill="#475569", width=1)
            self.canvas.create_oval(leg_x - 2, phase_y - 2, leg_x + 2, phase_y + 2, fill=colour, outline=colour)
            self.canvas.create_line(leg_x, phase_y, leg_x, ls_centre_y - switch_half_height, fill="#475569", width=1)
            self.canvas.create_rectangle(leg_x - 8, ls_centre_y - switch_half_height, leg_x + 8, ls_centre_y + switch_half_height, outline="#991b1b", fill=ls_fill)
            self.canvas.create_text(leg_x, ls_centre_y, text="LS", font=("Segoe UI", 6, "bold"), fill=ls_text_colour)
            self.canvas.create_line(leg_x, ls_centre_y + switch_half_height, leg_x, minus_rail_y, fill="#475569", width=1)
            self.canvas.create_line(leg_x, phase_y, inverter_right, phase_y, fill=colour, width=2)
            self.canvas.create_line(inverter_right, phase_y, 954, phase_y, fill=colour, width=2)
            self.canvas.create_text(877, phase_y - 8, text=phase.upper(), font=("Segoe UI", 7, "bold"), fill=colour)
            sensor_left, sensor_right = 886, 921
            self.canvas.create_rectangle(sensor_left, phase_y - 7, sensor_right, phase_y + 7, outline=colour, fill="#ffffff")
            self.canvas.create_text((sensor_left + sensor_right) / 2, phase_y, text=f"I_{phase}", font=("Segoe UI", 7, "bold"), fill=colour)
            sensor_centres.append(((sensor_left + sensor_right) / 2, phase_y))

        self.canvas.create_text(802, 586, text="gruen: HS | rot: LS", font=("Segoe UI", 7), fill="#475569")

        # Motor und RLG-Sensor.
        motor_x, motor_y, motor_radius = 982, 550, 27
        self.canvas.create_oval(motor_x - motor_radius, motor_y - motor_radius, motor_x + motor_radius, motor_y + motor_radius, outline="#334155", width=2, fill="#f8fafc")
        self.canvas.create_text(motor_x, motor_y - 5, text="M", font=("Segoe UI", 14, "bold"), fill="#0f172a")
        self.canvas.create_text(motor_x, motor_y + 12, text="3~", font=("Segoe UI", 8), fill="#475569")

        # iabc-Rueckfuehrung: alle drei Stromsensoren gehen auf einen
        # gemeinsamen ADC-Eingang im µC.
        feedback_y = 594
        for sensor_x, sensor_y in sensor_centres:
            self.canvas.create_line(sensor_x, sensor_y + 7, sensor_x, feedback_y, fill="#64748b", dash=(2, 2))
        current_input_x = 704
        self.canvas.create_line(sensor_centres[0][0], feedback_y, current_input_x, feedback_y, fill="#64748b", dash=(2, 2))
        self.canvas.create_line(current_input_x, feedback_y, current_input_x, 566, fill="#64748b", dash=(2, 2), arrow=tk.LAST)
        self.canvas.create_text(790, feedback_y + 8, text="iabc", font=("Segoe UI", 8), fill="#475569")

        # Rotorlage: der RLG hat einen separaten Eingang in den µC.
        rlg_left, rlg_right, rlg_y = 916, 974, 628
        self.canvas.create_rectangle(rlg_left, rlg_y - 12, rlg_right, rlg_y + 12, outline="#111827", width=2, fill="#f9fafb")
        self.canvas.create_text((rlg_left + rlg_right) / 2, rlg_y, text="RLG", font=("Segoe UI", 9, "bold"), fill="#111827")
        self.canvas.create_line(motor_x, motor_y + motor_radius, motor_x, rlg_y - 12, fill="#111827", dash=(3, 2))
        rotor_input_x = 720
        self.canvas.create_line(rlg_left, rlg_y, rotor_input_x, rlg_y, fill="#111827", dash=(3, 2))
        self.canvas.create_line(rotor_input_x, rlg_y, rotor_input_x, 556, fill="#111827", dash=(3, 2), arrow=tk.LAST)
        self.canvas.create_text(815, rlg_y - 9, text="Winkel", font=("Segoe UI", 8), fill="#111827")

if __name__ == "__main__":
    InverterVisualizer().mainloop()
