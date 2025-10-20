#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # PyQt6-kompatibel

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CavitySim(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reflektierte Leistung – Pulsbetrieb (PyQt6, interaktiv)")
        self.resize(1100, 700)

        # --- Fix-Parameter ---
        self.f0   = 1.3e9         # Hz
        self.Q0   = 2.0e10
        self.P_f  = 10.0          # W (Vorwärtsleistung während "ON")

        # --- Plot ---
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        # --- Controls ---
        controls = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(controls)

        # Pulsdauer T_on
        self.spin_Ton = QtWidgets.QDoubleSpinBox()
        self.spin_Ton.setRange(1e-6, 3600.0)
        self.spin_Ton.setDecimals(6)
        self.spin_Ton.setValue(10.0)                 # default 10 s
        self.spin_Ton.setSingleStep(0.1)
        form.addRow("Pulsdauer T_on [s]", self.spin_Ton)

        # Duty-Cycle in %
        self.spin_duty = QtWidgets.QDoubleSpinBox()
        self.spin_duty.setRange(1.0, 99.0)           # 1 .. 99 %
        self.spin_duty.setDecimals(2)
        self.spin_duty.setValue(50.0)                # default 50 %
        self.spin_duty.setSingleStep(1.0)
        form.addRow("Duty-Cycle [%]", self.spin_duty)

        # beta
        self.spin_beta = QtWidgets.QDoubleSpinBox()
        self.spin_beta.setRange(1e-6, 1e3)
        self.spin_beta.setDecimals(6)
        self.spin_beta.setValue(1.0)                 # default 1
        self.spin_beta.setSingleStep(0.1)
        form.addRow("β (Kopplung)", self.spin_beta)

        # dt
        self.spin_dt = QtWidgets.QDoubleSpinBox()
        self.spin_dt.setRange(1e-7, 1.0)
        self.spin_dt.setDecimals(7)
        self.spin_dt.setValue(2e-4)                  # default
        self.spin_dt.setSingleStep(1e-4)
        form.addRow("Δt [s] (Datenabstand)", self.spin_dt)

        # --- NEU: t_end-Steuerung ---
        self.chk_tend_auto = QtWidgets.QCheckBox("t_end automatisch aus T_on & Duty")
        self.chk_tend_auto.setChecked(True)
        form.addRow(self.chk_tend_auto)

        self.spin_tend = QtWidgets.QDoubleSpinBox()
        self.spin_tend.setRange(1e-6, 24*3600.0)
        self.spin_tend.setDecimals(6)
        self.spin_tend.setValue(30.0)                # sinnvoller Startwert
        self.spin_tend.setSingleStep(0.1)
        self.spin_tend.setEnabled(False)             # nur aktiv, wenn Auto aus
        form.addRow("Gesamtanzeige t_end [s]", self.spin_tend)

        # Info
        self.label_info = QtWidgets.QLabel("")
        self.label_info.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.label_info)

        # Update-Button
        self.btn_update = QtWidgets.QPushButton("Neu berechnen")
        form.addRow(self.btn_update)

        # Layout: Plot links, Controls rechts
        central = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(central)
        hl.addWidget(self.canvas, stretch=3)
        hl.addWidget(controls, stretch=2)
        self.setCentralWidget(central)

        # Signals
        self.btn_update.clicked.connect(self.update_plot)
        self.spin_Ton.valueChanged.connect(self.update_plot)
        self.spin_duty.valueChanged.connect(self.update_plot)
        self.spin_beta.valueChanged.connect(self.update_plot)
        self.spin_dt.valueChanged.connect(self.update_plot)
        self.spin_tend.valueChanged.connect(self.update_plot)
        self.chk_tend_auto.toggled.connect(self.on_tend_auto_toggled)

        # Initial plot
        self.update_plot()

    def on_tend_auto_toggled(self, checked: bool):
        self.spin_tend.setEnabled(not checked)
        self.update_plot()

    def simulate(self, T_on, duty, beta, dt, t_end_override=None):
        """
        Periodischer Rechteckpuls:
        - T_on: ON-Dauer je Periode.
        - duty in [0..1] -> Periodendauer T = T_on / duty.
        - t_end_override: wenn gesetzt, erzwingt diese Anzeige-/Simulationsdauer.
                          Andernfalls: 2 Perioden + 0.5*T_on (mind. 1.2*T_on).
        """
        # Abgeleitete Größen
        w0  = 2*np.pi*self.f0
        k_i = w0 / self.Q0
        k_e = beta * k_i
        k   = k_i + k_e
        tau_a = 2.0 / k
        QL = self.Q0 / (1.0 + beta)

        # Periodendauer aus T_on und duty
        duty = max(1e-6, min(0.99, duty))
        T_per = T_on / duty

        # Gesamtdauer
        if t_end_override is None:
            t_end = max(2.0*T_per + 0.5*T_on, 1.2*T_on)
            t_end_mode = "auto"
        else:
            # Verhindere Unsinn wie t_end < dt
            t_end = max(float(t_end_override), float(dt))
            t_end_mode = "manual"

        # Max. Punkte begrenzen (GUI-Responsiveness)
        nmax = 2_000_000
        npts = int(np.ceil(t_end / dt)) + 1
        if npts > nmax:
            dt = t_end / (nmax - 1)
            npts = nmax

        t = np.linspace(0.0, t_end, npts)

        # Eingangssignal s_in (|s|^2 = Power), Pulszug
        s_in = np.zeros_like(t, dtype=complex)
        modt = np.mod(t, T_per)
        on_mask = modt < T_on
        s_in[on_mask] = np.sqrt(self.P_f)  # Phase 0, auf Resonanz

        # Integration des Envelopes (vorwärts-Euler)
        a = np.zeros_like(t, dtype=complex)
        s_out = np.zeros_like(t, dtype=complex)
        sqrt_ke = np.sqrt(k_e)
        decay = -0.5 * k
        for i in range(len(t) - 1):
            a[i+1] = a[i] + dt * (decay * a[i] + sqrt_ke * s_in[i])
            s_out[i] = -s_in[i] + sqrt_ke * a[i]
        s_out[-1] = -s_in[-1] + sqrt_ke * a[-1]

        P_ref = np.abs(s_out)**2

        # Referenzwerte (auf Resonanz)
        Gamma_inf = (beta - 1.0) / (beta + 1.0)
        P_ref_ss  = (Gamma_inf**2) * self.P_f
        P_ref_Tp  = (2.0*beta/(1.0+beta))**2 * self.P_f

        return {
            "t": t, "P_ref": P_ref, "T_on": T_on, "T_per": T_per, "dt": dt,
            "QL": QL, "tau_a": tau_a, "beta": beta,
            "P_ref_ss": P_ref_ss, "P_ref_Tp": P_ref_Tp,
            "t_end": t_end, "t_end_mode": t_end_mode
        }

    def update_plot(self):
        T_on = float(self.spin_Ton.value())
        duty = float(self.spin_duty.value())/100.0
        beta = float(self.spin_beta.value())
        dt   = float(self.spin_dt.value())

        t_end_override = None if self.chk_tend_auto.isChecked() else float(self.spin_tend.value())

        res = self.simulate(T_on, duty, beta, dt, t_end_override=t_end_override)

        t        = res["t"]
        P_ref    = res["P_ref"]
        T_per    = res["T_per"]
        P_ref_ss = res["P_ref_ss"]
        P_ref_Tp = res["P_ref_Tp"]
        T_on     = res["T_on"]
        t_end    = res["t_end"]
        dt_used  = res["dt"]

        self.ax.clear()
        self.ax.plot(t, P_ref, lw=2, label=rf"β={beta:.3g}, Duty={duty*100:.1f}%")
        # Pulsfenster der ersten Perioden
        for k in range(3):
            t_on_start = k*T_per
            t_on_end   = k*T_per + T_on
            if t_on_start <= t[-1]:
                self.ax.axvspan(t_on_start, min(t_on_end, t[-1]), alpha=0.08, color="C0")
                self.ax.axvline(t_on_start, ls="--", color="k", alpha=0.4)
                if t_on_end <= t[-1]:
                    self.ax.axvline(t_on_end, ls="--", color="k", alpha=0.4)

        self.ax.hlines(P_ref_ss, 0, min(t[-1], T_on), colors='gray', linestyles=':',
                       label="Steady-state (Formel)")
        self.ax.hlines(P_ref_Tp, T_on, t[-1], colors='C3', linestyles=':',
                       label="T+ (Formel)")

        self.ax.set_title("Reflektierte Leistung – Pulsbetrieb")
        self.ax.set_xlabel("Zeit [s]")
        self.ax.set_ylabel("P_ref [W]")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

        info = (
            f"f0 = {self.f0/1e9:.3f} GHz, Q0 = {self.Q0:.3e}, P_f = {self.P_f:.3f} W\n"
            f"β = {res['beta']:.6g}, Q_L = {res['QL']:.3e}, τ_a = {res['tau_a']:.6g} s\n"
            f"T_on = {T_on:.6g} s, Duty = {duty*100:.2f} %, T_per = {T_per:.6g} s\n"
            f"t_end ({res['t_end_mode']}) = {t_end:.6g} s, Δt = {dt_used:.6g} s, Punkte = {len(t)}\n"
            f"Steady-state P_ref (Formel) = {P_ref_ss:.6g} W | "
            f"P_ref(T+) (Formel) = {P_ref_Tp:.6g} W"
        )
        # Hinweis, falls manuell zu kurz für typischen Ringdown gewählt
        if (res['t_end_mode'] == "manual") and (t_end < 1.2*T_on):
            info += "\nHinweis: t_end < 1.2·T_on – Ringdown evtl. abgeschnitten."
        self.label_info.setText(info)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = CavitySim()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
