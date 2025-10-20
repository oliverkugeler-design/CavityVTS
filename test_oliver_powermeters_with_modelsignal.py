#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, math, csv
from collections import deque
from typing import List, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QLineEdit, QCheckBox, QFormLayout, QDoubleSpinBox, QFileDialog
)

from pathlib import Path
import pyqtgraph as pg

# ---- VISA / PyVISA
try:
    import pyvisa
except ImportError:
    pyvisa = None


def is_keysight_xseries_power_sensor(resource: str, idn: str) -> bool:
    u_idn = (idn or "").upper()
    if not u_idn:
        return False
    if not any(v in u_idn for v in ("KEYSIGHT", "AGILENT", "HEWLETT-PACKARD")):
        return False
    return any(m in u_idn for m in ("U204", "U205", "L206"))


def dbm_to_watt(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


def watt_to_dbm(w: float) -> float:
    if w <= 0:
        return float("-inf")
    return 10.0 * math.log10(w) + 30.0


# -------------------- VISA Sensor Wrapper --------------------
class VisaPowerSensor:
    def __init__(self, rm, resource: str, timeout_ms: int = 800):
        self.rm = rm
        self.resource = resource
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = timeout_ms
        self.inst.read_termination = '\n'
        self.inst.write_termination = '\n'
        try:
            self.inst.chunk_size = 102400
        except Exception:
            pass
        try:
            self.inst.clear()
            self.inst.write('*CLS')
        except Exception:
            pass
        self.inst.write('UNIT:POW DBM')
        self.inst.write('TRIG:SOUR IMM')
        self.inst.write('INIT:CONT ON')
        try:
            self._idn = self.inst.query('*IDN?').strip()
        except Exception:
            self._idn = ""

    def read_power_dbm(self) -> float:
        for cmd in ['FETCh?', 'FETCh:POW:AVG?', 'READ?']:
            try:
                return float(self.inst.query(cmd))
            except Exception:
                continue
        raise RuntimeError("Read failed")

    def close(self):
        try:
            self.inst.close()
        except Exception:
            pass


# -------------------- Reader Thread --------------------
class SensorReaderThread(QThread):
    value = pyqtSignal(int, float)  # idx, measured_dBm (raw)
    status = pyqtSignal(int, str)

    def __init__(self, index: int, sensor: VisaPowerSensor):
        super().__init__()
        self.index = index
        self.sensor = sensor
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                p_dbm = self.sensor.read_power_dbm()
                self.value.emit(self.index, p_dbm)
            except Exception as e:
                self.status.emit(self.index, f"Read error: {e}")
                self.msleep(100)


# -------------------- Simulation Engine (rate-matched) --------------------
class RateMatchedCavitySim:
    """
    Diskrete Aktualisierung genau an den Zeitpunkten, an denen Messwerte eintreffen.
    Liefert drei Leistungen:
      - P_ref:   reflektierte Leistung   (über vorhandenen Koppler mit beta)
      - P_fwd:   vorwärtslaufende Leistung (Rechteck, Amplitude = P_f)
      - P_tx:    transmittierte Leistung über einen zusätzlichen Antennenport
                  mit Kopplung Q_ext_tx (Standard: 3e11).
    """
    def __init__(self):
        # Basisparameter
        self.f0 = 1.3e9
        self.Q0 = 2.0e10
        self.P_f = 10.0      # Amplitude (Forward-Power während ON)
        self.beta = 1.0
        self.T_on = 10.0
        self.duty = 0.5      # 50%

        # Zusätzlicher Auskoppelport (Transmission) über externes Q:
        self.Q_ext_tx = 3.0e11

        # Abgeleitete Größen
        self._reset_derived()

        # Zustand
        self.a = 0+0j
        self.t0 = None
        self.t_last = None

        # Buffer (absolute Zeit, P_ref[W], P_fwd[W], P_tx[W])
        self.buffer: deque[Tuple[float, float, float, float]] = deque()

    def _reset_derived(self):
        w0 = 2*math.pi*self.f0
        # interner + externer Koppler (vom Messport) wie gehabt
        self.k_i = w0 / self.Q0
        self.k_e = self.beta * self.k_i
        self.k = self.k_i + self.k_e
        self.sqrt_ke = math.sqrt(max(self.k_e, 0.0))

        # zusätzlicher Transmissionsport (Antennen-Koppler) als separate Rate
        self.k_tx = w0 / self.Q_ext_tx

        d = max(1e-6, min(0.99, self.duty))
        self.T_per = self.T_on / d

    def set_params(self, *, beta=None, T_on=None, duty=None, P_f=None, f0=None, Q0=None, Q_ext_tx=None):
        if beta is not None:
            self.beta = float(beta)
        if T_on is not None:
            self.T_on = float(T_on)
        if duty is not None:
            self.duty = float(duty)
        if P_f is not None:
            self.P_f = float(P_f)
        if f0 is not None:
            self.f0 = float(f0)
        if Q0 is not None:
            self.Q0 = float(Q0)
        if Q_ext_tx is not None:
            self.Q_ext_tx = float(Q_ext_tx)
        self._reset_derived()

    def reset_run(self, t_start_abs: float):
        self.a = 0+0j
        self.t0 = t_start_abs
        self.t_last = t_start_abs
        self.buffer.clear()

    def _s_in_at(self, t_rel: float) -> complex:
        modt = t_rel % self.T_per
        on = modt < self.T_on
        # Vorwärts-Signal als Anregung: s_in = sqrt(P_f) während ON-Phase, sonst 0
        return math.sqrt(self.P_f) if on else 0.0

    def step_to(self, t_abs: float):
        if self.t0 is None:
            return None  # nicht gestartet
        if self.t_last is None:
            self.t_last = t_abs

        t_prev = self.t_last
        dt = max(0.0, t_abs - t_prev)
        t_rel_prev = t_prev - self.t0
        s_in_prev = self._s_in_at(t_rel_prev)

        # Dynamik der Feldamplitude a: da/dt = -0.5*k*a + sqrt(k_e)*s_in
        decay = -0.5 * self.k
        self.a = self.a + dt * (decay * self.a + self.sqrt_ke * s_in_prev)

        # Reflektiert: s_out = -s_in + sqrt(k_e)*a
        s_out = -s_in_prev + self.sqrt_ke * self.a
        P_ref = (s_out.real**2 + s_out.imag**2)  # |s_out|^2

        # Vorwärtsleistung (Rechteck): |s_in|^2 = P_f während ON
        P_fwd = (s_in_prev.real**2 + s_in_prev.imag**2)

        # Transmission am zusätzlichen Port: s_tx = sqrt(k_tx)*a  => P_tx = k_tx*|a|^2
        P_tx = self.k_tx * (self.a.real**2 + self.a.imag**2)

        self.t_last = t_abs
        return P_ref, P_fwd, P_tx

    def append_sample(self, t_abs: float):
        res = self.step_to(t_abs)
        if res is None:
            return
        P_ref, P_fwd, P_tx = res
        self.buffer.append((t_abs, P_ref, P_fwd, P_tx))

    def trim(self, window_sec: float):
        if not self.buffer:
            return
        t_abs = self.buffer[-1][0]
        while self.buffer and (t_abs - self.buffer[0][0] > window_sec):
            self.buffer.popleft()


# -------------------- Probleme mit EXE antizipieren und Pfad zur Datenspeicherung --------------------

def _app_base_dir() -> Path:
    import sys
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

# Einmalig bestimmen und (falls nötig) anlegen
DATA_DIR = _app_base_dir() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power Meter Live (+ Computed Overlay)")
        self.resize(1200, 1180)

        if pyvisa is None:
            QMessageBox.critical(self, "Missing dependency", "pyvisa not installed")
            sys.exit(1)

        try:
            self.rm = pyvisa.ResourceManager()
        except Exception as e:
            QMessageBox.critical(self, "VISA error", str(e))
            sys.exit(2)

        self.sensors: List[VisaPowerSensor] = []
        self.threads: List[SensorReaderThread] = []
        self.selected = self._detect_devices()[:3]

        # --- State ---
        self.window_sec = 20.0
        self.start_time = time.time()
        self.unit_mode = "dBm"          # or "W"
        self.time_mode = "elapsed"      # or "local"
        self.offset_db = [0.0, 0.0, 0.0]
        self.last_dbm = [None, None, None]
        self.buffers = [deque(), deque(), deque()]  # (t_abs, raw_dBm)

        # Simulation
        self.sim = RateMatchedCavitySim()
        self.sim_overlay_enabled = True
        self.sim_running = False

        # --- Throttled UI updates ---
        self._dirty_meas = [False, False, False]
        self._dirty_sim = False
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(33)  # ~30 FPS
        self.ui_timer.timeout.connect(self.on_ui_timer)
        self.ui_timer.start()

        # ----- UI -----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Header ---
        header_row = QHBoxLayout()
        title_label = QLabel("Live Power (offset-corrected) + Computed")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_row.addWidget(title_label)
        header_row.addStretch(1)

        self.btn_time = QPushButton("Time: elapsed  (click for local)")
        header_row.addWidget(self.btn_time)

        self.btn_unit = QPushButton("Unit: dBm  (click for W)")
        header_row.addWidget(self.btn_unit)

        self.btn_extract = QPushButton("Extract data")
        self.btn_extract.setToolTip("Save CSV with timestamp, 3 raw dBm, 3 corrected dBm, and computed power (W, dBm)")
        header_row.addWidget(self.btn_extract)

        main_layout.addLayout(header_row)

        # --- Big values ---
        value_row = QHBoxLayout()
        self.value_labels: List[QLabel] = []
        for i in range(3):
            lbl = QLabel("--.-- dBm")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 48px; font-weight: bold; color: #00bfff;")
            value_row.addWidget(lbl, 1)
            self.value_labels.append(lbl)
        main_layout.addLayout(value_row)

        # --- Offsets ---
        offset_row = QHBoxLayout()
        self.offset_edits: List[QLineEdit] = []
        dv = QDoubleValidator(-200.0, 200.0, 3)
        for i in range(3):
            cap = QLabel(f"Offset {i+1} [dB]: ")
            edit = QLineEdit("0.0")
            edit.setValidator(dv)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setPlaceholderText("attenuation [dB]")
            edit.setFixedWidth(160)
            edit.setStyleSheet("font-size: 16px;")
            edit.textChanged.connect(lambda _t, i=i: self.on_offset_changed(i))
            edit.returnPressed.connect(lambda i=i: self.on_offset_changed(i))
            edit.editingFinished.connect(lambda i=i: self.on_offset_changed(i))
            offset_row.addWidget(cap)
            offset_row.addWidget(edit)
            self.offset_edits.append(edit)
            offset_row.addStretch(1)
        main_layout.addLayout(offset_row)

        # --- History window input ---
        hist_row = QHBoxLayout()
        hist_row.addWidget(QLabel("History window [s]:"))
        self.history_edit = QLineEdit(f"{self.window_sec:g}")
        self.history_edit.setValidator(QDoubleValidator(0.1, 1e9, 2))
        self.history_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_edit.setFixedWidth(120)
        self.history_edit.setStyleSheet("font-size: 16px;")
        self.history_edit.textChanged.connect(self.on_history_changed)
        self.history_edit.returnPressed.connect(self.on_history_changed)
        self.history_edit.editingFinished.connect(self.on_history_changed)
        hist_row.addWidget(self.history_edit)
        hist_row.addStretch(1)
        main_layout.addLayout(hist_row)

        # --- Simulation controls (Apply parameters for β, T_on, Duty, P_f) ---
        sim_ctrl = QWidget()
        sim_form = QFormLayout(sim_ctrl)

        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setRange(1e-6, 1e3)
        self.spin_beta.setDecimals(6)
        self.spin_beta.setValue(1.0)
        self.spin_beta.setSingleStep(0.1)
        sim_form.addRow("β (coupling)", self.spin_beta)

        self.spin_Ton = QDoubleSpinBox()
        self.spin_Ton.setRange(1e-6, 1e6)
        self.spin_Ton.setDecimals(6)
        self.spin_Ton.setValue(10.0)
        self.spin_Ton.setSingleStep(0.1)
        sim_form.addRow("T_on [s]", self.spin_Ton)

        self.spin_duty = QDoubleSpinBox()
        self.spin_duty.setRange(1.0, 99.0)
        self.spin_duty.setDecimals(2)
        self.spin_duty.setValue(50.0)
        self.spin_duty.setSingleStep(1.0)
        sim_form.addRow("Duty [%]", self.spin_duty)

        self.spin_Pf = QDoubleSpinBox()
        self.spin_Pf.setRange(1e-9, 1e9)
        self.spin_Pf.setDecimals(6)
        self.spin_Pf.setValue(10.0)
        self.spin_Pf.setSingleStep(0.1)
        sim_form.addRow("P_f [W] (amplitude)", self.spin_Pf)

        self.chk_overlay = QCheckBox("Overlay computed on device plots")
        self.chk_overlay.setChecked(True)
        sim_form.addRow(self.chk_overlay)

        sim_btn_row = QHBoxLayout()
        self.btn_params_apply = QPushButton("Apply parameters")
        self.btn_params_apply.setToolTip("Apply β, T_on, Duty, P_f to the running simulation without reset")
        self.btn_sim_start = QPushButton("Start Computed Signal")
        self.btn_sim_reset  = QPushButton("Reset")
        sim_btn_row.addWidget(self.btn_params_apply)
        sim_btn_row.addWidget(self.btn_sim_start)
        sim_btn_row.addWidget(self.btn_sim_reset)
        sim_form.addRow(sim_btn_row)

        main_layout.addWidget(sim_ctrl)

        # --- Plots (3 devices) ---
        self.graphs, self.graph_curves = [], []
        self.sim_overlay_curves = []
        for i in range(3):
            plot = pg.PlotWidget(title=f"Device {i+1}")
            plot.setLabel("left", "Power", units="dBm")
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True)
            plot.setClipToView(True)
            plot.enableAutoRange(y=True)  # set once
            meas_curve = plot.plot([], [], pen=pg.mkPen(width=2, color=(0, 180, 255)))
            meas_curve.setDownsampling(auto=True)
            sim_curve = plot.plot([], [], pen=pg.mkPen(width=2, style=Qt.PenStyle.DashLine))
            sim_curve.setDownsampling(auto=True)
            self.graphs.append(plot)
            self.graph_curves.append(meas_curve)
            self.sim_overlay_curves.append(sim_curve)
            main_layout.addWidget(plot)

        # --- Vierter Plot: simulierte Signale (P_ref, P_fwd, P_tx) ---
        self.plot_sim = pg.PlotWidget(title="Computed (P_ref, P_fwd, P_tx)")
        self.plot_sim.setLabel("left", "Power", units="W")
        self.plot_sim.setLabel("bottom", "Time", units="s")
        self.plot_sim.showGrid(x=True, y=True)
        self.plot_sim.setClipToView(True)
        self.plot_sim.enableAutoRange(y=True)
        self.plot_sim.addLegend()

        # Drei Kurven: unterschiedliche Stile/Farben
        self.curve_sim_ref = self.plot_sim.plot([], [], name="P_ref", pen=pg.mkPen(width=2))
        self.curve_sim_fwd = self.plot_sim.plot([], [], name="P_fwd", pen=pg.mkPen(width=2, style=Qt.PenStyle.DashLine))
        self.curve_sim_tx  = self.plot_sim.plot([], [], name="P_tx",  pen=pg.mkPen(width=2, color=(200, 0, 0)))

        main_layout.addWidget(self.plot_sim)




        # --- Bottom control buttons ---
        button_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_rescan = QPushButton("Rescan")
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_stop)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_rescan)
        main_layout.addLayout(button_row)

        # connections
        self.btn_start.clicked.connect(self.start_readout)
        self.btn_stop.clicked.connect(self.stop_readout)
        self.btn_rescan.clicked.connect(self.rescan)

        self.btn_params_apply.clicked.connect(self.on_params_apply_clicked)
        self.btn_sim_start.clicked.connect(self.on_sim_start_clicked)
        self.btn_sim_reset.clicked.connect(self.on_sim_reset_clicked)
        self.chk_overlay.toggled.connect(self.on_overlay_toggled)

        self.btn_time.clicked.connect(self.toggle_time_mode)
        self.btn_unit.clicked.connect(self.toggle_unit)
        self.btn_extract.clicked.connect(self.on_extract_clicked)

        if self.selected:
            self.start_readout()

    # ------------------ Device handling ------------------
    def _detect_devices(self):
        resources = []
        try:
            resources = list(self.rm.list_resources("?*INSTR"))
        except Exception:
            pass
        found = []
        for r in resources:
            try:
                inst = self.rm.open_resource(r)
                inst.timeout = 300
                try:
                    idn = inst.query("*IDN?").strip()
                except Exception:
                    idn = ""
                inst.close()
                if is_keysight_xseries_power_sensor(r, idn):
                    found.append((r, idn))
            except Exception:
                continue
        return found

    # ------------------ Unit / Time / History ------------------
    @pyqtSlot()
    def toggle_unit(self):
        self.unit_mode = "W" if self.unit_mode == "dBm" else "dBm"
        if self.unit_mode == "W":
            self.btn_unit.setText("Unit: W  (click for dBm)")
            for g in self.graphs:
                g.setLabel("left", "Power", units="W")
        else:
            self.btn_unit.setText("Unit: dBm  (click for W)")
            for g in self.graphs:
                g.setLabel("left", "Power", units="dBm")
        self._dirty_meas = [True, True, True]
        self._dirty_sim = True
        self.on_ui_timer()

    @pyqtSlot()
    def toggle_time_mode(self):
        self.time_mode = "local" if self.time_mode == "elapsed" else "elapsed"
        if self.time_mode == "local":
            self.btn_time.setText("Time: local  (click for elapsed)")
            for g in self.graphs + [self.plot_sim]:
                date_axis = pg.DateAxisItem(orientation='bottom')
                g.getPlotItem().setAxisItems({'bottom': date_axis})
                g.setLabel("bottom", "Local Time")
        else:
            self.btn_time.setText("Time: elapsed  (click for local)")
            for g in self.graphs + [self.plot_sim]:
                num_axis = pg.AxisItem(orientation='bottom')
                g.getPlotItem().setAxisItems({'bottom': num_axis})
                g.setLabel("bottom", "Time", units="s")
        self._dirty_meas = [True, True, True]
        self._dirty_sim = True
        self.on_ui_timer()

    def _parse_float(self, s: str) -> float:
        try:
            return float(s.replace(",", ".").strip())
        except Exception:
            return float("nan")

    @pyqtSlot()
    def on_history_changed(self):
        v = self._parse_float(self.history_edit.text())
        if math.isnan(v) or v <= 0:
            return
        self.window_sec = v
        self._dirty_meas = [True, True, True]
        self._dirty_sim = True

    # ------------------ Simulation param handlers ------------------
    def on_params_apply_clicked(self):
        # Apply β, T_on, Duty, P_f live (no reset)
        self.sim.set_params(
            beta=self.spin_beta.value(),
            T_on=self.spin_Ton.value(),
            duty=self.spin_duty.value()/100.0,
            P_f=self.spin_Pf.value()
        )
        self._dirty_sim = True
        self.on_ui_timer()

    def on_sim_start_clicked(self):
        self.sim.set_params(
            beta=self.spin_beta.value(),
            T_on=self.spin_Ton.value(),
            duty=self.spin_duty.value()/100.0,
            P_f=self.spin_Pf.value()
        )
        t0 = time.time()
        self.sim.reset_run(t0)
        self.sim.append_sample(t0)   # sofort was sehen
        self.sim_running = True      # <<< WICHTIG
        self._dirty_sim = True
        self.on_ui_timer()



    def on_sim_reset_clicked(self):
        self.sim.reset_run(time.time())
        self._dirty_sim = True
        self.on_ui_timer()

    def on_overlay_toggled(self, checked: bool):
        self.sim_overlay_enabled = checked
        self._dirty_meas = [True, True, True]
        self.on_ui_timer()

    # ------------------ Readout control ------------------
    @pyqtSlot()
    def start_readout(self):
        self.stop_readout()
        self.sensors.clear()
        self.threads.clear()
        self.start_time = time.time()
        for i in range(3):
            self.buffers[i].clear()
            self.last_dbm[i] = None
            self.value_labels[i].setText("--.-- " + self.unit_mode)

        for i, (res, _) in enumerate(self.selected):
            try:
                s = VisaPowerSensor(self.rm, res)
                t = SensorReaderThread(i, s)
                t.value.connect(self.on_value)
                t.status.connect(self.on_status)
                self.sensors.append(s)
                self.threads.append(t)
                t.start()
            except Exception as e:
                self.value_labels[i].setText(f"Error: {e}")

    @pyqtSlot()
    def stop_readout(self):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.wait(300)
        for s in self.sensors:
            s.close()
        self.threads.clear()
        self.sensors.clear()
        # Simulation ebenfalls stoppen
        self.sim_running = False

    @pyqtSlot()
    def rescan(self):
        self.stop_readout()
        self.selected = self._detect_devices()[:3]
        for i in range(3):
            self.value_labels[i].setText("--.-- " + self.unit_mode)
            self.buffers[i].clear()
            self.last_dbm[i] = None

    # ------------------ Data updates (mark dirty) ------------------
    @pyqtSlot(int, float)
    def on_value(self, idx: int, measured_dbm: float):
        t_abs = time.time()

        # update measurement buffers
        self.last_dbm[idx] = measured_dbm
        self.buffers[idx].append((t_abs, measured_dbm))
        while self.buffers[idx] and (t_abs - self.buffers[idx][0][0] > self.window_sec):
            self.buffers[idx].popleft()

        # simulation step (rate-matched)
        if self.sim_running and self.sim.t0 is not None:
            self.sim.append_sample(t_abs)
            self.sim.trim(self.window_sec)
            self._dirty_sim = True

        # mark for throttled redraw
        self._dirty_meas[idx] = True

    # ------------------ Throttled UI drawing ------------------
    def on_ui_timer(self):
        # Falls Simulation läuft, aber keine Sensor-Threads aktiv sind,
        # treiben wir die Simulation über den UI-Timer.
        if self.sim_running and self.sim.t0 is not None and len(self.threads) == 0:
            now_abs = time.time()
            self.sim.append_sample(now_abs)
            self.sim.trim(self.window_sec)
            self._dirty_sim = True

        # Messplots ...
        for i, dirty in enumerate(self._dirty_meas):
            if dirty:
                self._redraw_from_buffer(i)
                self._dirty_meas[i] = False

        if self._dirty_sim:
            self._redraw_sim_plots()
            self._dirty_sim = False


    def _convert_time_x(self, t_abs_list: List[float]) -> List[float]:
        if self.time_mode == "elapsed":
            return [t - self.start_time for t in t_abs_list]
        else:
            return t_abs_list

    def _redraw_from_buffer(self, idx: int):
        buf = self.buffers[idx]
        if not buf:
            return

        xs_abs = [t for (t, _) in buf]
        xs = self._convert_time_x(xs_abs)

        off = self.offset_db[idx]
        ys = []
        for _, p_meas_dbm in buf:
            p_corr_dbm = p_meas_dbm + off
            if self.unit_mode == "dBm":
                ys.append(p_corr_dbm)
            else:
                ys.append(dbm_to_watt(p_corr_dbm))

        # Update Messkurve
        self.graph_curves[idx].setData(xs, ys)

        # X-Range (Sliding Window)
        now_abs = time.time()
        if self.time_mode == "elapsed":
            now_rel = now_abs - self.start_time
            self.graphs[idx].setXRange(max(0, now_rel - self.window_sec), now_rel, padding=0)
        else:
            self.graphs[idx].setXRange(now_abs - self.window_sec, now_abs, padding=0)

        # Großer aktueller Wert
        current = ys[-1]
        if self.unit_mode == "dBm":
            self.value_labels[idx].setText(f"{current:0.2f} dBm")
        else:
            self.value_labels[idx].setText(self._format_watt(current))

        # Overlay der Simulation (P_fwd → Plot 1, P_ref → Plot 2, P_tx → Plot 3)
        if self.sim_overlay_enabled and self.sim.buffer:
            xs_sim_abs = [t for (t, _, _, _) in self.sim.buffer]
            # wähle die passende simulierte Größe je nach Plot-Index
            if idx == 0:      # oberster Plot
                series_W = [p for (_, _, p, _) in self.sim.buffer]   # P_fwd
            elif idx == 1:    # mittlerer Plot
                series_W = [p for (_, p, _, _) in self.sim.buffer]   # P_ref
            else:             # unterster Geräte-Plot
                series_W = [p for (_, _, _, p) in self.sim.buffer]   # P_tx

            xs_sim = self._convert_time_x(xs_sim_abs)
            if self.unit_mode == "dBm":
                ys_sim = [watt_to_dbm(max(1e-30, w)) for w in series_W]
            else:
                ys_sim = series_W
            self.sim_overlay_curves[idx].setData(xs_sim, ys_sim)
        else:
            self.sim_overlay_curves[idx].setData([], [])


    def _redraw_sim_plots(self):
        # Nichts zu zeichnen?
        if not self.sim.buffer:
            self.curve_sim_ref.setData([], [])
            self.curve_sim_fwd.setData([], [])
            self.curve_sim_tx.setData([], [])
            return

        # Zeitachse
        xs_abs = [t for (t, _, _, _) in self.sim.buffer]
        xs = self._convert_time_x(xs_abs)

        # Daten je nach Einheit
        P_ref_W = [p for (_, p, _, _) in self.sim.buffer]
        P_fwd_W = [p for (_, _, p, _) in self.sim.buffer]
        P_tx_W  = [p for (_, _, _, p) in self.sim.buffer]

        if self.unit_mode == "dBm":
            to_dbm = lambda w: watt_to_dbm(max(1e-30, w))
            ys_ref = list(map(to_dbm, P_ref_W))
            ys_fwd = list(map(to_dbm, P_fwd_W))
            ys_tx  = list(map(to_dbm,  P_tx_W))
            self.plot_sim.setLabel("left", "Power", units="dBm")
        else:
            ys_ref, ys_fwd, ys_tx = P_ref_W, P_fwd_W, P_tx_W
            self.plot_sim.setLabel("left", "Power", units="W")

        # Kurven setzen
        self.curve_sim_ref.setData(xs, ys_ref)
        self.curve_sim_fwd.setData(xs, ys_fwd)
        self.curve_sim_tx.setData(xs, ys_tx)

        # X-Range als Sliding Window wie bei den Geräteplots
        now_abs = time.time()
        if self.time_mode == "elapsed":
            now_rel = now_abs - self.start_time
            self.plot_sim.setXRange(max(0, now_rel - self.window_sec), now_rel, padding=0)
        else:
            self.plot_sim.setXRange(now_abs - self.window_sec, now_abs, padding=0)



    # ------------------ Offsets & statuses ------------------
    def _parse_offset(self, i: int) -> float:
        txt = self.offset_edits[i].text().replace(",", ".").strip()
        try:
            return float(txt)
        except ValueError:
            return float("nan")

    def on_offset_changed(self, i: int):
        val = self._parse_offset(i)
        if math.isnan(val):
            return
        self.offset_db[i] = val
        self._dirty_meas[i] = True

    @pyqtSlot(int, str)
    def on_status(self, idx: int, msg: str):
        self.value_labels[idx].setText(msg)

    def _format_watt(self, w: float) -> str:
        if w <= 0 or math.isnan(w):
            return "— W"
        if w >= 1.0:
            return f"{w:0.3f} W"
        elif w >= 1e-3:
            return f"{w/1e-3:0.3f} mW"
        elif w >= 1e-6:
            return f"{w/1e-6:0.3f} µW"
        else:
            return f"{w/1e-9:0.3f} nW"

    # ------------------ CSV export ------------------
    def on_extract_clicked(self):
        """
        Save displayed data to CSV:
          time_epoch_s, time_local, t_elapsed_s,
          meas1_raw_dBm, meas2_raw_dBm, meas3_raw_dBm,
          meas1_corr_dBm, meas2_corr_dBm, meas3_corr_dBm,
          sim_ref_W, sim_ref_dBm,
          sim_fwd_W, sim_fwd_dBm,
          sim_tx_W,  sim_tx_dBm
        Rows sind auf die Vereinigung aller Timestamps ausgerichtet
        (Forward-Fill, um das auf dem Bildschirm Sichtbare zu reflektieren).
        """
        # Snapshot der Buffer
        buf_meas = [list(self.buffers[i]) for i in range(3)]
        buf_sim  = list(self.sim.buffer)

        # Alle Timestamps sammeln
        ts_set = set()
        for i in range(3):
            ts_set.update(t for (t, _) in buf_meas[i])
        ts_set.update(t for (t, _, _, _) in buf_sim)
        if not ts_set:
            QMessageBox.information(self, "Extract data", "No data to export yet.")
            return
        ts_sorted = sorted(ts_set)

        # Forward-Fill-Cursor
        idxs = [0, 0, 0]
        last_raw = [None, None, None]
        sim_idx = 0
        last_sim_refW = None
        last_sim_fwdW = None
        last_sim_txW  = None

        # Dateiname & Zielpfad (in lokalen 'data'-Ordner, siehe frühere Änderung)
        fname = time.strftime("power_export_%Y%m%d_%H%M%S.csv", time.localtime())
        path = DATA_DIR / fname  # wird gleich in open(...) zu einem String; oder ersetze hier durch DATA_DIR / fname
        # Wenn du bereits DATA_DIR nutzt, nimm:
        # from pathlib import Path
        # path = Path("data") / fname
        # Path("data").mkdir(exist_ok=True)

        with open(str(path), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "time_epoch_s", "time_local", "t_elapsed_s",
                "meas1_raw_dBm", "meas2_raw_dBm", "meas3_raw_dBm",
                "meas1_corr_dBm", "meas2_corr_dBm", "meas3_corr_dBm",
                "sim_ref_W", "sim_ref_dBm",
                "sim_fwd_W", "sim_fwd_dBm",
                "sim_tx_W",  "sim_tx_dBm"
            ])
            for t in ts_sorted:
                # Messkanäle vorrücken bis <= t
                row_raw, row_corr = [], []
                for ch in range(3):
                    b = buf_meas[ch]
                    while idxs[ch] < len(b) and b[idxs[ch]][0] <= t:
                        last_raw[ch] = b[idxs[ch]][1]
                        idxs[ch] += 1
                    raw = last_raw[ch]
                    row_raw.append(raw if raw is not None else "")
                    row_corr.append((raw + self.offset_db[ch]) if raw is not None else "")

                # Simulation vorrücken bis <= t
                while sim_idx < len(buf_sim) and buf_sim[sim_idx][0] <= t:
                    # buf_sim: (t_abs, P_ref, P_fwd, P_tx)
                    _, s_ref, s_fwd, s_tx = buf_sim[sim_idx]
                    last_sim_refW = s_ref
                    last_sim_fwdW = s_fwd
                    last_sim_txW  = s_tx
                    sim_idx += 1

                # dBm-Werte nur, wenn > 0
                sim_ref_dBm = watt_to_dbm(last_sim_refW) if (last_sim_refW is not None and last_sim_refW > 0) else ""
                sim_fwd_dBm = watt_to_dbm(last_sim_fwdW) if (last_sim_fwdW is not None and last_sim_fwdW > 0) else ""
                sim_tx_dBm  = watt_to_dbm(last_sim_txW)  if (last_sim_txW  is not None and last_sim_txW  > 0) else ""

                # Zeitstempel
                t_epoch  = f"{t:.6f}"
                t_elapsed = f"{(t - self.start_time):.6f}"
                t_local  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

                w.writerow([
                    t_epoch, t_local, t_elapsed,
                    *row_raw, *row_corr,
                    (last_sim_refW if last_sim_refW is not None else ""), sim_ref_dBm,
                    (last_sim_fwdW if last_sim_fwdW is not None else ""), sim_fwd_dBm,
                    (last_sim_txW  if last_sim_txW  is not None else ""), sim_tx_dBm
                ])

        QMessageBox.information(self, "Extract data", f"Saved CSV:\n{path}")


    def closeEvent(self, ev):
        self.stop_readout()
        try:
            self.rm.close()
        except Exception:
            pass
        ev.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
