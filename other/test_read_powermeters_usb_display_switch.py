#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, math
from collections import deque
from typing import List

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QLineEdit
)
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


# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power Meter Live (Top 3)")
        self.resize(1150, 860)

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
        # buffers store absolute timestamps: (t_abs_seconds, raw_dBm)
        self.buffers = [deque(), deque(), deque()]

        # ----- UI -----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Header with unit + time toggles ---
        header_row = QHBoxLayout()
        title_label = QLabel("Live Power (offset-corrected)")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_row.addWidget(title_label)
        header_row.addStretch(1)

        self.btn_time = QPushButton("Time: elapsed  (click for local)")
        self.btn_time.clicked.connect(self.toggle_time_mode)
        header_row.addWidget(self.btn_time)

        self.btn_unit = QPushButton("Unit: dBm  (click for W)")
        self.btn_unit.clicked.connect(self.toggle_unit)
        header_row.addWidget(self.btn_unit)

        main_layout.addLayout(header_row)

        # --- Big values (3 columns) ---
        value_row = QHBoxLayout()
        self.value_labels: List[QLabel] = []
        for i in range(3):
            lbl = QLabel("--.-- dBm")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 48px; font-weight: bold; color: #00bfff;")
            value_row.addWidget(lbl, 1)
            self.value_labels.append(lbl)
        main_layout.addLayout(value_row)

        # --- Offsets (attenuation in dB) ---
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

        # --- Plots ---
        self.graphs, self.graph_curves = [], []
        # start with elapsed-time numeric axis
        for i in range(3):
            plot = pg.PlotWidget(title=f"Device {i+1}")
            plot.setLabel("left", "Power", units="dBm")
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True)
            plot.enableAutoRange(y=True)
            curve = plot.plot([], [], pen=pg.mkPen(width=2, color=(0, 180, 255)))
            self.graphs.append(plot)
            self.graph_curves.append(curve)
            main_layout.addWidget(plot)

        # --- Buttons ---
        button_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_rescan = QPushButton("Rescan")
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_stop)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_rescan)
        main_layout.addLayout(button_row)

        self.btn_start.clicked.connect(self.start_readout)
        self.btn_stop.clicked.connect(self.stop_readout)
        self.btn_rescan.clicked.connect(self.rescan)

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
        self.redraw_all_from_buffers()

    @pyqtSlot()
    def toggle_time_mode(self):
        self.time_mode = "local" if self.time_mode == "elapsed" else "elapsed"
        if self.time_mode == "local":
            self.btn_time.setText("Time: local  (click for elapsed)")
            # switch bottom axis to DateAxisItem
            for g in self.graphs:
                date_axis = pg.DateAxisItem(orientation='bottom')
                g.getPlotItem().setAxisItems({'bottom': date_axis})
                g.setLabel("bottom", "Local Time")
        else:
            self.btn_time.setText("Time: elapsed  (click for local)")
            # switch back to numeric axis (seconds)
            for g in self.graphs:
                # Replace with a standard AxisItem
                num_axis = pg.AxisItem(orientation='bottom')
                g.getPlotItem().setAxisItems({'bottom': num_axis})
                g.setLabel("bottom", "Time", units="s")
        self.redraw_all_from_buffers()

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
        self.redraw_all_from_buffers()

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
        self.redraw_from_buffer(i, force_with_last=True)

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

    @pyqtSlot()
    def rescan(self):
        self.stop_readout()
        self.selected = self._detect_devices()[:3]
        for i in range(3):
            self.value_labels[i].setText("--.-- " + self.unit_mode)
            self.buffers[i].clear()
            self.last_dbm[i] = None

    # ------------------ Data updates & drawing ------------------
    @pyqtSlot(int, float)
    def on_value(self, idx: int, measured_dbm: float):
        self.last_dbm[idx] = measured_dbm
        t_abs = time.time()  # store absolute time always
        self.buffers[idx].append((t_abs, measured_dbm))
        # keep only window_sec
        while self.buffers[idx] and (t_abs - self.buffers[idx][0][0] > self.window_sec):
            self.buffers[idx].popleft()
        self.redraw_from_buffer(idx)

    def redraw_all_from_buffers(self):
        for i in range(3):
            self.redraw_from_buffer(i, force_with_last=True)

    def redraw_from_buffer(self, idx: int, force_with_last: bool = False):
        buf = self.buffers[idx]
        off = self.offset_db[idx]

        if (not buf) and force_with_last and (self.last_dbm[idx] is not None):
            t_abs = time.time()
            buf.append((t_abs, self.last_dbm[idx]))

        if not buf:
            return

        xs = []
        ys = []
        for t_abs, p_meas_dbm in buf:
            p_corr_dbm = p_meas_dbm + off
            if self.time_mode == "elapsed":
                x = t_abs - self.start_time
            else:
                x = t_abs  # seconds since epoch; DateAxisItem expects this
            xs.append(x)
            if self.unit_mode == "dBm":
                ys.append(p_corr_dbm)
            else:
                ys.append(dbm_to_watt(p_corr_dbm))

        # Update plot
        self.graph_curves[idx].setData(xs, ys)

        now_abs = time.time()
        if self.time_mode == "elapsed":
            now_rel = now_abs - self.start_time
            self.graphs[idx].setXRange(max(0, now_rel - self.window_sec), now_rel)
        else:
            self.graphs[idx].setXRange(now_abs - self.window_sec, now_abs)

        self.graphs[idx].enableAutoRange(y=True)

        # Big current value
        current = ys[-1]
        if self.unit_mode == "dBm":
            self.value_labels[idx].setText(f"{current:0.2f} dBm")
        else:
            self.value_labels[idx].setText(self._format_watt(current))

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

    @pyqtSlot(int, str)
    def on_status(self, idx: int, msg: str):
        self.value_labels[idx].setText(msg)

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
