#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time
from collections import deque
from typing import List


from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox
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
    value = pyqtSignal(int, float)
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
                p = self.sensor.read_power_dbm()
                self.value.emit(self.index, p)
            except Exception as e:
                self.status.emit(self.index, f"Read error: {e}")
                self.msleep(100)


# -------------------- Main Window --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power Meter Live (Top 3)")
        self.resize(1000, 700)

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

        # ----- UI -----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Horizontal value display ---
        value_row = QHBoxLayout()
        self.value_labels: List[QLabel] = []
        for i in range(3):
            lbl = QLabel("--.-- dBm")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 48px; font-weight: bold; color: #00bfff;")
            value_row.addWidget(lbl, 1)
            self.value_labels.append(lbl)
        main_layout.addLayout(value_row)

        # --- Plots ---
        self.graphs, self.graph_curves, self.buffers = [], [], []
        self.window_sec = 20.0
        self.start_time = time.time()

        for i in range(3):
            plot = pg.PlotWidget(title=f"Device {i+1}")
            plot.setLabel("left", "Power", units="dBm")
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True)
            curve = plot.plot([], [], pen=pg.mkPen(width=2, color=(0, 180, 255)))
            self.graphs.append(plot)
            self.graph_curves.append(curve)
            self.buffers.append(deque())
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

        # Autostart if any devices
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

    @pyqtSlot()
    def start_readout(self):
        self.stop_readout()
        self.sensors.clear()
        self.threads.clear()

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

        self.start_time = time.time()

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
        for lbl in self.value_labels:
            lbl.setText("--.-- dBm")

    # ------------------ Data updates ------------------
    @pyqtSlot(int, float)
    def on_value(self, idx: int, dbm: float):
        self.value_labels[idx].setText(f"{dbm:0.2f} dBm")

        now = time.time() - self.start_time
        buf = self.buffers[idx]
        buf.append((now, dbm))
        while buf and now - buf[0][0] > self.window_sec:
            buf.popleft()

        xs, ys = zip(*buf) if buf else ([], [])
        self.graph_curves[idx].setData(xs, ys)
        self.graphs[idx].setXRange(max(0, now - self.window_sec), now)

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
