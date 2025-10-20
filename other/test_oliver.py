#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# routine reads all devices currently accessible via pyvisa interface (USB and LAN)

import sys
import re
from PyQt6.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView, QAbstractItemView, QLabel, QStatusBar
)

# ---- optional: fallbacks, falls pyvisa fehlt
try:
    import pyvisa
except ImportError as e:
    pyvisa = None

# -------- Worker Infrastruktur --------
class ScanSignals(QObject):
    finished = pyqtSignal(list, str)   # results, message
    error = pyqtSignal(str)

class ScanWorker(QRunnable):
    def __init__(self, only_u204x: bool):
        super().__init__()
        self.only_u204x = only_u204x
        self.signals = ScanSignals()

    def run(self):
        if pyvisa is None:
            self.signals.error.emit("pyvisa ist nicht installiert. Bitte 'pip install pyvisa' ausführen.")
            return
        try:
            rm = pyvisa.ResourceManager()  # nutzt Keysight VISA oder pyvisa-py
        except Exception as e:
            self.signals.error.emit(f"ResourceManager konnte nicht erstellt werden:\n{e}")
            return

        try:
            resources = rm.list_resources("?*INSTR")
        except Exception as e:
            self.signals.error.emit(f"Ressourcen konnten nicht gelistet werden:\n{e}")
            try:
                rm.close()
            except:
                pass
            return

        results = []
        for r in resources:
            idn = ""
            match = False
            note = ""
            try:
                inst = rm.open_resource(r)
                # kurze Timeouts, damit nichts hängt
                inst.timeout = 500  # ms
                # Manche Backends brauchen Terminierung nicht manuell
                try:
                    idn = inst.query("*IDN?").strip()
                except Exception as qe:
                    # Zweiter Versuch: getrennt schreiben/lesen
                    try:
                        inst.write("*IDN?")
                        idn = inst.read().strip()
                    except Exception as qe2:
                        note = f"IDN fehlgeschlagen: {qe2}"
                finally:
                    try:
                        inst.close()
                    except:
                        pass
            except Exception as oe:
                note = f"Open fehlgeschlagen: {oe}"

            # Heuristik: Keysight U204x erkennen
            # Beispiel-IDN: "Keysight Technologies,U2049XA,MY12345678,A.02.10"
            idn_upper = (idn or "").upper()
            if "U204" in idn_upper and ("KEYSIGHT" in idn_upper or "AGILENT" in idn_upper or "HEWLETT-PACKARD" in idn_upper):
                match = True

            # Optional zusätzlich nach USB-VendorID 0x2A8D im Ressourcenstring suchen
            # USB0::0x2A8D::0x1F01::SERIAL::INSTR
            if not match and r.startswith("USB"):
                m = re.search(r"0x([0-9A-Fa-f]{4})::0x([0-9A-Fa-f]{4})", r)
                if m:
                    vid = m.group(1).lower()
                    # Keysight VID ist i.d.R. 0x2a8d, ältere Agilent 0x0957
                    if vid in ("2a8d", "0957") and "U204" in idn_upper:
                        match = True

            if (not self.only_u204x) or match:
                results.append({
                    "resource": r,
                    "idn": idn,
                    "is_u204x": match,
                    "note": note
                })

        try:
            rm.close()
        except:
            pass

        msg = f"{len(results)} Einträge" + (" (gefiltert)" if self.only_u204x else "")
        self.signals.finished.emit(results, msg)

# -------- GUI --------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISA Scan – Keysight U204xA Finder")
        self.resize(900, 420)
        self.threadpool = QThreadPool()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Controls
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Scan / Refresh")
        self.btn_copy = QPushButton("Auswahl kopieren")
        self.chk_only = QCheckBox("Nur Keysight U204x zeigen")
        self.lbl_backend = QLabel(self._backend_info())
        self.lbl_backend.setToolTip("Anzeige des aktuell verwendeten VISA-Backends")
        self.lbl_backend.setStyleSheet("color: gray;")
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        row.addWidget(self.chk_only)
        row.addWidget(self.lbl_backend)
        layout.addLayout(row)

        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Resource", "IDN", "U204x", "Hinweis"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Statusbar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Signals
        self.btn_refresh.clicked.connect(self.start_scan)
        self.btn_copy.clicked.connect(self.copy_selected)
        self.table.itemDoubleClicked.connect(self.copy_selected)
        self.chk_only.stateChanged.connect(self.start_scan)

        # Initial scan
        self.start_scan()

    def _backend_info(self) -> str:
        if pyvisa is None:
            return "VISA: (pyvisa nicht installiert)"
        try:
            rm = pyvisa.ResourceManager()
            info = f"VISA: {rm.session}"  # zeigt Backend-Kennung
            rm.close()
            return info
        except Exception as e:
            return f"VISA-Backend Fehler: {e}"

    def start_scan(self):
        only_u204 = self.chk_only.isChecked()
        self.status.showMessage("Scanne VISA-Ressourcen …")
        self.table.setRowCount(0)
        worker = ScanWorker(only_u204)
        worker.signals.finished.connect(self.populate)
        worker.signals.error.connect(self.scan_error)
        self.threadpool.start(worker)

    def populate(self, results, msg):
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            it_res = QTableWidgetItem(r["resource"])
            it_idn = QTableWidgetItem(r["idn"] or "")
            it_match = QTableWidgetItem("Ja" if r["is_u204x"] else "Nein")
            it_note = QTableWidgetItem(r["note"] or "")
            # etwas Styling
            if r["is_u204x"]:
                it_match.setForeground(Qt.GlobalColor.darkGreen)
            else:
                it_match.setForeground(Qt.GlobalColor.darkRed)
            self.table.setItem(i, 0, it_res)
            self.table.setItem(i, 1, it_idn)
            self.table.setItem(i, 2, it_match)
            self.table.setItem(i, 3, it_note)
        self.status.showMessage(f"Fertig: {msg}", 4000)

    def scan_error(self, message: str):
        self.status.showMessage("Fehler beim Scan", 4000)
        QMessageBox.critical(self, "Scan-Fehler", message)

    def copy_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            # falls nichts selektiert ist, nimm die Zeile unter dem Cursor (wenn möglich)
            idx = self.table.currentRow()
            if idx < 0:
                QMessageBox.information(self, "Hinweis", "Bitte eine Zeile auswählen.")
                return
            rows = [self.table.model().index(idx, 0)]
        # sammle Ressourcenstrings
        res_list = []
        for idx in rows:
            row = idx.row()
            item = self.table.item(row, 0)
            if item:
                res_list.append(item.text())
        if res_list:
            text = "\n".join(res_list)
            QApplication.clipboard().setText(text)
            self.status.showMessage("Ressourcen in die Zwischenablage kopiert", 3000)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
