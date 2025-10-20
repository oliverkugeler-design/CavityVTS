#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import math
import socket
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvisa
except ImportError:
    pyvisa = None

# ------------------ Konfiguration ------------------
MEAS_FREQ_HZ = 1.0e9          # Messfrequenz, nach Bedarf anpassen
SAMPLE_PERIOD = 0.10          # Abtastintervall (s)
MAX_POINTS = 600              # wie viele Punkte in den Plots behalten
MAX_SENSORS = 4               # “erste 4” Sensoren

# Falls VISA nichts findet, kannst du hier IPs eintragen (Port 5025 = SCPI)
FALLBACK_IPS = [
    # "192.168.1.101",
    # "192.168.1.102",
    # "192.168.1.103",
    # "192.168.1.104",
]

# gut unterscheidbare Farben auf dunklem oder hellem Hintergrund
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ------------------ Hilfsfunktionen ------------------
def watt_to_dbm(w: float) -> float:
    if w is None or not np.isfinite(w) or w <= 0:
        return float('-inf')
    return 10.0 * math.log10(w) + 30.0

def is_l2061_idn(idn: str) -> bool:
    if not idn:
        return False
    s = idn.upper()
    return ("L2061" in s) or ("2050/60" in s and "LAN" in s)

# ------------------ VISA Discovery ------------------
def discover_sensors_via_visa(max_count=MAX_SENSORS) -> List[Tuple[str, object]]:
    devices = []
    if pyvisa is None:
        return devices
    try:
        rm = pyvisa.ResourceManager()
        for addr in rm.list_resources():
            # wir interessieren uns für LAN-Instrumente
            if not addr.startswith("TCPIP"):
                continue
            try:
                inst = rm.open_resource(addr, timeout=5000)
                inst.read_termination = '\n'
                inst.write_termination = '\n'
                idn = inst.query("*IDN?")
                if is_l2061_idn(idn):
                    devices.append((addr, inst))
                    if len(devices) >= max_count:
                        break
                else:
                    inst.close()
            except Exception:
                try:
                    inst.close()
                except Exception:
                    pass
                continue
    except Exception:
        pass
    return devices

# ------------------ Raw Socket Fallback ------------------
class SocketSensor:
    def __init__(self, ip, port=5025, timeout=5.0):
        self.ip = ip; self.port = port
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._w = self.sock.makefile("wb", buffering=0)
        self._r = self.sock.makefile("r", buffering=1, newline='\n')

    def write(self, cmd: str):
        self._w.write((cmd + "\n").encode("ascii"))

    def query(self, cmd: str) -> str:
        self.write(cmd)
        return self._r.readline().strip()

    def close(self):
        try:
            self._w.close(); self._r.close(); self.sock.close()
        except Exception:
            pass

def discover_sensors_via_ips(ips: List[str], max_count=MAX_SENSORS):
    found = []
    for ip in ips:
        try:
            s = SocketSensor(ip)
            idn = s.query("*IDN?")
            if is_l2061_idn(idn):
                found.append((f"TCPIP0::{ip}::inst0::INSTR", s))
                if len(found) >= max_count:
                    break
            else:
                s.close()
        except Exception:
            continue
    return found

# ------------------ Initialisierung ------------------
def init_sensor(session) -> None:
    """
    Setzt Frequenz u. sinnvolle Defaults.
    MEASure:SCALar:POWer:AC? liefert die Leistung in der aktuell gesetzten Einheit.
    Wir lassen standardmäßig Watt, rechnen dBm lokal.
    """
    try:
        # Sicherheitsreset ist optional; kann Messung kurz unterbrechen
        # session.write("*CLS")
        # Frequenz setzen (wichtig für Korrekturen im Sensor)
        session.write(f"SENSe:FREQuency {MEAS_FREQ_HZ}")
        # Einheit auf Watt, damit MEAS? in W zurückgibt
        session.write("UNIT:POWer W")
        # kontinuierliche Messung ist für MEAS? nicht nötig; die MEAS-Query triggert intern
        # trotzdem sicherstellen, dass nichts blockiert:
        # session.write("INIT:CONT ON")
    except Exception as e:
        print("Init warn:", e)

def query_power_watt(session) -> float:
    """
    Liest eine Einzelmessung in Watt.
    SCPI: MEASure[:SCALar]:POWer[:AC]? [expected, resolution, (@chan)]
    """
    try:
        val = session.query("MEAS:SCAL:POW:AC?")  # Kurzformen sind laut PG erlaubt
        return float(val)
    except Exception:
        return float("nan")

def close_session(session):
    try:
        session.close()
    except Exception:
        pass

# ------------------ Hauptprogramm ------------------
def main():
    # 1) Suche Sensoren (VISA, dann Fallback IPs)
    devices = discover_sensors_via_visa()
    if not devices and FALLBACK_IPS:
        devices = discover_sensors_via_ips(FALLBACK_IPS)

    if not devices:
        print("Keine L2061X-Sensoren gefunden. Prüfe Keysight IO Libraries oder trage IPs in FALLBACK_IPS ein.")
        return

    # nur die ersten vier
    devices = devices[:MAX_SENSORS]
    names = []
    for addr, sess in devices:
        try:
            idn = sess.query("*IDN?")
        except Exception:
            idn = addr
        names.append(idn.strip())
        init_sensor(sess)

    print("Verbunden mit:")
    for i, n in enumerate(names, 1):
        print(f"  {i}: {n}")

    # 2) Live-Plot vorbereiten (3 Achsen)
    plt.figure(figsize=(12, 8))
    ax_w = plt.subplot(3,1,1)
    ax_dbm = plt.subplot(3,1,2, sharex=ax_w)
    ax_rel = plt.subplot(3,1,3, sharex=ax_w)

    ax_w.set_title("Power (W) – live")
    ax_w.set_ylabel("W")
    ax_dbm.set_title("Power (dBm) – live (lokal umgerechnet)")
    ax_dbm.set_ylabel("dBm")
    ax_rel.set_title("Relativ zu Sensor 1 (dB)")
    ax_rel.set_ylabel("Δ (dB)")
    ax_rel.set_xlabel("Zeit (s)")

    lines_w = []
    lines_dbm = []
    lines_rel = []

    t0 = time.time()
    t_data = []
    y_w = [ [] for _ in devices ]
    y_dbm = [ [] for _ in devices ]
    y_rel = [ [] for _ in devices ]

    for i in range(len(devices)):
        color = COLORS[i % len(COLORS)]
        lw, = ax_w.plot([], [], color=color, label=f"S{i+1}")
        ld, = ax_dbm.plot([], [], color=color, label=f"S{i+1}")
        lr, = ax_rel.plot([], [], color=color, label=f"S{i+1}")
        lines_w.append(lw); lines_dbm.append(ld); lines_rel.append(lr)

    ax_w.legend(ncol=min(4,len(devices)), fontsize=9, loc="upper right")
    ax_dbm.legend(ncol=min(4,len(devices)), fontsize=9, loc="upper right")
    ax_rel.legend(ncol=min(4,len(devices)), fontsize=9, loc="upper right")
    plt.tight_layout()

    try:
        while plt.fignum_exists(plt.gcf().number):
            t_now = time.time() - t0
            t_data.append(t_now)
            # nur die letzten MAX_POINTS halten
            if len(t_data) > MAX_POINTS:
                t_data = t_data[-MAX_POINTS:]
                for i in range(len(devices)):
                    y_w[i] = y_w[i][-MAX_POINTS:]
                    y_dbm[i] = y_dbm[i][-MAX_POINTS:]
                    y_rel[i] = y_rel[i][-MAX_POINTS:]

            # Messungen
            vals_w = []
            for _, sess in devices:
                vals_w.append(query_power_watt(sess))

            # Umrechnung + Relativwerte
            for i, wval in enumerate(vals_w):
                y_w[i].append(wval)
                y_dbm[i].append(watt_to_dbm(wval))

            ref_dbm = y_dbm[0][-1] if len(y_dbm[0]) else float('nan')
            for i in range(len(devices)):
                cur = y_dbm[i][-1] if len(y_dbm[i]) else float('nan')
                y_rel[i].append(cur - ref_dbm if np.isfinite(cur) and np.isfinite(ref_dbm) else float('nan'))

            # Daten in Plots schieben
            for i in range(len(devices)):
                lines_w[i].set_data(t_data, y_w[i])
                lines_dbm[i].set_data(t_data, y_dbm[i])
                lines_rel[i].set_data(t_data, y_rel[i])

            # Achsen anpassen
            for ax in (ax_w, ax_dbm, ax_rel):
                ax.relim(); ax.autoscale_view()

            plt.pause(SAMPLE_PERIOD)

    finally:
        for _, sess in devices:
            close_session(sess)

if __name__ == "__main__":
    main()
