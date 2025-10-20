#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, sys, csv, math, glob
from typing import Dict, List, Tuple, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QCheckBox, QComboBox, QMessageBox,
    QScrollArea, QGridLayout
)
import pyqtgraph as pg


def dbm_to_watt(dbm: float) -> float:
    if dbm is None:
        return float("nan")
    if dbm == float("-inf") or dbm != dbm:
        return float("nan")
    return 10 ** ((dbm - 30.0) / 10.0)


def watt_to_dbm(w: float) -> float:
    if w is None or w != w or w <= 0:
        return float("-inf")
    return 10.0 * math.log10(w) + 30.0


class CsvPowerViewer(QMainWindow):
    """
    CSV viewer with:
      - Top plot (dBm): meas raw/corr + sim_dBm (togglable), and
      - Bottom plot (W): sim_W + selected dBm series converted to W.
    Peak analysis on sim_W:
      - Refined t0 (immediately before steepest pre-rise),
      - Spline extrapolation of post-peak window back to t0,
      - Pre-rise baseline windows (stable average).
    NEW:
      - Linear fits of sim_dBm starting at each peak, using only the first
        few samples, with slope→time-constant conversion correct for power.
    """

    DBM_SERIES = [
        "meas1_raw_dBm", "meas2_raw_dBm", "meas3_raw_dBm",
        "meas1_corr_dBm", "meas2_corr_dBm", "meas3_corr_dBm",
        "sim_dBm",
    ]
    W_NATIVE = ["sim_W"]

    PALETTE = {
        "meas1_raw_dBm": (0, 180, 255),
        "meas2_raw_dBm": (255, 140, 0),
        "meas3_raw_dBm": (50, 205, 50),
        "meas1_corr_dBm": (220, 20, 60),
        "meas2_corr_dBm": (160, 32, 240),
        "meas3_corr_dBm": (255, 215, 0),
        "sim_dBm": (200, 200, 200),
        "sim_W": (255, 255, 255),
    }

    # --- peak/spline config ---
    PEAK_SMOOTH_WIN   = 5     # moving average for peak detection
    PEAK_REL_THRESH   = 0.10  # relative threshold for peaks
    FIT_TAIL_LEN      = 80    # samples after peak for spline fit window
    MIN_FIT_PTS       = 4     # min points for cubic spline (else linear)
    PRERISE_LOOKBACK  = 60    # samples to look back before peak for "vertical rise"
    USE_PREV_OF_STEEPEST = True  # use point immediately before the steepest slope index

    # --- baseline config (pre-rise) ---
    BASELINE_WIN         = 20   # initial number of points to try for baseline
    BASELINE_GAP         = 2    # skip this many points right before steep slope
    BASELINE_MIN_PTS     = 6    # minimum acceptable number of points
    BASELINE_REL_STD_MAX = 0.05 # require std/mean <= 5% to call it "reasonably constant"

    # --- NEW: dBm linear fit config ---
    DBM_FIT_POINTS       = 20   # "first couple of points" after each peak (including the peak index)
    MIN_DBM_FIT_PTS      = 5    # minimum points to attempt a fit


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power CSV Viewer")
        self.resize(1560, 980)
 
        # always use the "data" subdirectory relative to current working dir
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        # data
        self.data: Dict[str, List[float]] = {}
        self.x_elapsed: List[float] = []
        self.x_epoch: List[float] = []



        # ----- layout: plots left, controls right -----
        root = QWidget(); self.setCentralWidget(root)
        hlayout = QHBoxLayout(root)

        plots_col = QVBoxLayout(); hlayout.addLayout(plots_col, stretch=1)

        self.plot_dbm = pg.PlotWidget(title="Power (dBm)")
        self.plot_dbm.showGrid(x=True, y=True)
        self.plot_dbm.setLabel("left", "Power", units="dBm")
        self._set_xaxis(self.plot_dbm, "elapsed")
        plots_col.addWidget(self.plot_dbm, stretch=1)

        self.plot_w = pg.PlotWidget(title="Power (W)")
        self.plot_w.showGrid(x=True, y=True)
        self.plot_w.setLabel("left", "Power", units="W")
        self._set_xaxis(self.plot_w, "elapsed")
        plots_col.addWidget(self.plot_w, stretch=1)

        sidebar = QVBoxLayout(); hlayout.addLayout(sidebar, stretch=0)

        # file controls
        head_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.clicked.connect(self.on_open)
        head_row.addWidget(self.btn_open)
        sidebar.addLayout(head_row)

        self.lbl_file = QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("font-weight: bold;")
        sidebar.addWidget(self.lbl_file)

        # x axis
        xrow = QHBoxLayout()
        xrow.addWidget(QLabel("X axis:"))
        self.combo_x = QComboBox()
        self.combo_x.addItems(["Elapsed (s)", "Local Time"])
        self.combo_x.currentIndexChanged.connect(self.on_xmode_changed)
        xrow.addWidget(self.combo_x)
        sidebar.addLayout(xrow)

        # select buttons
        selrow = QHBoxLayout()
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_none = QPushButton("Select none")
        self.btn_select_all.clicked.connect(self.on_select_all)
        self.btn_select_none.clicked.connect(self.on_select_none)
        selrow.addWidget(self.btn_select_all); selrow.addWidget(self.btn_select_none)
        sidebar.addLayout(selrow)

        # analysis buttons
        self.btn_peaks = QPushButton("Spline-extrapolate peaks (sim_W)")
        self.btn_peaks.setToolTip("Use post-peak data; extrapolate spline back to earlier t0 ('vertical rise'). Also computes pre-rise baselines.")
        self.btn_peaks.clicked.connect(self.on_spline_peaks)
        sidebar.addWidget(self.btn_peaks)

        self.btn_fit_dbm = QPushButton("Fit dBm decay from peaks (sim_dBm)")
        self.btn_fit_dbm.setToolTip("Linear fit of sim_dBm starting at each peak (first few points). Converts slope to time constant for power.")
        self.btn_fit_dbm.clicked.connect(self.on_fit_dbm_from_peaks)
        sidebar.addWidget(self.btn_fit_dbm)

        # series toggles in scroll area
        self.series_panel = QWidget()
        grid = QGridLayout(self.series_panel)
        grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("<b>Series</b>"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        self.chk_dbm_visible: Dict[str, QCheckBox] = {}
        self.chk_w_from_dbm: Dict[str, QCheckBox] = {}
        self.chk_sim_w: QCheckBox = QCheckBox("sim_W")
        self.chk_sim_w.setChecked(True)
        self.chk_sim_w.setStyleSheet(f"QCheckBox{{ color: rgb{self.PALETTE['sim_W']}; font-weight: 500; }}")
        self.chk_sim_w.toggled.connect(self.refresh_all)

        grid.addWidget(QLabel("<i>Top (dBm)</i>"), 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("<i>Bottom (→W)</i>"), 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        row = 2
        for key in self.DBM_SERIES:
            cb_dbm = QCheckBox(key); cb_dbm.setChecked(True)
            cb_dbm.setStyleSheet(f"QCheckBox{{ color: rgb{self.PALETTE.get(key,(200,200,200))}; font-weight: 500; }}")
            cb_dbm.toggled.connect(self.refresh_all)
            self.chk_dbm_visible[key] = cb_dbm

            cb_w = QCheckBox("→W"); cb_w.setChecked(key != "sim_dBm")
            cb_w.toggled.connect(self.refresh_all)
            self.chk_w_from_dbm[key] = cb_w

            grid.addWidget(cb_dbm, row, 0, alignment=Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(cb_w,   row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
            row += 1

        grid.addWidget(self.chk_sim_w, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(self.series_panel)
        sidebar.addWidget(scroll, stretch=1)

        # curves
        self.curves_dbm: Dict[str, pg.PlotDataItem] = {}
        self.curves_w_conv: Dict[str, pg.PlotDataItem] = {}
        self.curves_w_native: Dict[str, pg.PlotDataItem] = {}
        self._init_curves()

        # annotations
        self._anno_items_w: List[pg.GraphicsObject] = []     # lower plot annotations
        self._anno_items_dbm: List[pg.GraphicsObject] = []   # top plot annotations (new)

        # default file
        self.try_open_default()

    # ---------- helpers ----------
    def _init_curves(self):
        for key in self.DBM_SERIES:
            pen = pg.mkPen(self.PALETTE.get(key, (200, 200, 200)), width=2)
            c = self.plot_dbm.plot([], [], pen=pen, name=key)
            c.setDownsampling(auto=True)
            self.curves_dbm[key] = c

        for key in self.DBM_SERIES:
            pen = pg.mkPen(self.PALETTE.get(key, (180, 180, 180)), width=2)
            c = self.plot_w.plot([], [], pen=pen, name=f"{key}→W")
            c.setDownsampling(auto=True)
            self.curves_w_conv[key] = c

        for key in self.W_NATIVE:
            pen = pg.mkPen(self.PALETTE.get(key, (255, 255, 255)), width=2)
            c = self.plot_w.plot([], [], pen=pen, name=key)
            c.setDownsampling(auto=True)
            self.curves_w_native[key] = c

    def _set_xaxis(self, plot: pg.PlotWidget, mode: str):
        if mode == "local":
            try:
                date_axis = pg.DateAxisItem(orientation='bottom')
            except Exception:
                from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
                date_axis = DateAxisItem(orientation='bottom')
            plot.getPlotItem().setAxisItems({'bottom': date_axis})
            plot.setLabel("bottom", "Local Time")
        else:
            num_axis = pg.AxisItem(orientation='bottom')
            plot.getPlotItem().setAxisItems({'bottom': num_axis})
            plot.setLabel("bottom", "Time", units="s")

    def try_open_default(self):
        matches = sorted(glob.glob(os.path.join(self.data_dir, "power_export_*.csv")))
        if matches:
            self.load_file(matches[-1])  # latest file in dir

    # ---------- actions ----------
    def on_open(self):
        start = self.data_dir if os.path.isdir(self.data_dir) else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", start, "CSV Files (*.csv)")
        if path:
            self.load_file(path)

    def on_xmode_changed(self, _idx: int):
        mode = "elapsed" if self.combo_x.currentIndex() == 0 else "local"
        self._set_xaxis(self.plot_dbm, mode)
        self._set_xaxis(self.plot_w, mode)
        self.refresh_all()
        self._clear_annotations()

    def on_select_all(self):
        for key in self.DBM_SERIES:
            self.chk_dbm_visible[key].setChecked(True)
            self.chk_w_from_dbm[key].setChecked(True)
        self.chk_sim_w.setChecked(True)

    def on_select_none(self):
        for key in self.DBM_SERIES:
            self.chk_dbm_visible[key].setChecked(False)
            self.chk_w_from_dbm[key].setChecked(False)
        self.chk_sim_w.setChecked(False)

    # ---------- data load / plot ----------
    def load_file(self, path: str):
        try:
            with open(path, "r", newline="") as f:
                rdr = csv.DictReader(f)
                needed = set(["time_epoch_s", "t_elapsed_s"] + self.DBM_SERIES + self.W_NATIVE)
                self.data = {k: [] for k in needed}

                for row in rdr:
                    try:
                        t_epoch = float(row["time_epoch_s"])
                        t_elapsed = float(row["t_elapsed_s"])
                    except Exception:
                        continue
                    self.data["time_epoch_s"].append(t_epoch)
                    self.data["t_elapsed_s"].append(t_elapsed)

                    for key in self.DBM_SERIES:
                        v = row.get(key, "")
                        self.data[key].append(float(v) if v != "" else float("nan"))
                    for key in self.W_NATIVE:
                        v = row.get(key, "")
                        self.data[key].append(float(v) if v != "" else float("nan"))

            self.x_epoch = self.data["time_epoch_s"]
            self.x_elapsed = self.data["t_elapsed_s"]
            self.lbl_file.setText(os.path.basename(path))
            self.refresh_all()
            self._clear_annotations()

        except Exception as e:
            QMessageBox.critical(self, "Open CSV failed", str(e))

    def _current_x(self) -> List[float]:
        return self.x_elapsed if self.combo_x.currentIndex() == 0 else self.x_epoch

    def refresh_all(self):
        if not self.x_elapsed:
            for c in self.curves_dbm.values(): c.setData([], [])
            for c in self.curves_w_conv.values(): c.setData([], [])
            for c in self.curves_w_native.values(): c.setData([], [])
            return

        x = self._current_x()

        # Top (dBm)
        for key, curve in self.curves_dbm.items():
            if self.chk_dbm_visible[key].isChecked():
                y = self.data.get(key, [])
                y_clean = [(v if (v == v and abs(v) != float("inf")) else float("nan")) for v in y]
                curve.setData(x, y_clean)
                curve.setVisible(True)
            else:
                curve.setVisible(False)

        # Bottom (W): converted dBm
        for key, curve in self.curves_w_conv.items():
            if self.chk_w_from_dbm[key].isChecked():
                y_dbm = self.data.get(key, [])
                y_w = [dbm_to_watt(v) for v in y_dbm]
                curve.setData(x, y_w)
                curve.setVisible(True)
            else:
                curve.setVisible(False)

        # Bottom (W): native
        for key, curve in self.curves_w_native.items():
            if key == "sim_W" and self.chk_sim_w.isChecked():
                y = self.data.get(key, [])
                y_clean = [(v if (v == v) else float("nan")) for v in y]
                curve.setData(x, y_clean)
                curve.setVisible(True)
            else:
                curve.setVisible(False)

        self.plot_dbm.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.plot_w.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    # ---------- peak detection ----------
    def _find_first_two_peak_indices(self, y: List[float]) -> List[int]:
        n = len(y)
        if n < 3:
            return []
        win = max(1, int(self.PEAK_SMOOTH_WIN))
        half = win // 2
        ys = []
        for i in range(n):
            i0 = max(0, i-half); i1 = min(n, i+half+1)
            seg = [v for v in y[i0:i1] if v == v]
            ys.append(sum(seg)/len(seg) if seg else float("nan"))

        finite_vals = [v for v in ys if v == v]
        if not finite_vals:
            return []
        ymin, ymax = min(finite_vals), max(finite_vals)
        if ymax <= ymin:
            return []
        min_level = ymin + self.PEAK_REL_THRESH * (ymax - ymin)

        peaks = []
        for i in range(1, n-1):
            a, b, c = ys[i-1], ys[i], ys[i+1]
            if a == a and b == b and c == c:
                if a < b and b >= c and b >= min_level:
                    peaks.append(i)
            if len(peaks) >= 2:
                break
        return peaks

    # ---------- refine t0 (vertical rise just before peak) ----------
    def _refine_t0_vertical(self, t: List[float], y: List[float], i_peak: int) -> Tuple[int, int]:
        """
        Find steepest pre-rise index j in a lookback window; return (i_t0, j),
        where i_t0 = j-1 (point immediately before steepest slope).
        """
        n = len(y)
        if i_peak <= 1:
            return max(0, i_peak-1), max(1, i_peak)
        start = max(1, i_peak - self.PRERISE_LOOKBACK)
        best_j = start
        best_slope = -float("inf")
        for j in range(start, i_peak+1):
            dt = t[j] - t[j-1]
            if dt <= 0:
                continue
            dy = y[j] - y[j-1]
            slope = dy / dt
            if slope > best_slope:
                best_slope = slope
                best_j = j
        i_t0 = max(0, best_j - 1) if self.USE_PREV_OF_STEEPEST else best_j
        return i_t0, best_j

    # ---------- natural cubic spline ----------
    def _spline_coeffs(self, x: List[float], y: List[float]) -> Optional[Tuple[List[float], List[float], List[float], List[float]]]:
        n = len(x)
        if n < 2: return None
        if n == 2:
            a = [y[0]]; h = x[1]-x[0]
            if h == 0: return None
            b = [(y[1]-y[0])/h]; c = [0.0]; d = [0.0]
            return a,b,c,d
        for i in range(n-1):
            if not (x[i+1] > x[i]): return None

        h = [x[i+1]-x[i] for i in range(n-1)]
        alpha = [0.0]*n
        for i in range(1, n-1):
            alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1])

        l = [1.0]+[0.0]*(n-1); mu=[0.0]*n; z=[0.0]*n
        for i in range(1, n-1):
            l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
            if l[i] == 0: return None
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
        l[n-1] = 1.0; z[n-1] = 0.0

        ccoef=[0.0]*n; bcoef=[0.0]*(n-1); dcoef=[0.0]*(n-1); acoef=[y[i] for i in range(n-1)]
        for j in range(n-2, -1, -1):
            ccoef[j] = z[j] - mu[j]*ccoef[j+1]
            bcoef[j] = ((y[j+1]-y[j])/h[j]) - (h[j]*(2.0*ccoef[j]+ccoef[j+1])/3.0)
            dcoef[j] = (ccoef[j+1]-ccoef[j])/(3.0*h[j])
        return acoef, bcoef, ccoef[:-1], dcoef

    def _eval_spline_interval(self, xnodes, a, b, c, d, i, t):
        dx = t - xnodes[i]
        return a[i] + b[i]*dx + c[i]*(dx*dx) + d[i]*(dx*dx*dx)

    def _spline_extrapolate_to_left(self, xnodes: List[float], y: List[float], t_left: float) -> Optional[float]:
        coefs = self._spline_coeffs(xnodes, y)
        if not coefs: return None
        a,b,c,d = coefs
        return self._eval_spline_interval(xnodes, a,b,c,d, 0, t_left)

    # ---------- baseline extraction (pre-rise) ----------
    def _baseline_before_slope(self, t: List[float], y: List[float], j_steep: int) -> Optional[Tuple[List[int], float, float, float]]:
        """
        Select a window ending BEFORE the steep slope:
          end_idx = j_steep - 1 - BASELINE_GAP
        Start with BASELINE_WIN points, shrink from the left until
        std/mean <= BASELINE_REL_STD_MAX or until BASELINE_MIN_PTS.
        Returns (indices_used, mean, std, rel_std) or None.
        """
        end_idx = j_steep - 1 - self.BASELINE_GAP
        if end_idx < 0:
            return None
        start_idx = max(0, end_idx - self.BASELINE_WIN + 1)

        indices = [i for i in range(start_idx, end_idx+1) if (y[i] == y[i])]
        if len(indices) < self.BASELINE_MIN_PTS:
            extra = self.BASELINE_MIN_PTS - len(indices)
            start_idx = max(0, start_idx - extra)
            indices = [i for i in range(start_idx, end_idx+1) if (y[i] == y[i])]
            if len(indices) < self.BASELINE_MIN_PTS:
                return None

        def stats(idxs):
            vals = [y[i] for i in idxs]
            m = sum(vals)/len(vals)
            s = math.sqrt(sum((v-m)*(v-m) for v in vals)/len(vals)) if len(vals) > 1 else 0.0
            rel = (s/abs(m)) if (m != 0) else float("inf")
            return m, s, rel

        i0 = 0
        chosen = indices[i0:]
        mean, std, rel = stats(chosen)
        while (rel > self.BASELINE_REL_STD_MAX) and (len(chosen) > self.BASELINE_MIN_PTS):
            i0 += 1
            chosen = indices[i0:]
            mean, std, rel = stats(chosen)

        if len(chosen) < self.BASELINE_MIN_PTS:
            return None
        return chosen, mean, std, rel

    # ---------- annotations ----------
    def _clear_annotations(self):
        for it in self._anno_items_w:
            try: self.plot_w.removeItem(it)
            except Exception: pass
        self._anno_items_w.clear()
        for it in self._anno_items_dbm:
            try: self.plot_dbm.removeItem(it)
            except Exception: pass
        self._anno_items_dbm.clear()

    def _add_ann_w(self, item: pg.GraphicsObject):
        self._anno_items_w.append(item)

    def _add_ann_dbm(self, item: pg.GraphicsObject):
        self._anno_items_dbm.append(item)

    # ---------- main spline peak workflow (+ baselines) ----------
    def on_spline_peaks(self):
        if not self.data or "sim_W" not in self.data or not self.data["sim_W"]:
            QMessageBox.information(self, "Spline peaks", "No sim_W data loaded.")
            return

        # clear only lower-plot annotations (keep dbm ones if any)
        for it in self._anno_items_w:
            try: self.plot_w.removeItem(it)
            except Exception: pass
        self._anno_items_w.clear()

        t_epoch = self.data["time_epoch_s"]
        t_elapsed = self.data["t_elapsed_s"]
        y = self.data["sim_W"]
        if len(y) < 3:
            QMessageBox.information(self, "Spline peaks", "Not enough data.")
            return

        peak_idxs = self._find_first_two_peak_indices(y)
        if not peak_idxs:
            QMessageBox.information(self, "Spline peaks", "No peaks found.")
            return

        xmode = self.combo_x.currentIndex()  # 0 elapsed, 1 local(epoch)
        lines = []
        baseline_values = []

        for pnum, i_peak in enumerate(peak_idxs[:2], start=1):
            # refine t0 and steep index
            i_t0, j_steep = self._refine_t0_vertical(t_epoch, y, i_peak)
            t0_epoch = t_epoch[i_t0]
            x_disp_t0 = (t_elapsed[i_t0] if xmode == 0 else t0_epoch)

            # baseline (pre-rise)
            base = self._baseline_before_slope(t_epoch, y, j_steep)
            if base:
                idxs_used, base_mean, base_std, base_rel = base
                baseline_values.append(base_mean)

                # plot baseline points
                def idx_to_x(i): return (t_elapsed[i] if xmode == 0 else t_epoch[i])
                xs_b = [idx_to_x(i) for i in idxs_used]
                ys_b = [y[i] for i in idxs_used]
                baseline_pts = pg.ScatterPlotItem(xs_b, ys_b, size=9, symbol='s',
                                                  brush=pg.mkBrush(80, 255, 120, 220),
                                                  pen=pg.mkPen(20, 60, 20, 220))
                self.plot_w.addItem(baseline_pts); self._add_ann_w(baseline_pts)
                if xs_b:
                    line = pg.PlotDataItem([xs_b[0], xs_b[-1]], [base_mean, base_mean],
                                           pen=pg.mkPen(80, 255, 120, width=2, style=Qt.PenStyle.DotLine))
                    self.plot_w.addItem(line); self._add_ann_w(line)
            else:
                base_mean = float("nan"); base_std = float("nan"); base_rel = float("nan")
                baseline_values.append(float("nan"))

            # post-peak window for spline extrapolation
            j0 = i_peak + 1
            j1 = min(len(y), j0 + self.FIT_TAIL_LEN)
            xs_epoch = []; ys = []; last_t = None
            for k in range(j0, j1):
                yy = y[k]; tt = t_epoch[k]
                if yy == yy and (last_t is None or tt > last_t):
                    xs_epoch.append(tt); ys.append(yy); last_t = tt

            # extrapolate back to refined t0
            if len(xs_epoch) >= self.MIN_FIT_PTS:
                yhat = self._spline_extrapolate_to_left(xs_epoch, ys, t0_epoch)
            elif len(xs_epoch) >= 2:
                x0,x1 = xs_epoch[0], xs_epoch[1]; y0,y1 = ys[0], ys[1]
                yhat = y0 + (y1-y0)/(x1-x0) * (t0_epoch - x0) if x1 != x0 else None
            else:
                yhat = None

            # visuals on lower plot
            # vertical at t0
            vline = pg.InfiniteLine(pos=x_disp_t0, angle=90,
                                    pen=pg.mkPen((255, 140, 200), width=1.5, style=Qt.PenStyle.DashLine))
            self.plot_w.addItem(vline); self._add_ann_w(vline)

            # used post-peak points (cyan)
            def epoch_to_disp(xs):
                if xmode == 0:
                    e0 = t_epoch[0]; el0 = t_elapsed[0]
                    return [ (xe - e0) + el0 for xe in xs ]
                else:
                    return xs
            xs_disp = epoch_to_disp(xs_epoch)
            used_pts = pg.ScatterPlotItem(xs_disp, ys, size=7,
                                          brush=pg.mkBrush(80, 220, 255, 220),
                                          pen=pg.mkPen(255, 255, 255, 150))
            self.plot_w.addItem(used_pts); self._add_ann_w(used_pts)

            # spline curve across used window (orange)
            if len(xs_epoch) >= self.MIN_FIT_PTS:
                a,b,c,d = self._spline_coeffs(xs_epoch, ys)
                x_start = xs_epoch[0]; x_end = xs_epoch[-1]
                Nvis = 200
                xs_vis_epoch = [x_start + (x_end - x_start)*k/(Nvis-1) for k in range(Nvis)]
                ys_vis = []
                for xv in xs_vis_epoch:
                    i = len(xs_epoch)-2
                    for ii in range(len(xs_epoch)-1):
                        if xs_epoch[ii] <= xv <= xs_epoch[ii+1]:
                            i = ii; break
                    ys_vis.append(self._eval_spline_interval(xs_epoch, a,b,c,d, i, xv))
                xs_vis_disp = epoch_to_disp(xs_vis_epoch)
                spline_curve = pg.PlotDataItem(xs_vis_disp, ys_vis, pen=pg.mkPen(255, 140, 0, width=2))
                self.plot_w.addItem(spline_curve); self._add_ann_w(spline_curve)

            # red peak dot
            if yhat is not None and yhat == yhat:
                peak_marker = pg.ScatterPlotItem([x_disp_t0], [yhat], size=10,
                                                 brush=pg.mkBrush(255, 70, 70, 230),
                                                 pen=pg.mkPen(255, 255, 255, 220))
                self.plot_w.addItem(peak_marker); self._add_ann_w(peak_marker)
                peak_text = f"ŷ(t0)= {yhat:.6g} W ({watt_to_dbm(yhat):.2f} dBm)"
            else:
                peak_text = "ŷ(t0)= n/a"

            # info per-peak
            if base_mean == base_mean:
                lines.append(
                    f"Peak {pnum}: {peak_text}; "
                    f"baseline_pre = {base_mean:.6g} W (std {base_std:.3g}, rel {base_rel*100:.2f}%), "
                    f"t0 elapsed {t_elapsed[i_t0]:.6f} s | local {self._fmt_local_time(t0_epoch)}"
                )
            else:
                lines.append(
                    f"Peak {pnum}: {a if False else peak_text}; baseline_pre = n/a, "
                    f"t0 elapsed {t_elapsed[i_t0]:.6f} s | local {self._fmt_local_time(t0_epoch)}"
                )

        # overall baseline max
        finite_bases = [v for v in baseline_values if v == v]
        if finite_bases:
            base_max = max(finite_bases)
            lines.append(f"\nMax baseline (of the two pre-rise ranges): {base_max:.6g} W ({watt_to_dbm(base_max):.2f} dBm)")
        else:
            lines.append("\nMax baseline: n/a")

        QMessageBox.information(self, "Spline peak & baseline results", "\n".join(lines))

    # ---------- NEW: linear fit on sim_dBm from peaks ----------
    def on_fit_dbm_from_peaks(self):
        if not self.data or "sim_dBm" not in self.data or not self.data["sim_dBm"]:
            QMessageBox.information(self, "dBm fits", "No sim_dBm data loaded.")
            return

        # clear only top-plot annotations (keep lower-plot ones if any)
        for it in self._anno_items_dbm:
            try: self.plot_dbm.removeItem(it)
            except Exception: pass
        self._anno_items_dbm.clear()

        t_epoch = self.data["time_epoch_s"]
        t_elapsed = self.data["t_elapsed_s"]
        sim_w = self.data.get("sim_W", [])
        sim_dbm = self.data["sim_dBm"]

        if len(sim_dbm) < 3 or not sim_w:
            QMessageBox.information(self, "dBm fits", "Not enough data.")
            return

        # Reuse peak indices from sim_W (features are distinct there)
        peak_idxs = self._find_first_two_peak_indices(sim_w)
        if not peak_idxs:
            QMessageBox.information(self, "dBm fits", "No peaks found in sim_W.")
            return

        xmode = self.combo_x.currentIndex()  # 0 elapsed, 1 local(epoch)
        results = []
        K = 10.0 / math.log(10.0)  # 10 / ln(10) for slope→tau conversion

        for pnum, i_peak in enumerate(peak_idxs[:2], start=1):
            # take first N points starting at the peak index (finite dBm)
            i0 = i_peak
            i1 = min(len(sim_dbm), i0 + self.DBM_FIT_POINTS)
            xs_t = []
            ys_dBm = []
            for i in range(i0, i1):
                yv = sim_dbm[i]
                if yv == yv and abs(yv) != float("inf"):
                    xs_t.append(t_epoch[i])
                    ys_dBm.append(yv)

            if len(xs_t) < self.MIN_DBM_FIT_PTS:
                results.append(f"Peak {pnum}: not enough dBm points to fit ({len(xs_t)}<{self.MIN_DBM_FIT_PTS}).")
                continue

            # linear regression on (t - tref)
            tref = xs_t[0]
            X = [xt - tref for xt in xs_t]
            Y = ys_dBm
            n = len(X)
            sx = sum(X); sy = sum(Y)
            sxx = sum(x*x for x in X); sxy = sum(x*y for x,y in zip(X,Y))
            denom = n*sxx - sx*sx
            if denom == 0:
                results.append(f"Peak {pnum}: degenerate fit.")
                continue
            m = (n*sxy - sx*sy)/denom         # slope [dBm/s]
            b = (sy - m*sx)/n                  # intercept at t=tref, i.e., Y(0)
            # convert slope to time constant for power decay
            tau = -K / m if m != 0 else float("inf")

            # plot used points in top plot
            def epoch_to_disp(xs):
                if xmode == 0:
                    e0 = t_epoch[0]; el0 = t_elapsed[0]
                    return [ (xe - e0) + el0 for xe in xs ]
                else:
                    return xs

            xs_disp = epoch_to_disp(xs_t)
            used_pts = pg.ScatterPlotItem(xs_disp, ys_dBm, size=9, symbol='s',
                                          brush=pg.mkBrush(255, 100, 255, 220),
                                          pen=pg.mkPen(255, 255, 255, 200))
            self.plot_dbm.addItem(used_pts); self._add_ann_dbm(used_pts)

            # fitted line (cover exactly the span of used points)
            x0_disp, x1_disp = xs_disp[0], xs_disp[-1]
            # evaluate line at those bounds in display x-domain
            y0 = b + m * (xs_t[0] - tref)  # = b
            y1 = b + m * (xs_t[-1] - tref)
            fit_line = pg.PlotDataItem([x0_disp, x1_disp], [y0, y1],
                                       pen=pg.mkPen(255, 100, 255, width=2))
            self.plot_dbm.addItem(fit_line); self._add_ann_dbm(fit_line)

            # report
            tau_ms = tau * 1e3 if (tau == tau and abs(tau) != float("inf")) else float("nan")
            msg = (
                f"Peak {pnum} (sim_dBm fit): slope = {m:+.4f} dBm/s, "
                f"dBm@t_peak = {b:.3f} dBm, "
                f"τ = {tau:.6g} s ({tau_ms:.3g} ms)"
            )
            # warn if slope non-negative
            if m >= 0:
                msg += "  [warning: non-decaying slope]"
            results.append(msg)

        QMessageBox.information(self, "sim_dBm linear fits from peaks", "\n".join(results))

    # ----------------------------------
    def _fmt_local_time(self, t_epoch: float) -> str:
        try:
            import time
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_epoch))
        except Exception:
            return f"{t_epoch:.3f}"


def main():
    app = QApplication(sys.argv)
    w = CsvPowerViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
