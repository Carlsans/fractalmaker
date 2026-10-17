"""fractalmaker GUI. Run from the repository root:  python -m gui.app"""
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from gui import bridge  # noqa: E402


_JOBS = []


class Job(QThread):
    log = Signal(str)
    result = Signal(object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        _JOBS.append(self)
        self.fn, self._stop = fn, threading.Event()

    def should_stop(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self))
        except Exception:
            self.failed.emit(traceback.format_exc())


class LogBox(QPlainTextEdit):
    def __init__(self, height=90):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(300)
        self.setMaximumHeight(height)

    def add(self, text):
        self.appendPlainText(text)


class Runner:
    def __init__(self, log, buttons=()):
        self.log, self.buttons, self.job = log, buttons, None

    def start(self, fn, on_done=None, on_result=None):
        if self.job is not None and self.job.isRunning():
            return
        self.job = Job(fn)
        self.job.log.connect(self.log.add)
        self.job.failed.connect(lambda tb: self.log.add(tb.strip().splitlines()[-1]))
        if on_done:
            self.job.finished_ok.connect(on_done)
        if on_result:
            self.job.result.connect(on_result)
        self.job.finished.connect(lambda: [b.setEnabled(True) for b in self.buttons])
        for b in self.buttons:
            b.setEnabled(False)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.stop()


def spin(lo, hi, v):
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(v); return w


def dspin(lo, hi, v, dec=6, step=0.1):
    w = QDoubleSpinBox(); w.setDecimals(dec); w.setRange(lo, hi); w.setSingleStep(step); w.setValue(v); return w


def row(*ws):
    w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
    for x in ws:
        l.addWidget(x)
    return w


class State:
    def __init__(self):
        self.formula = bridge.random_formula()
        self.center, self.half = (0.0, 0.0), 2.0


class ZoomView(QLabel):
    """Shows an image; drag a rectangle to zoom, right-click to go back. Image rows follow the real axis (as in the original)."""
    zoom_to = Signal(float, float, float, float)     # xmin, xmax, ymin, ymax
    back = Signal()

    def __init__(self, size=560):
        super().__init__("no image")
        self.setMinimumSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888")
        self.rgb, self.window_ = None, (-2, 2, -2, 2)
        self.start, self.end = None, None

    def show_rgb(self, rgb, window):
        self.rgb, self.window_ = np.ascontiguousarray(rgb), window
        h, w, _ = self.rgb.shape
        self.setPixmap(QPixmap.fromImage(QImage(self.rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _image_rect(self):
        pm = self.pixmap()
        if pm is None:
            return QRect()
        r = pm.rect(); r.moveCenter(self.rect().center()); return r

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.back.emit(); return
        self.start = self.end = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self.start is not None:
            self.end = e.position().toPoint(); self.update()

    def mouseReleaseEvent(self, e):
        if self.start is None or self.rgb is None:
            self.start = None; return
        r = self._image_rect()
        a, b = self.start, e.position().toPoint()
        self.start = self.end = None
        self.update()
        if r.width() < 2 or abs(a.x() - b.x()) < 4 or abs(a.y() - b.y()) < 4:
            return
        xmin, xmax, ymin, ymax = self.window_

        def to_world(p):
            u = (min(max(p.x() - r.left(), 0), r.width())) / r.width()      # across the image: imaginary axis
            v = (min(max(p.y() - r.top(), 0), r.height())) / r.height()     # down the image: real axis
            return xmin + v * (xmax - xmin), ymin + u * (ymax - ymin)
        (x1, y1), (x2, y2) = to_world(a), to_world(b)
        self.zoom_to.emit(min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.start is not None and self.end is not None:
            p = QPainter(self)
            p.setPen(QPen(Qt.white, 1, Qt.DashLine))
            p.drawRect(QRect(self.start, self.end).normalized())


class ExploreTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state, self.history, self.matrix = state, [], None
        self.view = ZoomView()
        self.view.zoom_to.connect(self.on_zoom)
        self.view.back.connect(self.go_back)
        self.res = spin(32, 4000, 500); self.iters = spin(5, 255, 100)
        self.double = QCheckBox("Double precision (needed for deep zooms)"); self.double.setChecked(True)
        self.cmap = QComboBox(); self.cmap.addItems(bridge.COLORMAPS); self.cmap.currentTextChanged.connect(lambda _: self.draw())
        self.xmin, self.xmax = dspin(-1e6, 1e6, -2, 12), dspin(-1e6, 1e6, 2, 12)
        self.ymin, self.ymax = dspin(-1e6, 1e6, -2, 12), dspin(-1e6, 1e6, 2, 12)
        self.table = QTableWidget(5, 7)
        self.table.setHorizontalHeaderLabels(["operator", "function", "variable", "coef 1", "coef 2", "coef 3", "coef 4"])
        self.table.setMaximumHeight(210)
        self.meaning = QPlainTextEdit(); self.meaning.setReadOnly(True); self.meaning.setMaximumHeight(120)
        b_render = QPushButton("Render"); b_render.clicked.connect(self.render)
        b_rand = QPushButton("Random formula"); b_rand.clicked.connect(self.randomize)
        b_open = QPushButton("Open .pt"); b_open.clicked.connect(self.open_pt)
        b_save = QPushButton("Save .pt"); b_save.clicked.connect(self.save_pt)
        b_png = QPushButton("Save PNG"); b_png.clicked.connect(self.save_png)
        b_back = QPushButton("Back"); b_back.clicked.connect(self.go_back)
        self.info = QLabel(""); self.log = LogBox(60)
        self.runner = Runner(self.log, (b_render,))
        form = QFormLayout()
        form.addRow("Resolution", self.res); form.addRow("Iterations (max 255)", self.iters); form.addRow(self.double)
        form.addRow("Real from / to", row(self.xmin, self.xmax)); form.addRow("Imaginary from / to", row(self.ymin, self.ymax))
        form.addRow("Colormap", self.cmap)
        left = QVBoxLayout(); left.addLayout(form)
        left.addWidget(QLabel("Formula: 5 terms of 7 numbers in [0, 1]. Each term is  operator  coef2*function( coef1*[variable] + coef3 ) + coef4")); left.addWidget(self.table)
        left.addWidget(self.meaning); left.addWidget(row(b_render, b_rand, b_open, b_save, b_png, b_back)); left.addWidget(self.info)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addWidget(self.view, 1)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(QLabel("Drag a rectangle on the picture to zoom, right-click to go back.")); lay.addWidget(self.log)
        self.fill_table(); self.update_meaning()
        self.table.itemChanged.connect(lambda _: self.update_meaning())

    def fill_table(self):
        self.table.blockSignals(True)
        t = np.asarray(self.state.formula).reshape(-1, 7)
        for r in range(5):
            for c in range(7):
                self.table.setItem(r, c, QTableWidgetItem(f"{t[r, c]:.4f}"))
        self.table.blockSignals(False)

    def read_table(self):
        vals = [float(self.table.item(r, c).text()) for r in range(5) for c in range(7)]
        self.state.formula = np.clip(np.array(vals), 0, 1)

    def update_meaning(self):
        try:
            self.read_table()
            self.meaning.setPlainText("\n".join(bridge.describe(self.state.formula)))
        except ValueError:
            pass

    def randomize(self):
        self.state.formula = bridge.random_formula(); self.fill_table(); self.update_meaning(); self.render()

    def window(self):
        return (self.xmin.value(), self.xmax.value(), self.ymin.value(), self.ymax.value())

    def set_window(self, w):
        for spin_, v in zip((self.xmin, self.xmax, self.ymin, self.ymax), w):
            spin_.setValue(v)

    def render(self, push=True):
        self.read_table()
        w, res, iters, dbl, formula = self.window(), self.res.value(), self.iters.value(), self.double.isChecked(), self.state.formula.copy()
        if push:
            self.history.append(w)

        def work(job):
            import time
            t = time.time()
            m = bridge.escape_matrix(res, res, w[0], w[1], w[2], w[3], formula, iters, dbl)
            return m, time.time() - t

        def done(result):
            self.matrix, secs = result
            self.state.center = ((w[0] + w[1]) / 2, (w[2] + w[3]) / 2)
            self.state.half = (w[1] - w[0]) / 2
            self.info.setText(f"{res} x {res} in {secs:.2f} s on {bridge.DEVICE}, entropy {bridge.entropy(self.matrix):.3f} bits")
            self.draw()
        self.runner.start(work, done)

    def draw(self):
        if self.matrix is not None:
            self.view.show_rgb(bridge.to_rgb(self.matrix, self.iters.value(), self.cmap.currentText()), self.window())

    def on_zoom(self, x0, x1, y0, y1):
        self.set_window((x0, x1, y0, y1)); self.render()

    def go_back(self):
        if len(self.history) > 1:
            self.history.pop()
            self.set_window(self.history.pop()); self.render()

    def open_pt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Weights", "", "PyTorch (*.pt)")
        if path:
            try:
                w = bridge.load_weights(path)
            except Exception as exc:
                self.log.add(f"could not read {path}: {exc} (older files use a different formula format)"); return
            self.state.formula = w["formula"]; self.fill_table(); self.update_meaning()
            self.set_window((w["xrange"][0], w["xrange"][1], w["yrange"][0], w["yrange"][1])); self.render()

    def save_pt(self):
        path, _ = QFileDialog.getSaveFileName(self, "Weights", "fractal.pt", "PyTorch (*.pt)")
        if path:
            self.read_table()
            w = self.window(); bridge.save_weights(path, (w[0], w[1]), (w[2], w[3]), self.state.formula)

    def save_png(self):
        if self.view.rgb is not None:
            path, _ = QFileDialog.getSaveFileName(self, "PNG", "fractal.png", "PNG (*.png)")
            if path:
                from matplotlib import pyplot as plt
                plt.imsave(path, self.view.rgb)


class SearchTab(QWidget):
    def __init__(self, state, explore):
        super().__init__()
        self.state, self.explore, self.best = state, explore, None
        self.pop = spin(10, 2000, 60); self.gens = spin(1, 5000, 30); self.res = spin(8, 128, 16); self.iters = spin(5, 255, 40)
        self.halves = QComboBox(); self.halves.addItems(["whole picture", "right half only (assymetric = 0)", "bottom half only (assymetric = 1)"])
        self.mut = dspin(0, 1, 0.2, 2, 0.05)
        self.b_run = QPushButton("Evolve a formula"); b_stop = QPushButton("Stop")
        self.b_run.clicked.connect(self.run); b_stop.clicked.connect(lambda: self.r1.stop())
        b_send = QPushButton("Send best to Explore"); b_send.clicked.connect(self.send)
        self.view = ZoomView(360); self.log = LogBox(140)
        self.r1 = Runner(self.log, (self.b_run,))
        gb = QLabel("Zone search: choose where to zoom");
        self.zoomf = dspin(0.01, 1, 0.5, 2, 0.05); self.zgens = spin(1, 100, 8); self.zpop = spin(10, 1000, 120)
        self.b_zone = QPushButton("Find a high-entropy zone of the current window"); self.b_zone.clicked.connect(self.zone)
        self.b_reg = QPushButton("Legacy: 'gradient descent' zoom (really a random network search)"); self.b_reg.clicked.connect(self.regression)
        self.r2 = Runner(self.log, (self.b_zone, self.b_reg))
        form = QFormLayout()
        form.addRow("Population", self.pop); form.addRow("Generations", self.gens); form.addRow("Fitness resolution", self.res)
        form.addRow("Iterations", self.iters); form.addRow("Entropy of", self.halves)
        form.addRow("Mutation chance (the original used 0.01)", self.mut)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(row(self.b_run, b_stop, b_send))
        left.addWidget(gb)
        zf = QFormLayout(); zf.addRow("Zoom factor", self.zoomf); zf.addRow("Generations", self.zgens); zf.addRow("Population", self.zpop)
        left.addLayout(zf); left.addWidget(self.b_zone); left.addWidget(self.b_reg)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addWidget(self.view, 1)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)

    def run(self):
        p = dict(population=self.pop.value(), generations=self.gens.value(), resolution=self.res.value(), iterations=self.iters.value(),
                 halves=self.halves.currentIndex() - 1, mutation_chance=self.mut.value())

        def work(job):
            def on_best(v, f):
                self.best = v
                job.result.emit(v)
            return bridge.evolve_formula(job=job, log=job.log.emit, on_best=on_best, **p)

        def preview(v):
            m = bridge.escape_matrix(256, 256, -2, 2, -2, 2, v, 60)
            self.view.show_rgb(bridge.to_rgb(m, 60), (-2, 2, -2, 2))
        self.r1.start(work, lambda r: (setattr(self, "best", r[0]), self.log.add(f"finished: entropy {r[1]:.4f}")), preview)

    def send(self):
        if self.best is not None:
            self.state.formula = self.best
            self.explore.fill_table(); self.explore.update_meaning(); self.explore.set_window((-2, 2, -2, 2)); self.explore.render()

    def zone(self):
        self.explore.read_table()
        formula, c, hw, zf = self.state.formula.copy(), self.state.center, self.state.half, self.zoomf.value()
        g, p = self.zgens.value(), self.zpop.value()
        iters = self.explore.iters.value()

        def work(job):
            return bridge.optimize_zone(formula, c, hw, zf, g, p, 64, iters, job, job.log.emit)

        def done(r):
            (cx, cy), half, e = r
            self.explore.set_window((cx - half, cx + half, cy - half, cy + half)); self.explore.render()
            self.log.add(f"zoomed to ({cx:.5f}, {cy:.5f}) half-width {half:.3g}, entropy {e:.3f}")
        self.r2.start(work, done)

    def regression(self):
        self.explore.read_table()
        formula, iters = self.state.formula.copy(), self.explore.iters.value()

        def done(r):
            if r is None:
                return
            cx, cy, width = r
            self.explore.set_window((cx - width / 2, cx + width / 2, cy - width / 2, cy + width / 2)); self.explore.render()
        self.r2.start(lambda job: bridge.regression_zoom(formula, 30, 64, iters, job, job.log.emit), done)


class AnimateTab(QWidget):
    def __init__(self, state, explore):
        super().__init__()
        self.state, self.explore, self.frames = state, explore, []
        self.count = spin(2, 600, 60); self.res = spin(64, 2000, 300); self.zoom = dspin(1e-15, 1, 0.001, 15, 0.001)
        self.fps = spin(1, 60, 15)
        self.b_run = QPushButton("Render the zoom frames from the Explore window's centre"); self.b_run.clicked.connect(self.run)
        b_stop = QPushButton("Stop"); b_stop.clicked.connect(lambda: self.runner.stop())
        b_gif = QPushButton("Save GIF..."); b_gif.clicked.connect(self.save_gif)
        b_png = QPushButton("Save frames as PNG..."); b_png.clicked.connect(self.save_png)
        self.slider = QSpinBox(); self.slider.valueChanged.connect(self.show_frame)
        self.view = ZoomView(360); self.log = LogBox(100)
        self.runner = Runner(self.log, (self.b_run,))
        form = QFormLayout()
        form.addRow("Frames", self.count); form.addRow("Frame size", self.res); form.addRow("Final radius / start radius", self.zoom); form.addRow("GIF speed (fps)", self.fps)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(self.b_run); left.addWidget(row(b_stop, b_gif, b_png))
        left.addWidget(QLabel("Frame")); left.addWidget(self.slider)
        left.addWidget(QLabel("The radius follows the original interpolation: linear radius x (1 - t^0.1), which falls fast at the start and reaches 0 at the end."))
        top = QHBoxLayout(); top.addLayout(left, 1); top.addWidget(self.view, 1)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)

    def run(self):
        self.explore.read_table()
        formula, c, hw = self.state.formula.copy(), self.state.center, self.state.half
        n, res, zoom, iters, cmap = self.count.value(), self.res.value(), self.zoom.value(), self.explore.iters.value(), self.explore.cmap.currentText()

        def done(frames):
            self.frames = frames
            self.slider.setRange(0, max(len(frames) - 1, 0)); self.show_frame()
        self.runner.start(lambda job: bridge.zoom_frames(formula, c, hw, zoom, n, res, iters, cmap, True, job, job.log.emit), done)

    def show_frame(self):
        if self.frames:
            self.view.show_rgb(self.frames[min(self.slider.value(), len(self.frames) - 1)], (0, 1, 0, 1))

    def save_gif(self):
        if self.frames:
            path, _ = QFileDialog.getSaveFileName(self, "GIF", "zoom.gif", "GIF (*.gif)")
            if path:
                bridge.save_gif(self.frames, path, self.fps.value()); self.log.add(f"wrote {path}")

    def save_png(self):
        if self.frames:
            folder = QFileDialog.getExistingDirectory(self, "Folder")
            if folder:
                from matplotlib import pyplot as plt
                for i, f in enumerate(self.frames):
                    plt.imsave(Path(folder) / f"zoomframe{i:04d}.png", f)
                self.log.add(f"wrote {len(self.frames)} frames")


class NeuralTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.target = QComboBox(); self.target.addItems(list(bridge.TARGETS))
        self.layers = spin(2, 12, 4); self.hidden = spin(8, 512, 64); self.steps = spin(10, 100000, 1500)
        self.lr = dspin(1e-5, 1, 0.01, 5, 0.005); self.res = spin(32, 1024, 256); self.iters = spin(5, 255, 60)
        self.cmap = QComboBox(); self.cmap.addItems(bridge.COLORMAPS)
        self.b_train = QPushButton("Train the network"); b_stop = QPushButton("Stop")
        self.b_train.clicked.connect(self.train); b_stop.clicked.connect(lambda: self.runner.stop())
        b_draw = QPushButton("Draw"); b_draw.clicked.connect(self.draw)
        self.v1, self.v2 = ZoomView(300), ZoomView(300)
        self.info = QLabel(""); self.log = LogBox(100)
        self.runner = Runner(self.log, (self.b_train,))
        form = QFormLayout()
        form.addRow("Target function", self.target); form.addRow("Layers / hidden size", row(self.layers, self.hidden))
        form.addRow("Training steps", self.steps); form.addRow("Learning rate", self.lr)
        form.addRow("Picture size", self.res); form.addRow("Iterations of the fractal", self.iters); form.addRow("Colormap", self.cmap)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(row(self.b_train, b_stop, b_draw)); left.addWidget(self.info)
        left.addWidget(QLabel("A network is fitted to a complex function (L1 loss), then drawn two ways: its output, and the escape-time fractal of iterating it. "
                              "This is the DLFractals experiment; the checkpoints it left (fractalNN*.pt) are not included."))
        left.itemAt(left.count() - 1).widget().setWordWrap(True)
        pics = QHBoxLayout(); pics.addWidget(self.v1); pics.addWidget(self.v2)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addLayout(pics, 2)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)

    def train(self):
        t, L, H, S, lr = self.target.currentText(), self.layers.value(), self.hidden.value(), self.steps.value(), self.lr.value()
        self.runner.start(lambda job: bridge.train_neural_fractal(t, L, H, S, 64, lr, 0.001, job, job.log.emit),
                          lambda m: (setattr(self, "model", m), self.draw()))

    def draw(self):
        if self.model is None:
            self.log.add("train the network first"); return
        a, b, e = bridge.neural_images(self.model, self.res.value(), self.iters.value(), 2.0, self.cmap.currentText())
        self.v1.show_rgb(a, (0, 1, 0, 1)); self.v2.show_rgb(b, (0, 1, 0, 1))
        self.info.setText(f"entropy of the iterated fractal: {e:.3f} bits")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("fractalmaker")
        state = State()
        self.explore = ExploreTab(state)
        tabs = QTabWidget()
        for w, name in ((self.explore, "Explore"), (SearchTab(state, self.explore), "Search"), (AnimateTab(state, self.explore), "Zoom animation"),
                        (NeuralTab(), "Neural fractals (legacy)")):
            tabs.addTab(w, name)
        self.setCentralWidget(tabs)
        self.resize(1250, 800)

    def closeEvent(self, event):
        running = [job for job in _JOBS if job.isRunning()]
        for job in running:
            job.stop()
        # a QThread destroyed while running aborts the process, and terminating one inside CUDA does too
        if not all(job.wait(15000) for job in running):
            import os
            os._exit(0)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
