"""Glue between the PySide6 GUI and the fractalmaker engine.

The fast part of the original is a C++ extension (``superfastcode``, built with Visual Studio for Windows) that evaluates a formula
given as two small matrices. ``sources/fractals.py`` imports it at the top, so nothing in ``sources`` can be imported on another machine.
This module re-implements the extension's formula evaluation with torch (CPU or CUDA), decodes the formula tensors the same way
``formulainterpreter.py`` does, and ports the exploration and search code that sat on top of it.

Orientation follows the original: the returned arrays are indexed [i, j] with i along the real axis and j along the imaginary axis.
"""
import math
import random
from pathlib import Path

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_LENGTH = 7 + 7 * 5            # xrange(2), yrange(2), bias(2), exponent, then 5 formula rows of 7 numbers
OPERATORS = {1: "+", 2: "-", 3: "*", 4: "/", 5: "^"}
FUNCTIONS = {1: "sin", 2: "cos", 3: "tan", 4: "asin", 5: "acos", 6: "atan", 7: "sinh", 8: "cosh", 9: "tanh", 10: "asinh",
             11: "acosh", 12: "atanh", 13: "log", 14: "log10", 15: "exp", 16: "sqrt", 17: "abs", 18: "conj", 19: "square", 20: "identity"}
VARIABLES = {1: "c", 2: "-c", 3: "z", 4: "-z", 5: "c+z", 6: "c-z", 7: "c*z", 8: "c/z", 9: "-c*z", 10: "-c/z", 11: "c^z", 12: "z^c",
             13: "c^-z", 14: "z^-c", 15: "z*z", 16: "-(z*z)", 17: "c*c", 18: "-(c*c)", 19: "(-z)(-z)", 20: "-((-z)(-z))",
             21: "(-c)(-c)", 22: "-((-c)(-c))"}
COLORMAPS = ['gist_stern', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gist_earth', 'terrain', 'nipy_spectral', 'turbo',
             'twilight', 'twilight_shifted', 'hsv', 'prism', 'ocean', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
             'gist_rainbow', 'rainbow', 'jet', 'gist_ncar', 'binary', 'bone', 'copper', 'Spectral', 'coolwarm']


# ---------------------------------------------------------------- formula tensors (formulainterpreter.py)
def float_to_int(value, divisor):
    """Number of grid points i*divisor, i in [0,1), that are <= value: 1..N (the engine's quirky bucket index)."""
    n = 0
    for i in np.arange(0, 1, divisor):
        if i > value:
            return n
        n += 1
    return n


def decode(formula, bias_free=True):
    """35 numbers in [0,1] -> (integer matrix rows [variable, function, operator], coefficient matrix mapped to [-1, 1])."""
    t = np.asarray(formula, dtype=np.float64).reshape(-1, 7)
    ints = np.zeros((t.shape[0], 3), dtype=np.int64)
    for n, row in enumerate(t):
        ints[n] = [float_to_int(row[2], 1 / 22), float_to_int(row[1], 1 / 20), float_to_int(row[0], 1 / 5)]
    coef = (t[:, 3:7] - 0.5) * 2
    return ints, coef


def describe(formula):
    ints, coef = decode(formula)
    lines = []
    for r in range(len(ints)):
        var, fn, op = ints[r]
        a = coef[r]
        lines.append(f"{OPERATORS.get(op, '?')}  {a[1]:+.2f}*{FUNCTIONS.get(fn, '?')}( {a[0]:+.2f}*[{VARIABLES.get(var, '?')}] {a[2]:+.2f} ) {a[3]:+.2f}")
    return lines


def random_formula(rows=5):
    return torch.rand(rows * 7, dtype=torch.double).numpy()


# ---------------------------------------------------------------- the C++ evaluation, vectorised
def _variable(k, c, z):
    one = torch.ones_like(c)
    table = {1: c, 2: -c, 3: z, 4: -z, 5: c + z, 6: c - z, 7: c * z,
             8: torch.where(z == 0, one, c / z), 9: -c * z, 10: torch.where(z == 0, one, -c / z),
             11: torch.pow(c, z), 12: torch.pow(z, c), 13: torch.pow(c, -z), 14: torch.pow(z, -c),
             15: z * z, 16: -(z * z), 17: c * c, 18: -(c * c), 19: (-z) * (-z), 20: -((-z) * (-z)),
             21: (-c) * (-c), 22: -((-c) * (-c))}
    return table[int(k)]


def _function(k, w):
    k = int(k)
    if k == 1: return torch.sin(w)
    if k == 2: return torch.cos(w)
    if k == 3: return torch.tan(w)
    if k == 4: return torch.asin(w)
    if k == 5: return torch.acos(w)
    if k == 6: return torch.atan(w)
    if k == 7: return torch.sinh(w)
    if k == 8: return torch.cosh(w)
    if k == 9: return torch.tanh(w)
    if k == 10: return torch.asinh(w)
    if k == 11: return torch.acosh(w)
    if k == 12: return torch.atanh(w)
    if k == 13: return torch.log(w)
    if k == 14: return torch.log10(w)
    if k == 15: return torch.exp(w)
    if k == 16: return torch.sqrt(w)
    if k == 17: return torch.abs(w).to(w.dtype)
    if k == 18: return torch.conj(w)
    if k == 19: return w * w
    return w


def _operator(k, tempz, z):
    zero = (tempz == 0)
    k = int(k)
    if k == 1: return z + tempz
    if k == 2: return z - tempz
    if k == 3: return torch.where(zero, z + tempz, z * tempz)
    if k == 4: return torch.where(zero, z - tempz, z / torch.where(zero, torch.ones_like(tempz), tempz))
    return torch.where(zero, torch.full_like(tempz, -1), torch.pow(z, tempz))


def escape_matrix(width, height, xmin, xmax, ymin, ymax, formula, iterations=100, double=True, device=None):
    """Iteration at which |sum of the formula's terms|^2 first exceeds 4, for every c (uint8, indexed [real, imag])."""
    device = device or DEVICE
    dtype = torch.complex128 if double else torch.complex64
    ftype = torch.float64 if double else torch.float32
    ints, coef = decode(formula)
    coef_t = torch.tensor(coef, dtype=ftype, device=device)
    xs = torch.linspace(0, 1, width + 1, dtype=ftype, device=device)[:-1] * (xmax - xmin) + xmin
    ys = torch.linspace(0, 1, height + 1, dtype=ftype, device=device)[:-1] * (ymax - ymin) + ymin
    c = torch.complex(xs.view(-1, 1).expand(width, height), ys.view(1, -1).expand(width, height)).flatten().to(dtype)
    n = c.numel()
    result = torch.full((n,), iterations, dtype=torch.int32, device=device)
    active = torch.arange(n, device=device)
    buffer = torch.zeros(n, dtype=dtype, device=device)
    lastz = torch.zeros(n, dtype=dtype, device=device)
    rows = len(ints)
    with torch.no_grad():
        for j in range(iterations):
            if active.numel() == 0:
                break
            ca, za, ba = c[active], lastz[active], buffer[active]
            acc = torch.zeros_like(ca)
            for i in range(rows):
                var, fn, op = ints[i]
                a = coef_t[i]
                t = _variable(var, ca, za) * a[0] + a[2]
                t = _function(fn, t) * a[1] + a[3]
                if op in (3, 4, 5):
                    ba = _operator(op, t, ba)
                else:
                    acc = acc + ba
                    ba = _operator(op, t, torch.zeros_like(ba))
            buffer[active] = ba
            lastz[active] = acc
            escaped = (acc.real ** 2 + acc.imag ** 2) > 4.0
            result[active[escaped]] = j
            active = active[~escaped]
    return (result.cpu().numpy() % 256).astype(np.uint8).reshape(width, height)


def entropy(matrix):
    """Shannon entropy (bits) of the distribution of iteration counts (Fractal.entropy)."""
    _, counts = np.unique(matrix.ravel(), return_counts=True)
    p = counts / matrix.size
    return float(-(p * np.log2(p)).sum())


def fractal_entropy(formula, xrange_, yrange_, resolution, iterations=40, halves=-1, center_too=False, double=True):
    """Fractal.getfractalentropy: entropy of the iteration image, optionally of one half only (the engine's 'assymetric'),
    optionally adding the entropy of the central quarter."""
    m = escape_matrix(resolution, resolution, xrange_[0], xrange_[1], yrange_[0], yrange_[1], formula, iterations, double)
    if halves == 0:
        m = m[:, m.shape[1] // 2:]
    elif halves == 1:
        m = m[m.shape[0] // 2:, :]
    e = entropy(m)
    if center_too:
        full = escape_matrix(resolution, resolution, xrange_[0], xrange_[1], yrange_[0], yrange_[1], formula, iterations, double)
        c, mg = resolution // 2, resolution // 4
        e += entropy(full[mg:c + mg, mg:c + mg])
    return e


def to_rgb(matrix, iterations, cmap="gist_stern"):
    from matplotlib import colormaps
    arr = matrix.astype(np.float64) / max(iterations, 1)
    return np.ascontiguousarray((colormaps.get_cmap(cmap)(arr)[..., :3] * 255).astype(np.uint8))


# ---------------------------------------------------------------- weight files
def load_weights(path):
    """A '<id>.pt' file: [xmin, xmax, ymin, ymax, bias_re, bias_im, exponent] + 35 formula numbers."""
    w = torch.load(path, map_location="cpu").double().numpy()
    return dict(xrange=(float(w[0]), float(w[1])), yrange=(float(w[2]), float(w[3])), formula=w[7:7 + 42].copy())


def save_weights(path, xrange_, yrange_, formula):
    t = torch.zeros(WEIGHT_LENGTH, dtype=torch.double)
    t[0], t[1], t[2], t[3] = xrange_[0], xrange_[1], yrange_[0], yrange_[1]
    t[7:7 + len(formula)] = torch.as_tensor(np.asarray(formula), dtype=torch.double)
    torch.save(t, path)


# ---------------------------------------------------------------- genetic search (genetics.getgeneticfitfrommodel, pygad replaced)
def evolve_formula(population=60, generations=30, resolution=16, iterations=40, halves=-1, mutation_chance=0.2, job=None, log=print,
                   on_best=None, rows=5, seed=None):
    """Steady-state GA over the 35-number formula vector with the entropy of a tiny render as fitness.

    pygad (used by the original) is not needed: parents are picked by fitness, children by scattered crossover, and the
    worst individuals are replaced. The original mutation function only acted 1 % of the time (``mutation_chance=0.01``).
    """
    rng = np.random.default_rng(seed)
    genes = rows * 7
    pop = rng.random((population, genes))

    def fitness(v):
        return fractal_entropy(v, (-2, 2), (-2, 2), resolution, iterations, halves)
    fit = np.array([fitness(v) for v in pop])
    for g in range(generations):
        if job is not None and job.should_stop():
            break
        order = np.argsort(-fit)
        p1, p2 = pop[order[0]], pop[order[1 + rng.integers(0, max(2, population // 4))]]
        for _ in range(max(2, population // 6)):
            mask = rng.random(genes) < 0.5
            child = np.where(mask, p1, p2)
            if rng.random() < mutation_chance:
                k = rng.integers(0, genes)
                child[k] = np.clip(child[k] + (0.5 - rng.random()) / 5, 0, 1)
            worst = int(np.argmin(fit))
            pop[worst], fit[worst] = child, fitness(child)
        best = int(np.argmax(fit))
        log(f"generation {g + 1}: best entropy {fit[best]:.4f}")
        if on_best:
            on_best(pop[best].copy(), float(fit[best]))
    best = int(np.argmax(fit))
    return pop[best].copy(), float(fit[best])


def optimize_zone(formula, center, half_width, zoom_factor=0.5, generations=8, population=120, resolution=64, iterations=100,
                  job=None, log=print):
    """genetics.optimizefractalrange / fractalexploration.findhientropyzonecenter: search the centre of a zoomed window
    (half-width * zoom_factor) inside the current window that maximises the entropy of itself plus its central quarter."""
    rng = np.random.default_rng()
    hw = half_width * zoom_factor
    pop = np.stack([center[0] + (rng.random(population) * 2 - 1) * half_width,
                    center[1] + (rng.random(population) * 2 - 1) * half_width], axis=1)

    def fitness(p):
        return fractal_entropy(formula, (p[0] - hw, p[0] + hw), (p[1] - hw, p[1] + hw), resolution, iterations, center_too=True)
    fit = np.array([fitness(p) for p in pop])
    for g in range(generations):
        if job is not None and job.should_stop():
            break
        order = np.argsort(-fit)
        parents = pop[order[:4]]
        for _ in range(population // 4):
            child = np.where(rng.random(2) < 0.5, parents[rng.integers(0, 4)], parents[rng.integers(0, 4)]).copy()
            if rng.random() < 0.8:
                child[rng.integers(0, 2)] += (0.5 - rng.standard_normal()) * hw
            worst = int(np.argmin(fit))
            pop[worst], fit[worst] = child, fitness(child)
        log(f"generation {g + 1}: best entropy {fit.max():.4f}")
    best = int(np.argmax(fit))
    return tuple(pop[best]), hw, float(fit[best])


# ---------------------------------------------------------------- zoom animation (fractalexploration.py)
def interpolation_radius(start, end, t):
    """AnimationInterpolator.animatorfromnormalizedvalue: linear radius times (1 - t^0.1)."""
    return (start - (start - end) * t) * (1 - t ** 0.1)


def zoom_frames(formula, center, start_radius, zoom_factor, count, resolution=400, iterations=100, cmap="gist_stern", double=True,
                job=None, log=print):
    end = start_radius * zoom_factor
    frames = []
    for i in range(count):
        if job is not None and job.should_stop():
            break
        r = max(interpolation_radius(start_radius, end, i / count), 1e-12)
        m = escape_matrix(resolution, resolution, center[0] - r, center[0] + r, center[1] - r, center[1] + r, formula, iterations, double)
        frames.append(to_rgb(m, iterations, cmap))
        log(f"frame {i + 1}/{count}, radius {r:.3e}")
    return frames


def save_gif(frames, path, fps=15):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


# ---------------------------------------------------------------- "gradient descent" zoom (fractalexploration.zoomgradientdescent)
class FractalRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 256, dtype=torch.double)
        self.linear2 = torch.nn.Linear(256, 3, dtype=torch.double)

    def forward(self, x):
        return self.linear2(self.linear(x))


def regression_zoom(formula, candidates=30, resolution=64, iterations=100, job=None, log=print):
    """The original builds the loss with ``torch.tensor(..., requires_grad=True)`` so no gradient reaches the network:
    it is a random search. Here that is what it is: a randomly initialised network maps random inputs to (centre x, centre y, width),
    and the candidate with the best entropy per unit of width wins."""
    model = FractalRegression()
    best, best_loss = None, 1e18
    for k in range(candidates):
        if job is not None and job.should_stop():
            break
        out = model(torch.rand(3, dtype=torch.double) * 8).detach().numpy()
        width = max(abs(float(out[2])), 1e-6)
        half = width / 2
        e = fractal_entropy(formula, (out[0] - half, out[0] + half), (out[1] - half, out[1] + half), resolution, iterations)
        loss = -e / width
        if loss < best_loss:
            best, best_loss = (float(out[0]), float(out[1]), width), loss
            log(f"candidate {k}: centre ({out[0]:.3f}, {out[1]:.3f}) width {width:.3f} entropy {e:.3f}")
    return best


# ---------------------------------------------------------------- neural fractals (DLFractals)
TARGETS = {
    "sigmoid of a product of sin and exp (TrainingFunctions.sigmoidtest1)":
        lambda z: torch.sigmoid(torch.square(torch.sin(z * z + z / (z - 1)) * torch.exp(z * z - z / (z - 1)))) * 4,
    "(sin z + z^2) atan z, cubed and scaled (sigmoidtest2 without the normalisation layer)":
        lambda z: _scale(((torch.sin(z) + torch.square(z)) * torch.atan(z)) ** 3),
    "(cos z + z^2) cos z, cubed (sigmoidtest3)":
        lambda z: _scale(((torch.cos(z) + torch.square(z)) * torch.cos(z)) ** 3),
    "z^2 + 0.3 + 0.5i (a Julia map, to see whether the network can copy it)": lambda z: z * z + (0.3 + 0.5j),
}


def _scale(z, target=4.0):
    m = torch.abs(z).max().clamp(min=1e-9)
    return z / m * target


class MLP(torch.nn.Module):
    """DLFractals.fractalbasedlayers.LinearLayer: (re, im) -> hidden layers with LeakyReLU -> (re, im)."""

    def __init__(self, layers=4, hidden=64):
        super().__init__()
        modules = [torch.nn.Linear(2, hidden, dtype=torch.double), torch.nn.LeakyReLU()]
        for _ in range(layers - 2):
            modules += [torch.nn.Linear(hidden, hidden, dtype=torch.double), torch.nn.LeakyReLU()]
        modules.append(torch.nn.Linear(hidden, 2, dtype=torch.double))
        self.net = torch.nn.Sequential(*modules)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, z):
        out = self.net(torch.stack((z.real, z.imag), dim=-1))
        return torch.complex(out[..., 0], out[..., 1])


def train_neural_fractal(target_name, layers=4, hidden=64, steps=1500, grid=64, lr=0.01, noise=0.001, job=None, log=print, on_step=None):
    """DLFractalsTraining.train_one_epoch: fit the network to a target function of z on a grid (L1 loss, SGD with momentum)."""
    target_fn = TARGETS[target_name]
    model = MLP(layers, hidden).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    axis = torch.linspace(-2, 2, grid, dtype=torch.double, device=DEVICE)
    z0 = torch.complex(axis.view(-1, 1).expand(grid, grid), axis.view(1, -1).expand(grid, grid)).flatten()
    loss_fn = torch.nn.L1Loss()
    for s in range(steps):
        if job is not None and job.should_stop():
            break
        z = z0 + torch.rand(z0.shape, device=DEVICE, dtype=torch.double) * noise
        with torch.no_grad():
            target = torch.nan_to_num(target_fn(z).real) + 1j * torch.nan_to_num(target_fn(z).imag)
        out = model(z)
        loss = loss_fn(torch.view_as_real(out), torch.view_as_real(target))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            log(f"step {s}: loss {loss.item():.4f}")
            if on_step:
                on_step(model, s, loss.item())
    return model


def neural_images(model, resolution=256, iterations=60, zoom=2.0, cmap="gist_stern"):
    """The network's output image (DLFractals.drawfractal) and the escape-time fractal of iterating it (DLFractals.iterateformula)."""
    axis = torch.linspace(-zoom, zoom, resolution, dtype=torch.double, device=DEVICE)
    z0 = torch.complex(axis.view(-1, 1).expand(resolution, resolution), axis.view(1, -1).expand(resolution, resolution)).flatten()
    with torch.no_grad():
        out = model(z0).abs().reshape(resolution, resolution)
        out = ((out - out.min()) / (out.max() - out.min() + 1e-12)).cpu().numpy()
        z = z0.clone()
        result = torch.full((resolution * resolution,), iterations, dtype=torch.int32, device=DEVICE)
        active = torch.arange(resolution * resolution, device=DEVICE)
        zz = z0.clone()
        for it in range(iterations):
            if active.numel() == 0:
                break
            zz_a = model(zz[active])
            zz[active] = zz_a
            esc = zz_a.abs() > 2
            result[active[esc]] = it
            active = active[~esc]
    from matplotlib import colormaps
    img1 = (colormaps.get_cmap(cmap)(out)[..., :3] * 255).astype(np.uint8)
    fr = result.cpu().numpy().reshape(resolution, resolution)
    img2 = (colormaps.get_cmap(cmap)(fr / iterations)[..., :3] * 255).astype(np.uint8)
    return np.ascontiguousarray(img1), np.ascontiguousarray(img2), entropy(fr)
