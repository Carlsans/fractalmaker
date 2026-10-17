# fractalmaker: how it works, what worked, what did not

The oldest of the fractal projects (July 2022 - April 2023, Windows, Python 3.7-3.9). Written from the code and from what it left on disk: 16,001 formula tensors with rating files in
`localtraindir/test10/`, thousands of zoom frames (`fractalsanimations`), about a dozen `fractalNN*.pt` checkpoints (`dlfractaltraining`) and the early samples in `fractals/nn1`.
Statements marked *(inferred)* are conclusions from that evidence, not recorded results.

## The idea

Find beautiful escape-time fractals **automatically** and **zoom into their interesting regions automatically**, learning what "beautiful" means from the author's own ratings. The project grew in layers:

1. **A formula that is a vector of numbers.** A fractal is `5` terms of `7` numbers in [0, 1] (35 numbers), plus the window (`xrange`, `yrange`). A term is `operator  coef2 * function( coef1 * [variable] + coef3 ) + coef4`; the first three numbers pick an operator
   (`+ - * / ^`), a function (20: `sin ... atanh, log, exp, sqrt, abs, conj, square`) and a variable (22 combinations of `c` and `z`, e.g. `c*z`, `z^c`, `-(c*c)`); the last four are coefficients in [-1, 1].
   Because everything is a number in [0, 1], a GA, a gradient method or a neural network can all work on the same vector. (`formulainterpreter.py`)
2. **A fast evaluator.** A C++ extension (pybind11, Visual Studio, `superfastcode`) evaluates the two matrices for every pixel; a multithreaded loader splits the window into blocks. (`ctopython/`, `MultithreadedFractalLoader.py`)
3. **Automatic exploration.** A genetic algorithm (pygad) searches formulas; another searches the *window* (where to zoom) using the **entropy of the iteration-count image** as fitness. (`genetics.py`, `fractalexploration.py`)
4. **Learning from ratings.** A viewer shows random fractals and the author types a 0-9 mark; a regression network learns the mark from the 22-number parameters and then picks the best of many random candidates. (`userinteraction.py`, `training.py`, `models.FractalModel`)
5. **Learning where to zoom.** A convolutional "visual layer" model predicts a good centre from a picture, and an iterator plunges into a fractal step by step, going back when the entropy drops. (`visuallayertrainer.py`, `FractalIterator.py`, `IterationsManager.py`)
6. **Fractals made by networks.** `DLFractals/` trains a network to imitate a complex function and iterates the result. (`DLFractals/`)
7. **A viewer.** A tkinter window with rubber-band zoom and a file-name based launcher for Windows. (`ManualFractalExploration/`, `openfractalbybatchfile.py`)

## What worked

- **A formula as a bounded vector.** Every later experiment (GA, regression, networks) reused it; it is the idea that survived into the later projects (MatrixFractals, RewriteFractal).
- **Entropy of the iteration image as a fitness.** One line (`-sum(p log2 p)` over the histogram of iteration counts) and cheap at 16x16, it rejects empty and uniform pictures. The later projects replaced it with compression size, which is the same idea with spatial structure.
- **Coarse-to-fine evaluation.** Formulas are tested at 16x16 with 40 iterations, zoom zones at 64x64, finals at 1000+ pixels.
- **Moving the inner loop to C++.** It made 1200 x 1200 renders interactive; the GUI's torch/CUDA version reaches the same in about a second.
- **Zoom animations.** `fractalgradationtocenter` with an interpolation radius `r(t) = linear(t) * (1 - t^0.1)` produced the frame sets in `fractalsanimations`. The radius shrinks fast first and ends at 0, which feels like a dive.
- **Entropy-guided zoom.** Searching the centre of the zoomed window with a GA found regions where the picture stays busy instead of collapsing to one colour.

## What did not work

- **The rating model.** In the 16,001-file set that remains, every rating file reads `0`, so the learned preference cannot be recovered from that set; the trained model the viewer loads (`model2.pt`) is not in this repository. Tiny datasets of subjective marks with a 22-2000-1000-1 MLP are noisy *(inferred)*.
- **"Gradient descent" zoom.** `zoomgradientdescent` builds its loss with `torch.tensor(..., requires_grad=True)` from a Python float, so no gradient reaches the network. It is a random search that happens to pick the last candidate (the best-loss test starts at 0 and the losses are negative).
- **pygad parameters.** The formula GA mutates only 1 % of the time (`if np.random.random() > .01: return`), and `allow_duplicate_genes=False` on continuous genes does little.
- **Neural fractals.** `DLFractals` tries a dozen network designs (transposed convolutions, attention, complex linear layers, composite models). The checkpoints kept (`fractalNN - absexpsquare`, `asin(sqr(x))+cos(x)`, `atan(sqr(x))+x`, `fractalNN19`) suggest the networks were trained to imitate chosen formulas rather than to discover fractals *(inferred)*. Iterating a smooth network rarely produces fine structure.
- **Platform lock-in.** Hard-coded host names and `G:\` paths in `globalvariables.py`, `winsound`, `powershell -c` to display images, `ImageGlass`, and the C++ extension built only with Visual Studio. Nothing in `sources/` imports without `superfastcode`.
- **Iterations stored as `unsigned char`.** The evaluator returns counts modulo 256.
- **Deprecated parameters** (`fractalbias`, `fractalexponential`) are still stored in the weight tensor.
- **Unrelated code in the same folder** (a phone-number scraper and a Stable Diffusion test) was left out of this repository, and a credential found in one of them was not published.

## Lessons

1. Put the whole fractal in one bounded vector; every search method then works unchanged.
2. Entropy of the iteration histogram is a good first interestingness score, but it ignores spatial structure.
3. A loss built from a detached float does not train anything: check that a gradient reaches the weights.
4. Keep paths and host names out of code.
5. Rendering through a compiled extension is worth it only until the GPU framework does the same job.

## Running the original code

Windows, with `superfastcode` built from `sources/ctopython/superfastcode` (Visual Studio, `python setup.py build`), PyTorch, pygad, matplotlib, tkinter, and your machine's name added to `sources/globalvariables.py`.
Open a fractal by dragging a numbered `.pt` onto `openfractal.bat`.

## About the GUI's formula evaluator

`gui/bridge.py` re-implements the extension with torch, following `module2.cpp` closely, including its quirks: the term buffer persists between iterations, the last term is not added to the sum, a zero argument changes `*` and `/` into `+` and `-`, and counts wrap at 256.
It was checked against a scalar transcription of the C++; they agree except exactly on branch cuts of complex `atan`/`asin` (for example when `c` is purely imaginary). No stored image of a 35-number formula was available for a pixel-level comparison.
