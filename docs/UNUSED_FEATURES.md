# Features that exist in the code but are not part of the final workflow

| # | Feature | Where in the code | Why unused | In the GUI |
|---|---|---|---|---|
| 1 | Formula GA | `genetics.getgeneticfitfrommodel` | Needed pygad and the rating model | Search > "Evolve a formula" (own GA; entropy of the whole picture or of one half; mutation chance adjustable) |
| 2 | Zone search | `genetics.optimizefractalrange`, `fractalexploration.findhientropyzonecenter` | Slow in Python | Search > "Find a high-entropy zone" |
| 3 | "Gradient descent" zoom | `fractalexploration.zoomgradientdescent` | Never trained (see the notes) | Search > legacy button, written for what it does: a random-network search |
| 4 | Zoom animation | `fractalexploration.fractalgradationtocenter`, `AnimationInterpolator` | Script | Zoom animation tab, GIF and PNG output |
| 5 | Asymmetric entropy | `Fractal.getfractalentropy(assymetric=...)` | Used to prefer fractals that are not symmetric | Search > "Entropy of" |
| 6 | Neural fractals | `DLFractals/` | Experiments | Neural fractals tab: fit a network to a target function and draw it two ways |
| 7 | Rating model and user rating loop | `userinteraction.py`, `training.py`, `models.FractalModel` | Needs a labelled set (the remaining rating files are all 0) | Not integrated: the viewer has no rating mode, and there is no data to train on. The Explore tab saves `.pt` files the original tools can read |
| 8 | Visual centre prediction | `visuallayertrainer.py`, `FractalIterator.py`, `IterationsManager.py` | Needed image and centre datasets and a trained model | Not integrated: the datasets and the checkpoint are not in the repository |
| 9 | Pixel-by-pixel Python renderer | `Fractal.generatefractal`, `escape_count` | Slow; its formula line is commented out | Not integrated: the torch evaluator replaces both it and the C++ one |
| 10 | Multithreaded blocks | `MultithreadedFractalLoader.py` | Replaced by the extension's own loop | Not integrated: the GPU does the whole window |
| 11 | tkinter viewer | `ManualFractalExploration/` | Windows-only launcher | Explore tab (rubber-band zoom, right-click to go back) |
| 12 | Random colormaps | `FractalParameters.getrandomcmap` | | Explore tab: 29 colormaps |
| 13 | G-code and SVG helpers | `gcodegeneration.py`, `svgsplitter.py` | Belong to the plotter projects | Not integrated: see gcodefractal |
| 14 | Training images with random shapes | `trainingenvironnement.py` | Synthetic data for the visual model | Not integrated |
