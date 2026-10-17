# fractalmaker

An escape-time fractal is a vector of 35 numbers in [0, 1] (five terms of operator, function, variable and four coefficients). This early project (2022-2023) searches such vectors with a genetic algorithm,
zooms into regions where the entropy of the picture is high, makes zoom animations, and experiments with learned preferences and with networks that draw fractals.

- How it works, what worked and what did not: [docs/PROJECT.md](docs/PROJECT.md)
- Features that were tried and dropped: [docs/UNUSED_FEATURES.md](docs/UNUSED_FEATURES.md)

## GUI

```
pip install -r requirements-gui.txt
python -m gui.app
```

- **Explore**: edit the 35 numbers (with a plain reading of each term), render on the GPU, drag a rectangle to zoom, right-click to go back, open and save `.pt` weight files.
- **Search**: evolve formulas by entropy, find a high-entropy zone to zoom into, and the old regression-network zoom.
- **Zoom animation**: frames and GIFs with the original interpolation.
- **Neural fractals (legacy)**: fit a network to a function and iterate it.

The image orientation follows the original: rows are the real axis.

## Original scripts

`sources/` (Windows; it imports a C++ extension, `superfastcode`, that is built with Visual Studio from `sources/ctopython/`). Not included: the phone-number scraper and the diffusion test that sat in the same folder.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Personal, research and hobby use are allowed; commercial use is not.
