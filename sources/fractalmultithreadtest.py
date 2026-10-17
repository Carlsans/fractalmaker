from sources import fractals

if __name__ == '__main__':
    FractalParameters = fractals.FractalParameters(xrange=(-2, 2), yrange=(-2, 2), max_iterations=20,
                                                                 fractalbias=0.1 + 0.7j, fractalexponential=3,
                                                                 fractaldimensions=(512, 512))
    fractal = fractals.Fractal(FractalParameters)
    fractal.generatefractalmultithread(4)
