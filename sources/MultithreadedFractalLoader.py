import cProfile
import math

import numpy
import numpy as np
import superfastcode
import multiprocessing as mp

from PIL import Image
import sys
from sources.formulainterpreter import FormulaInterpreter


class MultithreadedFractalLoader:

    def __init__(self,fractalparameters):
        self.parameters = fractalparameters
    def getMultithreadedFractal(self,numdivisions,overridedimensions=(0,0),doubleprecision=False,returnpillowimage=False):
        fi = FormulaInterpreter(self.parameters.getformulatensor())
        if overridedimensions[0] == 0 and overridedimensions[1] == 0:
            width = self.parameters.fractaldimensions[0]
            height = self.parameters.fractaldimensions[1]
        else:
            width = overridedimensions[0]
            height = overridedimensions[1]
        minx, maxx, miny, maxy = self.parameters.xrange[0], self.parameters.xrange[1], self.parameters.yrange[0], \
                                 self.parameters.yrange[1]
        totalxrange = maxx - minx
        totalyrange = maxy - miny
        xstep = totalxrange / width
        ystep = totalyrange / height
        if width % numdivisions != 0:
            print("invalid width, insure that the pixel width of the fractal is divisible by ", numdivisions)
            exit()
        if height % numdivisions != 0:
            print("invalid width, insure that the pixel height of the fractal is divisible by ", numdivisions)
            exit()
        dividedwidth = int(width / numdivisions)
        dividedheight = int(height / numdivisions)
        dividedxrange = totalxrange / numdivisions
        dividedyrange = totalyrange / numdivisions
        fractalparameters = []
        for i in range(numdivisions):
            for j in range(numdivisions):
                print(i, j)
                fractalparameters.append([i, j, dividedwidth, dividedheight, minx + dividedxrange * i,
                                          minx + dividedxrange * i + dividedxrange, miny + dividedyrange * j,
                                          miny + dividedyrange * j + dividedyrange, fi.formulamatrix,
                                          fi.coefficientmatrix.numpy(), self.parameters.max_iterations])
            # npimg = numpy.array(superfastcode.fractalmatrixdouble(width, height, minx, maxx, miny, maxy, fi.formulamatrix,fi.coefficientmatrix,self.parameters.max_iterations))
        print(fractalparameters)

        print("Starting calculation in multithread mode")
        ctx = mp.get_context('spawn')
        # pool object with number of element
        pool = ctx.Pool(processes=12)
        # input list
        # inputs = [(2800,100,30,2000),(2900,100,20,4000),(3000,100,40,4000),(3100,100,20,10000)]
        # inputs = [(8000,47000),(8001,47100),(8002,47200),(8012,47300),(8013,47400),(8035,47500)]
        # npimg = numpy.array(superfastcode.fractalmatrixdouble(width, height, minx, maxx, miny, maxy, fi.formulamatrix,fi.coefficientmatrix,self.parameters.max_iterations))

        profile = cProfile.Profile()
        profile.enable()



        if doubleprecision:
            # Use cProfile to profile the worker function
            #result = cProfile.runctx('pool.starmap(self.generatefractalmultithreadwrapperdouble, fractalparameters)', globals(), locals())
            outputs = pool.starmap(self.generatefractalmultithreadwrapperdouble, fractalparameters)
        else:
            #result = cProfile.runctx('pool.starmap(self.generatefractalmultithreadwrapperfloat, fractalparameters)', globals(), locals())

            outputs = pool.starmap(self.generatefractalmultithreadwrapperfloat, fractalparameters)
            # Print the profile results to the console
        profile.disable()
        profile.print_stats()
        # columns =
        # for column in range(1,numdivisions):
        #
        #     numpy.concatenate
        print("Calculation ended, concatenating obtained images")

        npimg = self.concatenatemultithreadoutput(outputs) / self.parameters.max_iterations
        print("Image concatenated, displaying...")
        im = Image.fromarray(np.uint8(self.parameters.palette(npimg) * 255))
        if returnpillowimage:
            return Image.fromarray(np.uint8(self.parameters.palette(npimg) * 255))
        im.show()

    def generatefractalmultithreadwrapperdouble(self,column,row,width, height, minx, maxx, miny, maxy, formulamatrix,coefficientmatrix,max_iterations):
        print("torch is loaded ?",'torch' in sys.modules)
        return [column,row,numpy.array(superfastcode.fractalmatrixdouble(width, height, minx, maxx, miny, maxy, formulamatrix,coefficientmatrix,max_iterations))]

    def generatefractalmultithreadwrapperfloat(self,column,row,width, height, minx, maxx, miny, maxy, formulamatrix,coefficientmatrix,max_iterations):
        print("torch is loaded ?", 'torch' in sys.modules)
        return [column,row,numpy.array(superfastcode.fractalmatrix(width, height, minx, maxx, miny, maxy, formulamatrix,coefficientmatrix,max_iterations))]

    def concatenatemultithreadoutput(self,multithreadoutput):
        dimensions = int(math.sqrt(len(multithreadoutput)))
        blockshape = multithreadoutput[0][2].shape[0]
        print("blockshape",blockshape)
        print("dimensions",dimensions)
        print("shape",multithreadoutput[0][2].shape)
        columnarray = numpy.empty((blockshape*dimensions, 0))
        for j in range(dimensions):
            rowarray = numpy.empty((0, blockshape))
            for i in range(dimensions):

                for anoutput in multithreadoutput :
                    if anoutput[0] == i and anoutput[1] == j:
                        rowarray = np.concatenate((rowarray,anoutput[2]))
                        break
            print("rowarray",rowarray.shape)
            columnarray = np.concatenate((columnarray, rowarray),axis=1)
        print("columnarray", columnarray.shape)
        print(dimensions)
        return columnarray
