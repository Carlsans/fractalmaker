
import math
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import numpy
import numpy as np
import superfastcode

import os
import multiprocessing as mp
import torch
#import torch
from PIL import Image
from matplotlib import cm
from torchvision.transforms import transforms

import sources.formulainterpreter as formulainterpreter
import time
from dataclasses import dataclass
from datetime import timedelta

import numpy
from PIL import Image
from matplotlib import pyplot as plt

import sources.globalvariables as globalvariables

#import torch
from IPython.core.display import clear_output

from sources.MultithreadedFractalLoader import MultithreadedFractalLoader


@dataclass
class FractalParameters:
    WEIGHTTENSORLENGTH = 7 + 7 * 5
    xrange: (torch.double, torch.double)
    yrange: (torch.double, torch.double)
    max_iterations: torch.int
    fractaldimensions: (torch.int, torch.int)
    fractalbias: torch.cdouble
    fractalexponential: torch.double
    formulaweighttensor : torch.Tensor

    def __init__(self, xrange, yrange, max_iterations, fractaldimensions=(128, 128), fractalbias=0 + 0j,
                 fractalexponential=2):
        self.xrange = xrange
        self.yrange = yrange
        self.max_iterations = max_iterations
        self.fractaldimensions = fractaldimensions
        self.fractalbias = fractalbias #deprecated
        self.fractalexponential = fractalexponential#deprecated
        self.formulaweighttensor = torch.zeros(self.WEIGHTTENSORLENGTH,dtype=torch.double)
        self.palette = cm.gist_stern

    def getweighttensorfromparam(self):
        weighttensor = torch.zeros(self.WEIGHTTENSORLENGTH, dtype=torch.double)
        weighttensor[0] = self.xrange[0]
        weighttensor[1] = self.xrange[1]
        weighttensor[2] = self.yrange[0]
        weighttensor[3] = self.yrange[1]
        weighttensor[4] = self.fractalbias.real
        weighttensor[5] = self.fractalbias.imag
        weighttensor[6] = self.fractalexponential
        for i in range(7,self.WEIGHTTENSORLENGTH):
            weighttensor[i] = self.formulaweighttensor[i]

        return weighttensor

    def setweighttensorfromparam(self, weighttensor):
        self.xrange = (weighttensor[0].item(), weighttensor[1].item())
        self.yrange = (weighttensor[2].item(), weighttensor[3].item())
        self.fractalbias = complex(weighttensor[4].item(), weighttensor[5].item())
        self.fractalexponential = weighttensor[6].item()
        self.setformulatensor(weighttensor[7:self.WEIGHTTENSORLENGTH])
        for i in range(7,self.WEIGHTTENSORLENGTH):
            self.formulaweighttensor[i] = weighttensor[i].item()
        return 0
    def setweightsfromfractalid(self,fractalid):
        weighttensor = torch.load(globalvariables.getbasedirectory() + str(fractalid) + '.pt')
        self.setweighttensorfromparam(weighttensor)

    def getformulatensor(self):
        return self.formulaweighttensor[7:self.WEIGHTTENSORLENGTH]

    def setformulatensor(self,formulatensor):
        self.formulaweighttensor[7:self.WEIGHTTENSORLENGTH] = formulatensor
    def setxrange(self,range):
        self.xrange = range
    def setyrange(self,range):
        self.yrange = range

    @staticmethod
    def defaultparameters():
        return FractalParameters(xrange=(-2, 2), yrange=(-2, 2), max_iterations=100,
                                                             fractalbias=0.1 + 0.7j, fractalexponential=3,
                                                             fractaldimensions=(256, 256))
    def getfractalcenter(self):
        xrangelength = self.xrange[1] - self.xrange[0]
        yrangelength = self.yrange[1] - self.yrange[0]
        return [self.xrange[0] + xrangelength/2,self.yrange[0] + yrangelength/2]
    def getfractalrangelength(self):
        xrangelength = self.xrange[1] - self.xrange[0]
        yrangelength = self.yrange[1] - self.yrange[0]
        return (xrangelength,yrangelength)
    def getfractalscaleddimensions(self,scalefactor):
        xrangelength = self.xrange[1] - self.xrange[0]
        yrangelength = self.yrange[1] - self.yrange[0]
        xcenter,ycenter = self.getfractalcenter()

        return [xcenter - xrangelength * scalefactor /2,xcenter + xrangelength * scalefactor /2 ,ycenter - yrangelength * scalefactor /2,ycenter + yrangelength * scalefactor /2]
    def setrangefromcentertoside(self,center,centertosidexyinitialrange):
        x,y = center
        xcentertoside, ycentertoside = centertosidexyinitialrange
        self.xrange = (x-xcentertoside,x+xcentertoside)
        self.yrange = (y-ycentertoside,y+ycentertoside)
        print("xrange",self.xrange,"yrange",self.yrange)

    def getrangefromcentertosidefrommagnification(self,zoomratio=1):
        xrangefromcentertoside = (self.xrange[1] - self.xrange[0])/2
        yrangefromcentertoside = (self.yrange[1] - self.yrange[0])/2
        return (xrangefromcentertoside*zoomratio,yrangefromcentertoside*zoomratio)
    def setrangesfromlinearlayeroutput(self,linearlayeroutput):
        midrange =linearlayeroutput[2]/2
        self.xrange = (linearlayeroutput[0]-midrange,linearlayeroutput[0]+midrange)
        self.yrange = (linearlayeroutput[1]-midrange, linearlayeroutput[1]+midrange)
    def getrangesdimensions(self):
        xrangelength = self.xrange[1] - self.xrange[0]
        yrangelength = self.yrange[1] - self.yrange[0]
        return xrangelength,yrangelength

    @staticmethod
    def getcompletecmaparray():
        uniform = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        sequential = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        sequential2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                       'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                       'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
        diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                     'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        cyclic = ['twilight', 'twilight_shifted', 'hsv']
        qualitative = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                       'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                       'tab20c']
        miscellaneous = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                         'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                         'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                         'turbo', 'nipy_spectral', 'gist_ncar']
        colormaplist = uniform + sequential + sequential2 + diverging + cyclic + qualitative + miscellaneous
        return colormaplist
    @staticmethod
    def getrandomcmap(type=""):
        colormap = random.choice(FractalParameters.getcompletecmaparray())
        print("Palette en utilisation : " + colormap)
        if type == "":
            return cm.get_cmap(colormap)


        if type == "completelyrandom":
            return matplotlib.colors.ListedColormap(numpy.random.rand(256, 3))

    def setpalette(self,palette):
        self.palette = palette
@dataclass
class Fractal:
    parameters: FractalParameters
    formulainterpreter : formulainterpreter.FormulaInterpreter
    maxbitsnumber: int
    colormultiplier: int

    # parameterized constructor
    def __init__(self, parameters):
        self.parameters = parameters
        self.formulainterpreter = formulainterpreter.FormulaInterpreter(self.parameters.getformulatensor())
        self.maxbitsnumber = self.parameters.max_iterations.bit_length()
        self.colormultiplier = int(256 / (2 ** round(self.maxbitsnumber / 3)))
        self.lastnormalizedminimumvalue = None
        self.lastnormalizedmaximumvalue = None
        #self.colormultiplier = 1
        #print("colmul", self.colormultiplier)

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def stability(self, c: complex) -> float:
        return self.escape_count(c) / self.parameters.max_iterations

    def escape_count(self, c: complex) -> int:
        #return mandelbrot(c.real, c.imag)

        z = 0.000001
        z=0
        for iteration in range(self.parameters.max_iterations):
            #z = self.formulainterpreter.computeformulavalue(c, z)
            #print("iteration=",iteration," z=",z," c=",c)
            #z = z ** self.parameters.fractalexponential + c + self.parameters.fractalbias

            if abs(z) > 2:
                return iteration
        return self.parameters.max_iterations

    def getpixelcolor(self, c: complex) -> (int, int, int):
        escapecount = self.escape_count(c)
        # print("escapecount",bin(escapecount))
        r, g, b = 0, 0, 0
        # print("rangei",range(0,round(self.maxbitsnumber/3)))

        for i in range(0, round(self.maxbitsnumber / 3)):
            r = r | (escapecount & 0b1)
            escapecount = escapecount >> 1
            g = g | (escapecount & 0b1)
            escapecount = escapecount >> 1
            b = b | (escapecount & 0b1)
            escapecount = escapecount >> 1
            r = r << 1
            g = g << 1
            b = b << 1
            # r = r + (2^i) * (escapecount % 2 == 0)
        r = r >> 1
        g = g >> 1
        b = b >> 1

        # print("rgb",(r*self.colormultiplier,g*self.colormultiplier,b*self.colormultiplier))
        return (r * self.colormultiplier, g * self.colormultiplier, b * self.colormultiplier)

    def generatefractal(self, iteration=0, display=True, verbose=True,usemathplot = True, suffix=""):
        start = time.time()

        width, height = self.parameters.fractaldimensions[0], self.parameters.fractaldimensions[1]
        minx, maxx, miny, maxy = self.parameters.xrange[0], self.parameters.xrange[1], self.parameters.yrange[0], \
                                 self.parameters.yrange[1]
        stepx = (maxx - minx) / width
        stepy = (maxy - miny) / height
        image = Image.new(mode="RGB", size=(width, height))
        xpos, ypos = minx, miny
        for y in range(height):
            for x in range(width):
                c = complex(xpos, ypos)
                image.putpixel((x, y), self.getpixelcolor(c))
                #print("x",x,"y",y,"c",c)
                xpos += stepx
            xpos = minx
            ypos += stepy

            if verbose:
                clear_output(wait=True)
                print("Completion ratio :", y / height, " Time elapsed ", str(timedelta(seconds=time.time() - start)))
        image.save(globalvariables.getbasedirectory() + "image" + str(iteration) + suffix + ".png")
        end = time.time()
        if verbose:
            print("Completed at : ", str(timedelta(seconds=time.time() - start)))
        if display:
            if usemathplot:
                npimg = numpy.array(image)
                plt.imshow(npimg)
                plt.show()
            else:
                image.show()

    def generatefractalmultithread(self,numdivisions,overridedimensions=(0,0),returnpillowimage=False,doubleprecision=False):
        multithreadedfractalloader = MultithreadedFractalLoader(self.parameters)
        return multithreadedfractalloader.getMultithreadedFractal(numdivisions,returnpillowimage=returnpillowimage,doubleprecision=doubleprecision)

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


    def generatefractalc(self,iteration=-1,suffix="",overridedimensions=(0,0),display = True, returnunnormalized = False,doubleprecision = True,subdirectory="",returnnormalized = False,nosyncdir =True,returnpillowimage=False, blackandwhitetoo=False):

        if overridedimensions[0]==0 and overridedimensions[1]==0 :
            width  = self.parameters.fractaldimensions[0]
            height = self.parameters.fractaldimensions[1]
        else:
            width = overridedimensions[0]
            height = overridedimensions[1]
        minx, maxx, miny, maxy = self.parameters.xrange[0], self.parameters.xrange[1], self.parameters.yrange[0], self.parameters.yrange[1]

        fi = formulainterpreter.FormulaInterpreter(self.parameters.getformulatensor())
        #print(fi.formulamatrix)

        if doubleprecision:
            npimg = numpy.array(superfastcode.fractalmatrixdouble(width, height, minx, maxx, miny, maxy, fi.formulamatrix,fi.coefficientmatrix,self.parameters.max_iterations))
        else:
            npimg = numpy.array(superfastcode.fractalmatrix(width, height, minx, maxx, miny, maxy, fi.formulamatrix,fi.coefficientmatrix,self.parameters.max_iterations))


        if returnunnormalized:
            return npimg
        #print(npimg)
        # npimg = numpy.array(superfastcode.mandelbrotmatrix(mandelbrotwidthx,mandelbrotwidthy,-2,1,-2,2))
        image = Image.new(mode="P", size=(width, height))

        #numpy.set_printoptions(threshold=sys.maxsize)
        #print("npimg", npimg)
        npimg = npimg / self.parameters.max_iterations
        self.lastnormalizedminimumvalue = npimg.min()
        self.lastnormalizedmaximumvalue = npimg.max()
        print("Min =",self.lastnormalizedminimumvalue,"Max =",self.lastnormalizedmaximumvalue)
        #print("npimgmod", npimg)
        #im = Image.fromarray(np.uint8(cm.gist_stern(npimg) * 255),mode="RGB")
        im = Image.fromarray(np.uint8(self.parameters.palette(npimg) * 255))
        if blackandwhitetoo:
            imbw = Image.fromarray(np.uint8(cm.binary(npimg) * 255))

        print("np.asarray(im).shape",np.asarray(im).shape)
        print("npimg.shape",npimg.shape)
        if returnpillowimage:

            return Image.fromarray(np.uint8(self.parameters.palette(npimg) * 255))

        if returnnormalized:
            im = Image.fromarray(np.uint8(self.parameters.palette(npimg) * 255), mode="RGB")
            pil_to_tensor = transforms.ToTensor()(im)
            print("pil_to_tensor",pil_to_tensor.shape)

            return pil_to_tensor
        if subdirectory != "":
            dirpath = globalvariables.getbasedirectory(nosyncdir=nosyncdir) + subdirectory
            if dirpath[-1] != "\\":
                dirpath += "\\"
            # Create if dir not exist
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            im.save(dirpath + str(iteration) + suffix + ".png")
            if blackandwhitetoo:
                imbw.save(dirpath + str(iteration) + suffix + "bw.png")
            if display:
                os.system("powershell -c "+dirpath + str(iteration) + suffix + ".png")
        else:
            im.save(globalvariables.getbasedirectory(nosyncdir=nosyncdir) + str(iteration) + suffix + ".png")
            if blackandwhitetoo:
                imbw.save(globalvariables.getbasedirectory(nosyncdir=nosyncdir) + str(iteration) + suffix + "bw.png")
            if display:
                os.system("powershell -c "+globalvariables.getbasedirectory(nosyncdir=nosyncdir) + str(iteration) + suffix + ".png")

        #sys.exit("test")



    def getfractalentropy(self,resolution,assymetric = -1,doubleprecision=True, addcenter = False):
        fractalmatrix = self.generatefractalc(self,overridedimensions=(resolution,resolution),display = False, returnunnormalized = True,doubleprecision=doubleprecision)

        if assymetric == 0:
            halfmatrix = int(fractalmatrix.shape[1]/2)
            fractalmatrix = fractalmatrix[0:fractalmatrix.shape[0],halfmatrix:fractalmatrix.shape[1]]
        if assymetric == 1:
            halfmatrix = int(fractalmatrix.shape[0]/2)
            fractalmatrix = fractalmatrix[halfmatrix:fractalmatrix.shape[0],0:fractalmatrix.shape[1]]

            #print(fractalmatrix.shape)
        #print("fractalmatrix",fractalmatrix.shape)
        entropy = self.entropy(torch.unsqueeze(torch.from_numpy(fractalmatrix),dim=0))
        if addcenter:
            if resolution % 4 != 0:
                raise Exception("Resolution must be dividable by 4 to use with the addcenter option")
            center = int(resolution / 2)
            margin = int(center / 2)
            fractalmatrix = fractalmatrix[margin:center+margin, margin:center+margin]
            entropy += self.entropy(torch.unsqueeze(torch.from_numpy(fractalmatrix), dim=0))
        #print("entropy = ",entropy)
        return entropy.item()
    def getfractalnormalizedsum(self,resolution,doubleprecision=False,maxiterations=100):
        fractalmatrix = self.generatefractalc(self,overridedimensions=(resolution,resolution),display = False, returnunnormalized = True,doubleprecision=doubleprecision)
        totalpixel = resolution * resolution
        normalizedmatrixsum = (np.sum(fractalmatrix)/totalpixel)/maxiterations
        return normalizedmatrixsum


    @staticmethod
    def entropy(residual):

        # residual.shape[0] -> Number of channels
        # residual.shape[1] -> Width
        # residual.shape[2] -> Height
        entropy = torch.zeros( 1, requires_grad=True)
        imageSize = float(residual.shape[0] * residual.shape[1] * residual.shape[2])
        _, counts = torch.unique(residual.data.flatten(start_dim=0), dim=0,
                                 return_counts=True)  # flatt tensor and compute the unique values
        p = (counts) / imageSize  # compute the
        entropy = torch.sum(p * torch.log2(p)) * -1
        return entropy
    @staticmethod
    def generatefractalfromdisk(id,resolutionx,resolutiony,subdirectory=""):
        weighttensor = torch.load(globalvariables.getbasedirectory() + str(id) + '.pt')
        #print(weighttensor)
        parameters = FractalParameters(0,0,60)
        parameters.setweighttensorfromparam(weighttensor)
        parameters.fractaldimensions = (resolutionx,resolutiony)
        mandelbrot = Fractal(parameters)
        mandelbrot.generatefractalc(iteration=id,display=False,suffix ="D" + str(resolutionx) + "X" + str(resolutiony),subdirectory=subdirectory)
        #mandelbrot.generatefractal(iteration=id,usemathplot=False,suffix ="D" + str(resolutionx) + "X" + str(resolutiony))