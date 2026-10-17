import numpy
import numpy as np
import superfastcode

import os
# from ctypes import *
#
# # build python array
# arr =  [1,2,3,4,5]
#
# # allocates memory for an equivalent array in C and populates it with
# # values from `arr`
# arr_c = (c_int * 5)(*arr)
# print(type(arr_c))
# superfastcode.sum_array(arr_c, c_int(5))

# amatrix = superfastcode.Matrix
# amatrix.data()
import torch
from PIL import Image
from matplotlib import cm

from sources import formulainterpreter

mandelbrotwidthx = 400
mandelbrotwidthy = 400
randomweight = torch.rand(12,dtype=torch.double)
print(randomweight)
fi = formulainterpreter.FormulaInterpreter(randomweight)
print(fi.formulamatrix)
#
npimg = numpy.array(superfastcode.fractalmatrix(mandelbrotwidthx,mandelbrotwidthy,-2,1,-2,2,fi.formulamatrix))
print(npimg)
# npimg = numpy.array(superfastcode.mandelbrotmatrix(mandelbrotwidthx,mandelbrotwidthy,-2,1,-2,2))
image = Image.new(mode="P", size=(mandelbrotwidthx, mandelbrotwidthy))
im = Image.fromarray(np.uint8(cm.gist_stern(npimg)*255))

im.show()
# nptest =numpy.zeros((4,4),dtype=int)
# #nptest[1][1]=5
# print(npimg)
# nptest = superfastcode.test(nptest)
# print(nptest)


#print(superfastcode.mul(npimg,npimg))