import copy
import datetime
import math
import multiprocessing
import random
import time
from datetime import timedelta
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import sources.fractalexploration as fractalexploration
import sources.fractals as fractals
import sources.training as training
import torch
import sources.userinteraction as userinteraction
import sources.globalvariables as globalvariables
from sources.FractalIterator import  generatetrainenvironnementbyfractalid
from sources.fractalsgenerationutils import generatemultithreadedGAfractals, \
    generatehigherresolutionfractalsfromimagefolder, generatehigherresolutionfractalswithzooms
from sources.genetics import getgeneticfitfrommodel
from sources.genetics import optimizefractalrange
import multiprocessing as mp
import torch.distributions.uniform
import sources.formulainterpreter as formulainterpreter
from sources.misc import createdir
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def randomvector(vectorlength):
    random.seed(datetime.now().timestamp())
    outputvector = []
    vectorlength = vectorlength / 4
    for i in range(math.floor(vectorlength)):
        outputvector.append(random.uniform(2,8))
        outputvector.append(random.uniform(.1, 2))
        outputvector.append(random.uniform(.001, .1))
        outputvector.append(random.uniform(.0000001, .001))
        outputvector.append(random.uniform(.0000000001, .000001))
    return outputvector


# if __name__ == '__main__':
#     while True:
#         zoomradiuslist = randomvector(6)
#         print(zoomradiuslist)
#         generatehigherresolutionfractalsfromimagefolder(sourcefolderpath=globalvariables.getbasedirectory(nosyncdir=True,subdirectory="hd\\singles2"),outputfoldersubdirectory="hd\\singles2\\dream",newresolution = (4000,4000),zoomradiuslist=zoomradiuslist,numprocess=4)
#generatehigherresolutionfractalswithzooms(4,512,512,"plunge1",[2, .1, 2, .001])


# Start fractal parameters training using user output
def trainmodel():
    trainer = training.Trainer()
    #trainer.testtimecost()
    trainer.start(epochcount=3000,startfromscratch=True)

#trainmodel()
# Model class must be defined somewhere
###
# Generate a list of fractals according to a user output model.
# Many fractals a generated randomly and the best one according to model is selected
# And then saved to disk
def testuserpreferencemodel():
    testall = False
    print(globalvariables.getbasedirectory() + "model2.pt")
    model = torch.load(globalvariables.getbasedirectory() + "model2.pt")
    model.eval()
    FractalParameters = fractals.FractalParameters(xrange=(-2, 2), yrange=(-2, 2), max_iterations=40,
                                                         fractalbias=0.1 + 0.7j, fractalexponential=3,fractaldimensions=(512,512))
    afilesystem = userinteraction.CreateFileSystem(14000)
    for j in range(0,5):
        minoutputs = 1000000
        mininputs = torch.zeros(16)
        maxoutputs = 0
        maxinputs = torch.zeros(16)
        if not testall:
            for i in range(0,500):
                fixedparameters = torch.Tensor([-2,2,-2,2,0,0,0])
                input = torch.concat((fixedparameters,torch.rand(15,dtype=torch.double)),axis=0)
                print("input",input)
                outputs = model(input)
                print("outputs",outputs)
                # if outputs.item() > 7 and outputs.item() < 8:
                #     FractalParameters.setweighttensorfromparam(maxinputs)
                #     mandelbrot_set = fractals.MandelbrotSet(FractalParameters)
                #     mandelbrot_set.generatefractal(verbose=False, usemathplot=False)

                if outputs.item() > maxoutputs:
                    maxoutputs = outputs.item()
                    maxinputs = input
                if outputs.item() < minoutputs:
                    minoutputs = outputs.item()
                    mininputs = input
            print("minoutputs",minoutputs)
            print("mininputs",mininputs)
            print("maxoutputs", maxoutputs)
            print("maxinputs", maxinputs)

            FractalParameters.setweighttensorfromparam(mininputs)
            print(FractalParameters)

            afilesystem.savefile(FractalParameters)
            #FractalParameters.setweighttensorfromparam(maxinputs)
            #afilesystem.savefile(FractalParameters)
        else:
            input = (torch.rand(7, dtype=torch.double) - .5) * 4
            print("input", input)
            outputs = model(input)
            print("outputs", outputs)
            FractalParameters.setweighttensorfromparam(input)
            afilesystem.savefile(FractalParameters,suffix="note" + str(int(outputs.item())))
#testuserpreferencemodel()
#testmodel()

#generatemultithreadedGAfractals2(7,2000,startindex=0)
# print(torch.finfo(torch.float32).max)
# nearinfinite = torch.zeros(1,dtype=torch.float32)
#
# nearinfinite[0] = 1 / (2 << 61)
# print("nearinfinite",nearinfinite[0])
# print(pow(2,126))
# print(pow(2,23)-1)
# print(pow(2,8)-1)
# m = torch.distributions.Uniform(0, nearinfinite[0])
# print(1 / m.sample())

#zoomfractalrange(11003)

#for i in range(1000):
#    fractalexploration.zoomgradientdescent(9588,"Pass"+str(i))


def createzoomfromlinearlayer(fractalid,subdirectory,trainparams):
    fractal,logevent = fractalexploration.zoomgradientdescent(fractalid,subdirectory,trainparams )



    #fractalexploration.fractalgradationtocenter(fractalid=fractalid, center=fractal.parameters.getfractalcenter(), gradationnumber=100, centertosidexyinitialrange=(3,3), zoomfactor=0.0000000001, fractaldimension=(1200, 1200),doubleprecision=True,subdirectory=subdirectory)
    return logevent
# trainparams = training.TrainingParameters(optimizertype=training.TrainingParameters.Optimizer.SGD)
# createzoomfromlinearlayer(9588,"test",trainparams)

def generatemultithreadedzoomsll(numprocess,filedir,afractalid):
    if __name__ == '__main__':
        #logqueue = Queue()



        # pool object with number of element
        pool = mp.Pool(processes=numprocess)
        # input list

        #inputs =[(9588,"0"),(9589,"1"),(9590,"2"),(9591,"3"),(9592,"4"),(9593,"5"),(9594,"6"),(9595,"7"),(9596,"8"),(9597,"9"),(9598,"5"),(9593,"10"),(9599,"11"),(9601,"13"),(9602,"14"),(9603,"15"),(9604,"16"),(9605,"17"),(9606,"18"),(9607,"19")]
        inputs = []
        trainparams = training.TrainingParameters(optimizertype=training.TrainingParameters.Optimizer.SGD,inputnumber=3,numberepochs=3,learningrate=1)



        for epochs in [10]:
            trainparams.numberepochs = epochs
            for numinputs in [1]:
                trainparams.inputnumber = numinputs
                for lr in range(3):
                    for i in range(training.TrainingParameters.Optimizer.OPTIMIZERCOUNT):
                        # Repeat a few time for consistancy
                        for w in range(6):
                            inputs.append((afractalid, filedir+"id"+str(afractalid)+"ep"+str(epochs)+"inp"+str(numinputs)+"lr"+str(lr), copy.copy(trainparams)))
                        trainparams.shiftoptimizer()
                    trainparams.learningrate = trainparams.learningrate / 300


        print(inputs)
        results = pool.starmap(createzoomfromlinearlayer, inputs)
        dirpath = createdir(filedir)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open(dirpath+'\\log'+timestr+'.txt', 'w') as f:
            for res in results:
                print(res)
                f.write(str(res))
        min = 0

        for res in results:

            if min < abs(res.lastloss):
                min = abs(res.lastloss)
                bestres = res
        bestparameters = bestres.fractalparameters
        print("minimum",str(min))
        print("bestparameters",bestparameters)
        print("besttrainparams",bestres.trainparams)
        fractal = fractals.Fractal(bestparameters)
        fractalexploration.fractalgradationtocenter(fractalid=afractalid, center=fractal.parameters.getfractalcenter(), gradationnumber=300, centertosidexyinitialrange=(3,3), zoomfactor=0.0000000001, fractaldimension=(512, 512),doubleprecision=True,subdirectory=filedir)

        #results = [res.get() for res in results]
#fractalid = 9576
#fractalexploration.fractalgradationtocenter(fractalid=20000, center=(0.0000001,0), gradationnumber=2000, centertosidexyinitialrange=(3,3), zoomfactor=0.0000000000000000000001, fractaldimension=(512, 512),doubleprecision=True,subdirectory="blou5")
#generatemultithreadedzoomsll(2,"comp"+str(fractalid),fractalid)

# animator = fractalexploration.AnimationInterpolator(startradius=10,endradius=0.2)
# animator.plotinterpolation(0.01)
def createzoomsathigherentropy(fractalid,dummy):
    torch.set_printoptions(precision=30)
    center = (0,0)
    centertosidexyinitialrange=(20,20)
    zoomfactor = 0.001
    shrinkfactor = 0.5
    shrinkpass = 20

    for i in range(shrinkpass):
        solution = fractalexploration.findhientropyzonecenter(fractalid,center,centertosidexyinitialrange,zoomfactor,doubleprecision=True,configuration="fast",minimumentropy=8)
        afractal = fractals.Fractal(solution)

        afractal.generatefractalc(iteration=fractalid, overridedimensions=(200, 200), display=False,
                                  doubleprecision=True, suffix="aftershrink" + str(i))
        zoomfactor *= shrinkfactor
        center = (solution.getfractalcenter()[0],solution.getfractalcenter()[1])
        centertosidexyinitialrange = solution.getrangefromcentertosidefrommagnification(1)
        afractal = fractals.Fractal(solution)

        afractal.generatefractalc(iteration=fractalid, overridedimensions=(200, 200), display=False,
                                  doubleprecision=True, suffix="aftershrinkmods" + str(i))
        print("solution shrink pass ",i,"center",solution.getfractalcenter(),"solutionrangefromcenter", solution.getrangefromcentertosidefrommagnification(1), "range length",solution.getfractalrangelength())


    print("solutionfractal",solution)
    print(solution.getfractalrangelength())
    selectionfractal = fractals.Fractal(solution)

    selectionfractal.generatefractalc(iteration=fractalid,overridedimensions=(200,200),display=False,doubleprecision=True,suffix="selection")
    #selectionfractal.generatefractalc(iteration=-21,overridedimensions=(1000,1000),display=False,doubleprecision=True)
    finalrange =(solution.getfractalrangelength()[0]/2,solution.getfractalrangelength()[1]/2)
    fractalexploration.fractalgradationtocenter(fractalid=fractalid, center=solution.getfractalcenter(), gradationnumber=10, centertosidexyinitialrange=centertosidexyinitialrange, zoomfactor=zoomfactor, fractaldimension=(512, 512),doubleprecision=True)
#createzoomsathigherentropy(9775,0)
def generatemultithreadedzooms(numprocess):
    if __name__ == '__main__':

        # multiprocessing pool object
        pool = mp.Pool()
        # pool object with number of element
        pool = mp.Pool(processes=numprocess)
        # input list
        #inputs = [(2800,100,30,2000),(2900,100,20,4000),(3000,100,40,4000),(3100,100,20,10000)]
        #inputs = [(8000,47000),(8001,47100),(8002,47200),(8012,47300),(8013,47400),(8035,47500)]
        inputs =[(9775,0)]


        outputs = pool.starmap(createzoomsathigherentropy, inputs)
#generatemultithreadedzooms(1)
#generatemultithreadedGAfractals(4)
from random import randrange
def createfilesystemfromrandomvalues():
    afilesystem = userinteraction.CreateFileSystem(16000)
    torch.manual_seed(18553)
    for i in range(0,1000):
        start = time.time()
        FractalParameters = fractals.FractalParameters(xrange=(-2, 1), yrange=(-2, 2), max_iterations=20,
                                                             fractalbias=0.1 + 0.7j, fractalexponential=3,
                                                             fractaldimensions=(512, 512))
        # random = randrange(4)
        # if random == 0:
        #     FractalParameters.xrange=(-2,0)
        # if random == 1:
        #     FractalParameters.xrange=(0,2)
        # if random == 2:
        #     FractalParameters.xrange=(-2,2)
        # if random == 3:
        #     FractalParameters.xrange=(-2,1)
        FractalParameters.xrange = (-10, 10)
        fractalentropy = 0
        while fractalentropy < 2:
            FractalParameters.setformulatensor(torch.rand(35))
            mandelbrot_set = fractals.Fractal(FractalParameters)
            fractalentropy = mandelbrot_set.getfractalentropy(64,addcenter=True)
            print("fractalentropy",fractalentropy)
        #mandelbrot_set.generatefractalc(iteration=i,display=False)

        print("File Number=",i)
        afilesystem.savefile(FractalParameters)
        print(" Time elapsed ", str(timedelta(seconds=time.time() - start)))



createfilesystemfromrandomvalues()

print(__name__)


# Process class
# class Process(mp.Process):
#     def __init__(self, id):
#         super(Process, self).__init__()
#         self.id = id
#
#     def run(self):
#         time.sleep(1)
#         print("I'm the process with id: {}".format(self.id))


#generatemultithreadedpictures(10000,10000)
#fractals.MandelbrotSet.generatefractalfromdisk(164,10000,10000)
# fractals.MandelbrotSet.generatefractalfromdisk(391,4000,4000)
# fractals.MandelbrotSet.generatefractalfromdisk(397,4000,4000)
# fractals.MandelbrotSet.generatefractalfromdisk(413,4000,4000)
# fractals.MandelbrotSet.generatefractalfromdisk(420,4000,4000)
# fractals.MandelbrotSet.generatefractalfromdisk(433,4000,4000)
# fractals.MandelbrotSet.generatefractalfromdisk(439,4000,4000)

# with torch.no_grad():
#     testmodel()
#FractalParameters = fractals.FractalParameters(xrange=(-2, 1), yrange=(-2, 2), max_iterations=20,
#                                                         fractalbias=0.1 + 0.7j, fractalexponential=3,fractaldimensions=(8,8))
#torch.manual_seed(0)
# randomweight = torch.rand(9)
# FractalParameters.setformulatensor(randomweight)
# print(randomweight)
# print(FractalParameters.getformulatensor())
# mandelbrot_set = fractals.MandelbrotSet(FractalParameters)
# mandelbrot_set.generatefractal(verbose=True, usemathplot=False)

# FractalParameters = fractals.FractalParameters(xrange=(-2, 1), yrange=(-2, 2), max_iterations=20,
#                                                           fractalbias=0.1 + 0.7j, fractalexponential=3,
#                                                           fractaldimensions=(2000, 2000))
# torch.manual_seed(1)
# FractalParameters.setformulatensor(torch.rand(9))
# mandelbrot_set = fractals.MandelbrotSet(FractalParameters)
# mandelbrot_set.generatefractalc()
# mandelbrot_set.generatefractal(verbose=True, usemathplot=False)

