import multiprocessing as mp
import os
from glob import glob
import calendar
import time
import torch
from sources import fractals, userinteraction
from sources.FractalIterator import FractalIterator
from sources.genetics import getgeneticfitfrommodel
from fractals import Fractal
###
# Create a fractal using entropy as selection function for a genetic algorithm.
# The use of a model to assist selection is currently disabled
from sources.globalvariables import getbasedirectory


def createfractalsfromgeneticalgorithmandnn(startindex,numeroffilestocreate,numgeneration=5, populationcount=100, symetricentropy =-1):
    #model = torch.load(globalvariables.getbasedirectory() + "model2.pt")
    model = None
    #print(model)

    FractalParameters = fractals.FractalParameters(xrange=(-2, 2), yrange=(-2, 2), max_iterations=40,
                                                         fractalbias=0.1 + 0.7j, fractalexponential=3,
                                                         fractaldimensions=(256, 256))
    fractal = fractals.Fractal(FractalParameters)
    afilesystem = userinteraction.CreateFileSystem(startindex)
    for j in range(0, numeroffilestocreate):
        geneticfit = getgeneticfitfrommodel(model, genenumber=fractal.parameters.getformulatensor().shape[0], numgeneration=numgeneration, populationcount=populationcount)
        geneticfit = torch.from_numpy(geneticfit)
        fixedparameters = torch.Tensor([-2, 2, -2, 2, 0, 0, 0])
        geneticfit = torch.concat((fixedparameters, geneticfit), axis=0)
        fractal.parameters.setweighttensorfromparam(geneticfit)
        entropy = fractal.getfractalentropy(16)
        formatedentropy = "{:.2f}".format(round(entropy, 2))
        formatedentropy = formatedentropy.replace('.','p')
        afilesystem.savefile(fractal.parameters)
            # FractalParameters.setweighttensorfromparam(maxinputs)
            # afilesystem.savefile(FractalParameters)
#createfractalsfromgeneticalgorithmandnn(startindex=10,numeroffilestocreate=1000)
###
# Generate multiple fractals using threads. blocksize = number of fractals by process
# Total number of fractals created = blocksize * numprocess
def generatemultithreadedGAfractals(numprocess,blocksize,startindex=0):
    if __name__ == '__main__':

        # multiprocessing pool object
        pool = mp.Pool()
        # pool object with number of element
        pool = mp.Pool(processes=numprocess)
        # input list
        #inputs = [(2800,100,30,2000),(2900,100,20,4000),(3000,100,40,4000),(3100,100,20,10000)]
        totalfractalnumber = blocksize * numprocess
        start,end= startindex,startindex + totalfractalnumber


        filerange = range(start,end,blocksize)
        print(filerange)
        inputs = []
        for i in filerange:
            inputs.append((i,blocksize,20,200,0))
        print(inputs)


        outputs = pool.starmap(createfractalsfromgeneticalgorithmandnn, inputs)
###
# Take a folder of selected fractals images and upscale them to the desired resolution(saved model needed !)
def generatehigherresolutionfractalsfromimagefolder(sourcefolderpath,outputfoldersubdirectory,newresolution,zoomradiuslist,numprocess=6):
    os.chdir(sourcefolderpath)
    print(sourcefolderpath)
    fileslist = glob("*.png")
    for i in range(len(fileslist)):
        try:
            fileslist[i] = int(fileslist[i][:-4])
        except ValueError:
            # Handle the exception
            fileslist.remove(fileslist[i])

    print(fileslist)
    # pool object with number of element
    pool = mp.Pool(processes=numprocess)

    #fractals.MandelbrotSet.generatefractalfromdisk(391, 4000, 4000,outputfoldersubdirectory)
    inputs = []
    for i in fileslist:
        inputs.append((i,newresolution[0],newresolution[1],outputfoldersubdirectory+"\\"+str(i),zoomradiuslist))
    print(inputs)


    outputs = pool.starmap(generatehigherresolutionfractalswithzooms, inputs)


def generatehigherresolutionfractalswithzooms(id,resolutionx,resolutiony,subdirectory,zoomradiuslist):
    #Fractal.generatefractalfromdisk(id,resolutionx,resolutiony,subdirectory)
    FractalIterator.iteratefractalbyfractalid(id,zoomradiuslist,(resolutionx,resolutiony),subdirectory)
    if os.path.isfile(getbasedirectory(subdirectory=subdirectory,nosyncdir=True)+str(id)+".png"):
        os.rename(getbasedirectory(subdirectory=subdirectory,nosyncdir=True)+str(id)+".png", getbasedirectory(subdirectory=subdirectory,nosyncdir=True)+"d"+str(id)+".png")
    else:
        print(getbasedirectory(subdirectory=subdirectory,nosyncdir=True)+str(id)+".png")
        print("Fichier non existant !")