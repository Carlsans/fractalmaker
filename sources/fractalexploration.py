from os import path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Parameter
from torch.optim import SGD
from torch import nn
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR

from sources import fractals, userinteraction, globalvariables, training
from sources.debugging import TensorVisualization
from sources.genetics import optimizefractalrange
from sources.globalvariables import getbasedirectory


def zoomfractalrange(fractalid,fileindex,entropyfuncprecision=128):
    weighttensor = torch.load(globalvariables.getbasedirectory() + str(fractalid) + '.pt')
    fractalparam = fractals.FractalParameters.defaultparameters()
    fractalparam.setweighttensorfromparam(weighttensor)
    fractalparam.fractaldimensions=(1200,1200)
    fractal = fractals.Fractal(fractalparam)
    print("scaled centers")
    print(fractal.parameters.getfractalscaleddimensions(.9))
    fractaloriginalentropy = fractal.getfractalentropy(entropyfuncprecision)
    minxrange = -2
    minyrange = -2
    xrange = 2
    yrange = 2
    scalefactor =.9
    afilesystem = userinteraction.CreateFileSystem(fileindex)
    for i in range(0,100):
        solution = optimizefractalrange(fractal,fractaloriginalentropy,minxrange,minyrange,xrange,yrange,entropyfuncprecision=entropyfuncprecision)
        fractal = fractals.Fractal(solution)
        print(fractal.parameters)
        #fractal.generatefractalc()
        afilesystem.savefile(fractal.parameters)
        minxrange,_,minyrange,_ = fractal.parameters.getfractalscaleddimensions(scalefactor)
        print("scaled centers")
        print(fractal.parameters.getfractalscaleddimensions(.9))
        xrange *= scalefactor
        yrange *= scalefactor
        # minxrange = fractal.parameters.xrange[0] + xrange * (1-shrinkingcoeff)
        # minyrange = fractal.parameters.yrange[0] + yrange * (1-shrinkingcoeff)
        # xrange = xrange * (1-shrinkingcoeff)
        # yrange = yrange * (1-shrinkingcoeff)

class AnimationInterpolator:
    startradius:float
    endradius:float
    def __init__(self,startradius,endradius):
        self.startradius = startradius
        self.endradius = endradius
    def linearinterpolationfromnormalizedvalue(self,normalizedvalue):
        self.isnormalized(normalizedvalue)
        y = self.startradius - (self.startradius-self.endradius) * normalizedvalue
        return y
    def animationfunctionfromnormalizedvalue(self,normalizedvalue):
        self.isnormalized(normalizedvalue)
        y = 1 - normalizedvalue ** 0.1
        if not self.isnormalized(y,stopexecution=False):
            print("Not normalized y value =",y)
        return y

    def animatorfromnormalizedvalue(self,normalizedvalue):
        return self.linearinterpolationfromnormalizedvalue(normalizedvalue) * self.animationfunctionfromnormalizedvalue(normalizedvalue)
    def plotinterpolation(self,stepsize):
        linear = []
        animation = []
        result = []
        xvalues = []
        for x in np.arange(0, 1 + stepsize/10, stepsize):
            print("linear function x=", x, " y=", self.linearinterpolationfromnormalizedvalue(x))
            linear.append(self.linearinterpolationfromnormalizedvalue(x))
            print("animation function x=", x, " y=", self.animationfunctionfromnormalizedvalue(x))
            animation.append(self.animationfunctionfromnormalizedvalue(x))
            print("Result=", self.animatorfromnormalizedvalue(x))
            result.append(self.animatorfromnormalizedvalue(x))
            xvalues.append(x)

        plt.plot(xvalues, linear,'r')
        plt.plot(xvalues, animation,'g')
        plt.plot(xvalues, result, 'b')
        plt.ylabel('results')
        plt.xlabel('normalised value')
        plt.legend(["Linear path", "Animation function","Result"], loc="lower left")
        plt.show()

    @staticmethod
    def isnormalized(number,stopexecution=True):
        if number < 0 or number > 1:
            if stopexecution:
                raise Exception("linear interpolation value must be normalized")
            print("linear interpolation value must be normalized")
            return False
        return True


def fractalgradationtocenter(fractalid,center,gradationnumber,centertosidexyinitialrange,zoomfactor,fractaldimension=(1200, 1200),doubleprecision=False,suffix="",subdirectory=""):
    """
    :param fractalid: Fractal identifiant
    :param center: Center on witch to zoom to
    :param gradationnumber: Number of images in the zoom
    :param initialrange: X and Y start range ex.: (2,2) gives (-2,2),(-2,2) range
    :param zoomfactor:
    :param fractaldimension: Number of pixels on both axis
    :param doubleprecision: Use doubles instead of floats for fractal computation
    """
    xcenter,ycenter = center[0],center[1]
    xcentertosideinitialrange, ycentertosideinitialrange = centertosidexyinitialrange
    xfinalrange, yfinalrange = xcentertosideinitialrange * zoomfactor, ycentertosideinitialrange * zoomfactor
    weighttensor = torch.load(globalvariables.getbasedirectory() + str(fractalid) + '.pt')
    fractalparam = fractals.FractalParameters.defaultparameters()
    fractalparam.setweighttensorfromparam(weighttensor)
    fractalparam.fractaldimensions = fractaldimension
    fractalparam.setrangefromcentertoside(center,centertosidexyinitialrange)
    fractal = fractals.Fractal(fractalparam)
    firstframefractal = fractal
    #xstep = abs(xfinalrange-xcentertosideinitialrange)/(gradationnumber)
    #ystep = abs(yfinalrange-ycentertosideinitialrange)/(gradationnumber)
    xstep = abs(xfinalrange/xcentertosideinitialrange*gradationnumber)
    ystep = abs(yfinalrange/ycentertosideinitialrange*gradationnumber)

    animator = AnimationInterpolator(startradius=xcentertosideinitialrange, endradius=xfinalrange)
    animator.plotinterpolation(0.01)
    print("xstep",xstep,"ystep",ystep)
    for i in range(0,gradationnumber):
       #print(fractal.parameters)
       #print(fractal.parameters.getweighttensorfromparam())
       fractalradius = animator.animatorfromnormalizedvalue(i / gradationnumber)
       fractal.parameters.setrangefromcentertoside(center,(fractalradius,fractalradius))
       fractal.generatefractalc(iteration=fractalid,suffix="zoomframe"+str(i)+suffix,display=False,doubleprecision=doubleprecision,subdirectory=subdirectory)
       #fractalparam.setrangefromcentertoside(center, ((gradationnumber-i)*xstep,(gradationnumber-i)*ystep))
       #fractal.parameters.setrangefromcentertoside(center, firstframefractal.parameters.getrangefromcentertosidefrommagnification((gradationnumber-i)/(gradationnumber)))

       #fractal.parameters.xrange = (fractal.parameters.xrange[0]+xstep,fractal.parameters.xrange[1]-xstep)
       #fractal.parameters.yrange = (fractal.parameters.yrange[0] + ystep, fractal.parameters.yrange[1] - ystep)
       print("xrange",fractal.parameters.xrange,"xspan",fractal.parameters.xrange[1]-fractal.parameters.xrange[0])
       print("yrange",fractal.parameters.yrange,"yspan",fractal.parameters.yrange[1]-fractal.parameters.yrange[0])



def findhientropyzonecenter(fractalid,center, centertosidexyinitialrange,zoomfactor,doubleprecision=False,configuration="fast",minimumentropy = -1):
    entropyfitness =-2
    while entropyfitness < minimumentropy:
        fractaldimension = (1200, 1200)
        xcenter, ycenter = center
        xinitialrange, yinitialrange = centertosidexyinitialrange
        weighttensor = torch.load(globalvariables.getbasedirectory() + str(fractalid) + '.pt')
        fractalparam = fractals.FractalParameters.defaultparameters()
        fractalparam.setweighttensorfromparam(weighttensor)
        fractalparam.fractaldimensions = fractaldimension
        fractal = fractals.Fractal(fractalparam)
        if configuration == "fast":
            numgeneration = 5
            populationcount = 100
        if configuration == "medium":
            numgeneration = 5
            populationcount = 100
        if configuration == "slow":
            numgeneration = 10
            populationcount = 500
        if configuration == "comprehensive":
            numgeneration = 20
            populationcount = 2000
        solution,entropyfitness = optimizefractalrange(fractal, -1, center, centertosidexyinitialrange,numgeneration=numgeneration, populationcount=populationcount, entropyfuncprecision = 64,zoomfactor = zoomfactor,doubleprecision=doubleprecision)
        if entropyfitness < minimumentropy:
            print("Failed with entropy fitness value of ",entropyfitness)
    print("solution",solution)
    return solution

def totalentropyloss(output,target):
    fractal = target
    midrange = output[2]/2
    fractal.parameters.xrange = [output[0]-midrange,output[0]+midrange]
    fractal.parameters.yrange = [output[1]-midrange, output[1]+midrange]
    fractalentropy = -fractal.getfractalentropy(128, doubleprecision=True)
    #fractalentropy = fractalentropy-fractal.getfractalnormalizedsum(128, doubleprecision=True)
    print("fractalentropy",fractalentropy)
    xrange,yrange = fractal.parameters.getrangesdimensions()
    xrange = abs(xrange)
    yrange = abs(yrange)
    if(xrange > yrange):
        dimensionloss = xrange
    else:
        dimensionloss = yrange
    print("dimensionloss",dimensionloss)
    loss = torch.tensor([fractalentropy/abs(dimensionloss)], requires_grad=True)
    print("loss",loss)
    #if output[1] < output[0] or output[3] < output[2] :
    #    return torch.tensor([10.0], requires_grad=True)
    return loss

class FractalRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 256,dtype=torch.double)
        self.linear2 = torch.nn.Linear(256, 3, dtype=torch.double)
        #self.weights = nn.Parameter(torch.tensor([[-2,2],[-2,2]],dtype=torch.double))

    def forward(self, x):
       # print("x",x)
       # print("self.linear(x)",self.linear(x))
        x = self.linear(x)
        return self.linear2(x)


def zoomgradientdescent(fractalid, subdirectory="",trainparams = None):
    torch.set_printoptions(precision=60)
    if trainparams==None:
        trainparams = training.TrainingParameters(optimizertype=training.TrainingParameters.Optimizer.SGD,inputnumber = 2, numberepochs = 2)

    tensorvisualizer = TensorVisualization()
    fractaldimension = (256, 256)
    weighttensor = torch.load(globalvariables.getbasedirectory() + str(fractalid) + '.pt')
    fractalparam = fractals.FractalParameters.defaultparameters()
    fractalparam.setweighttensorfromparam(weighttensor)
    fractalparam.fractaldimensions = fractaldimension
    fractal = fractals.Fractal(fractalparam)
    model = FractalRegression()
    # linearmodelpath = getbasedirectory() + "linearmodel.pt"
    # if path.exists(linearmodelpath):
    #     model = torch.load(linearmodelpath)
    #     model.eval()

    optimizer = trainparams.getoptimizer(model.parameters())
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_fn = totalentropyloss
    #inputs = torch.tensor([[-2,2,-2,2],[-1,1,-1,1],[-3,3,-3,3],[-4,4,-4,4],[-.5,.5,-.5,.5],[-2.5,2,-2,2.5],[-1.5,1,-1,1.5],[-3,3.5,-3,3],[-4.5,4,-4,4],[-.5,.25,-.5,.25]],dtype=torch.double,requires_grad=True)
    inputs = torch.rand(trainparams.inputnumber,3,dtype=torch.double,requires_grad=True) * 8
    runninginputs = torch.zeros(inputs.shape[0],inputs.shape[1],dtype=torch.double)
    runningloss = torch.zeros(inputs.shape[0])
    logevent = training.TrainingEvent(fractal.parameters,trainparams,"Fractal regression id = "+str(fractalid))
    for epoch in range(trainparams.numberepochs):
        i = 0
        print("epoch",epoch)
        print("inputs",inputs)
        for xb in inputs:
            #print("xb",xb)
            #print("model.parameters()",model.parameters())
            #print("model.linear",model.linear)
            #print("model.linear.grad",model.linear.)
            #print(model.weights.data)
            #print(model.weights.grad)
            #print("model[0].grad",model[0].grad)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs,fractal)

            loss.backward()


            optimizer.step()

            with torch.no_grad():
                runninginputs[i] = outputs
                runningloss[i] = loss
                i += 1
                fractal.parameters.setrangesfromlinearlayeroutput(outputs)
                #numpyimage = fractal.generatefractalc(returnunnormalized=True,overridedimensions=(64,64))

                #stepvisualisation = fractal.generatefractalc(returnunnormalized=True, overridedimensions=(32, 32))
                #print("shape",numpyimage.shape)
                #print("numpyimage",numpyimage)
                #tensorvisualizer.addcol2dnumpy(numpyimage)
        #scheduler.step()
        tensorvisualizer.newline()
        inputs = runninginputs
    #tensorvisualizer.showcurrentmatrix()
            #fractal.parameters.xrange = model[0][0]
            #fractal.parameters.yrange = model[0][1]

    print("outputs",inputs)
    print("runningloss",runningloss)
    lossmaxvalue = 0
    lossmaxindex = -1
    for i in range(inputs.shape[0]):
        if runningloss[i] > lossmaxvalue :
            lossmaxvalue = runningloss[i]
            lossmaxindex = i
    fractal.parameters.setrangesfromlinearlayeroutput(inputs[lossmaxindex])
    logevent.lastloss = loss.item()
    print("logevent.lastloss",logevent.lastloss)
    logevent.addinfo("Final loss = "+str(loss)+ "\n Total X range = " + str(fractal.parameters.getfractalrangelength()[0]) + "\nTotal Y Range = " + str(fractal.parameters.getfractalrangelength()[1]))
    #torch.save(model, linearmodelpath)

    fractal.generatefractalc(iteration=-333,doubleprecision=True,suffix="p"+str(lossmaxindex),display=False,subdirectory=subdirectory)
    return (fractal,logevent)