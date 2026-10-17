import os

import numpy as np
import torch
from torch import nn
from PIL import Image
from matplotlib import cm
from sources.DLFractals.DLFractalConfiguration import DLFractalConfiguration
from sources.DLFractals.DLFractalsTraining import DLFractalsTraining
from sources.DLFractals.filters import Basic_Filter
from sources.DLFractals.fractalbasedlayers import LinearLayer, MathLayer, LinearMathwithAttentionModel, CompositeModel
from sources.DLFractals.models import Linear_Model, MultiplicativeModel, IterativeModel
from sources.fractals import FractalParameters
#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"


class DLFractals:
    def __init__(self,parameters):

        self.parameters = parameters
        self.dlfractalconf = DLFractalConfiguration()
        #self.model = Linear_Model(self.dlfractalconf)
        #self.model = MultiplicativeModel(self.dlfractalconf)
        #self.model = LinearLayer(self.dlfractalconf)
        #self.model = LinearMathwithAttentionModel(self.dlfractalconf)
        self.model = CompositeModel(self.dlfractalconf)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else :
            self.device = torch.device('cpu:0')
        self.model.to(self.device)
        self.realrange = (-2,2)
        self.imagrange = (-2, 2)
        self.resolution = (128,128)
        #self.resolution = (8, 8)
        self.imagedimension = (1000,1000)
        #self.resolution = (4, 5)
        self.maxiter = 100
        self.training = DLFractalsTraining(self,[])
    def loadModelFromDisk(self):
        self.model = torch.load(self.dlfractalconf.modelpath)

    def getModelPrediction(self,real,imag):
        result = self.model(torch.complex(torch.tensor(real,dtype=torch.double)*torch.tensor(imag,dtype=torch.double)))

    def iterateformula(self,zreal,zimag):
        iteration = 0
        while iteration < self.parameters.max_iterations:
            z = self.getModelPrediction(zreal,zimag)
            print("iteration",iteration,"z",z)
            zreal = z.real
            zimag = z.imag
            if abs(z) > 2:
                return iteration
            iteration += 1
        return self.parameters.max_iterations

    def calc_entropy(self,input_tensor):
        lsm = nn.LogSoftmax(dim=0)
        log_probs = lsm(input_tensor)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        entropy = -p_log_p.mean()
        return entropy

    def getmodelfractalentropy(self,resolution):
        fractaltensor = self.getfractaltensor(resolution)
        return self.calc_entropy(fractaltensor)

    # def getfractaltensor(self,resolution=(0,0)):
    #     with torch.no_grad():
    #         if resolution == (0,0):
    #             xresolution,yresolution = self.parameters.fractaldimensions[0],self.parameters.fractaldimensions[1]
    #         else:
    #             xresolution, yresolution = resolution
    #         # imagetensor = torch.zeros([self.parameters.fractaldimensions[0],self.parameters.fractaldimensions[1]])
    #         fractaltensor = torch.zeros([0],dtype=torch.double).to(self.device)
    #         xsteps = (self.parameters.xrange[1] - self.parameters.xrange[0]) / xresolution
    #         xpos = self.parameters.xrange[0]
    #         for x in range(xresolution):
    #             inputtensor = torch.unsqueeze(
    #                 torch.linspace(xpos + self.parameters.yrange[0] * 1j, xpos + self.parameters.yrange[1] * 1j,
    #                                steps=yresolution, dtype=torch.cdouble), 1).to(self.device)
    #             print("inputtensor.shape",inputtensor.shape)
    #             outputline = torch.flatten(self.model(inputtensor.real,inputtensor.imag))
    #             print("outputline.shape",outputline.shape)
    #             fractaltensor = torch.hstack([fractaltensor, torch.unsqueeze(outputline, 1)])
    #             #fractaltensor = torch.hstack([fractaltensor, outputline])
    #
    #             xpos += xsteps
    #         #print("fractaltensor", fractaltensor)
    #
    #         fractaltensor -= fractaltensor.min()
    #         fractaltensor /= fractaltensor.max()
    #         return fractaltensor



    def drawfractal(self,number):
        with torch.no_grad():
            #imagetensor = self.getfractaltensor()

            #print("entropy",self.calc_entropy(imagetensor))
            #imagetensor = imagetensor.detach().cpu().numpy()
            inputs = self.generate2dcomplexlinspace().flatten()
            #print("inputs.shape", inputs.shape)
            imagetensor = self.model(torch.flatten(inputs.real), torch.flatten(inputs.imag))
            imagetensor = imagetensor.reshape(self.resolution,-1)

            #print("imagetensor.shape",imagetensor.shape)
            print("Linear space Model minimum before Norm",imagetensor.min(),"Linear space Model maximum before Norm",imagetensor.max())
            #print("imagetensor.shape", imagetensor.shape)
            imagetensor -= imagetensor.min()
            imagetensor /= imagetensor.max()
            im = Image.fromarray(np.uint8(self.parameters.palette(imagetensor.cpu()) * 255))

            im.save("G:\\localtraindirnosync\\test10\\test\\blou" + str(number) +".png")
            #os.system("powershell -c " + "G:\\localtraindirnosync\\test10\\test\\blou" + str(number) +".png")
    def draw2dtensor(self,a2dtensor,number=-1,suffix="",blackandwhites=False):
        with torch.no_grad():
            a2dtensor -= a2dtensor.min()
            a2dtensor /= a2dtensor.max()
            if blackandwhites:
                palette = cm.binary
            else:
                palette = self.parameters.palette
            im = Image.fromarray(np.uint8(palette(a2dtensor.detach().cpu()) * 255))
            if number==-1:
                im.show()
            else:
                im.save("G:\\localtraindirnosync\\test10\\test\\blou" + str(number) +suffix+ ".png")

    def generate2dcomplexlinspace(self,overridedimensions=(0,0)):
        realrange,imagrange,resolution = self.realrange,self.imagrange,self.resolution
        if overridedimensions[0] != 0:
            resolution = overridedimensions

        reallinearspace = torch.linspace(realrange[0], realrange[1],
                                   steps=resolution[0], dtype=torch.cdouble,device=self.device)
        #print(reallinearspace)
        reallinearspace2d = reallinearspace * torch.ones(resolution[1],1,dtype=torch.cdouble,device=self.device)
        #print(reallinearspace2d)
        imaglinearspace = torch.linspace(imagrange[0], imagrange[1],
                                   steps=resolution[1], dtype=torch.cdouble,device=self.device)
        #print(imaglinearspace)
        imaglinearspace2d = imaglinearspace * torch.ones(resolution[0],1,dtype=torch.cdouble,device=self.device)
        imaglinearspace2d = torch.transpose(imaglinearspace2d,0,1)
        #print(imaglinearspace2d)
        complexlinspace2d = reallinearspace2d + imaglinearspace2d * 1j
        #print("complexlinspace2d.shape",complexlinspace2d.shape)
        return complexlinspace2d
    def iterateovercomplexspace(self,splitin=1):
        with torch.no_grad():
            model = self.model
            model.eval()
            #fractalfilter = Basic_Filter()
            linearspace = self.generate2dcomplexlinspace(overridedimensions=self.imagedimension)
            print("linearspace.shape",linearspace.shape)
            flattenedlinearspace = linearspace.flatten()
            splitin = int(flattenedlinearspace.shape[0] / splitin)
            splittedlinearspace = torch.split(flattenedlinearspace,splitin)

            #print(splittedlinearspace)
            overalltensor = torch.zeros(0,dtype=torch.double,device=self.device)
            n = 1
            for i in splittedlinearspace:
                print("Passe :",str(n)+"/"+str(len(splittedlinearspace)))
                fractaltensor = torch.flatten(model(torch.flatten(i.real), torch.flatten(i.imag),returnascomplex=True))
                #print(torch.cuda.memory_summary(self.device))
                #print("fractaltensor1",fractaltensor)

                resulttensor = torch.ones(fractaltensor.shape,device=self.device)-2
                for iteration in range(self.maxiter):
                    resulttensor = torch.where(fractaltensor.real*fractaltensor.real + fractaltensor.imag*fractaltensor.imag > 4, iteration, resulttensor)
                    fractaltensor = torch.where(self.bailoutfunction(fractaltensor), float('nan'), fractaltensor)
                    fractaltensor = torch.flatten(model(torch.flatten(fractaltensor.real), torch.flatten(fractaltensor.imag),returnascomplex=True))


                    #print("resulttensor", resulttensor)
                    #print("fractaltensor2", fractaltensor)
                overalltensor = torch.cat((overalltensor,resulttensor))
                n += 1

            #print(overalltensor.shape)
            overalltensor = overalltensor.reshape(self.imagedimension[1], -1)
            #print(overalltensor.shape)

            return overalltensor
    def bailoutfunction(self,fractaltensor):

        return fractaltensor.real * fractaltensor.real * fractaltensor.imag * fractaltensor.imag > 4
    def generatemodeltensor(self,splitin=1,overridedimensions= None,mode="sum"):
        self.model.eval()
        if overridedimensions == None:
            dimensions = self.resolution
        else:
            dimensions = overridedimensions
        with torch.no_grad():
            linearspace = self.generate2dcomplexlinspace(overridedimensions=dimensions)
            flattenedlinearspace = linearspace.flatten()
            splitin = int(flattenedlinearspace.shape[0] / splitin)
            splittedlinearspace = torch.split(flattenedlinearspace, splitin)
            overalltensor = torch.zeros(0, dtype=torch.double, device=self.device)
            n = 1
            for i in splittedlinearspace:
                print("Passe :",str(n)+"/"+str(len(splittedlinearspace)))
                resulttensor = torch.flatten(self.model(torch.flatten(i.real), torch.flatten(i.imag),returnascomplex=True))
                if mode=="sum":
                    resulttensor = resulttensor.real + resulttensor.imag
                if mode == "squaredistance":
                    resulttensor = resulttensor.real*resulttensor.real + resulttensor.imag*resulttensor.imag
                if mode == "squaredistancerooted":
                    resulttensor = torch.sqrt(resulttensor.real * resulttensor.real + resulttensor.imag * resulttensor.imag)
                if mode == "squaredivision":
                    resulttensor = resulttensor.imag*resulttensor.imag/resulttensor.real*resulttensor.real
                if mode == "squaredivisioninverse":
                    resulttensor = resulttensor.real*resulttensor.real/resulttensor.imag*resulttensor.imag


                overalltensor = torch.cat((overalltensor, resulttensor))
                n += 1
            modeltensor = overalltensor.reshape(dimensions, -1)
            return modeltensor
    def generatemultiplehiresmodels(self,counter,blackandwhites=False):
        modeltensor = self.generatemodeltensor(splitin=32 * 4, overridedimensions=(8000, 8000), mode="sum")
        self.draw2dtensor(modeltensor, counter, suffix="modelhiressum",blackandwhites=blackandwhites)
        modeltensor = self.generatemodeltensor(splitin=32 * 4, overridedimensions=(8000, 8000),
                                                    mode="squaredistance")
        self.draw2dtensor(modeltensor, counter, suffix="modelhiressquaredistance",blackandwhites=blackandwhites)
        modeltensor = self.generatemodeltensor(splitin=32 * 4, overridedimensions=(8000, 8000),
                                                    mode="squaredistancerooted")
        self.draw2dtensor(modeltensor, counter, suffix="modelhiressquaredistancerooted",blackandwhites=blackandwhites)
        modeltensor = self.generatemodeltensor(splitin=32 * 4, overridedimensions=(8000, 8000),
                                                    mode="squaredivision")
        self.draw2dtensor(modeltensor, counter, suffix="modelhiressquaredivision",blackandwhites=blackandwhites)
        modeltensor = self.generatemodeltensor(splitin=32 * 4, overridedimensions=(8000, 8000),
                                                    mode="squaredivisioninverse")
        self.draw2dtensor(modeltensor, counter, suffix="modelhiressquaredivisioninverse",blackandwhites=blackandwhites)
# fractalparameters = FractalParameters.defaultparameters()
# fractalparameters.fractaldimensions = (4000,4000)
# dlfractal = DLFractals(fractalparameters)
# fractaltensor = dlfractal.iterateovercomplexspace()
# print(fractaltensor)
# dlfractal.draw2dtensor(fractaltensor)
# exit()

# dlfractalconfiguration = DLFractalConfiguration()
# iterativemodel = IterativeModel(dlfractalconfiguration)
# inputs = torch.rand(10,dtype=torch.complex128)
# x = iterativemodel(torch.flatten(inputs),returnascomplex=True)
# #print(x)
# exit()
#@torch.compile
def mainloop():
    torch.set_printoptions(profile="full")
    #torch.autograd.set_detect_anomaly(True)
    fractalparameters = FractalParameters.defaultparameters()
    fractalparameters.fractaldimensions = (1000,1000)
    dlfractal = DLFractals(fractalparameters)
    recurrentfractaltraining = False


    #device = torch.device('cuda:0')
    #mathlayer = MathLayer("",torch.sin).to(device)
    #linearmathmodel = LinearMathwithAttentionModel("").to(device)
    #print(mathlayer(torch.tensor(2),torch.tensor(3)))
    #inputs = dlfractal.generate2dcomplexlinspace(overridedimensions=(8,8)).flatten()
    #print(linearmathmodel(inputs.real,inputs.imag))
    #exit()
    #dlfractal.loadModelFromDisk()
    #dlfractal.imagedimension =(8000,8000)
    #fractaltensor = dlfractal.iterateovercomplexspace(splitin=160)
    #dlfractal.draw2dtensor(fractaltensor, 21501)

    #dlfractal.generatemultiplehiresmodels(21501,blackandwhites=False)
    #exit()


    dlfractal.bigbatch = 1
    for i in range(23300,23800):

        if dlfractal.bigbatch == 4:
            dlfractal.resolution = (256, 256)
        torch.manual_seed(i)
        #entropystart = dlfractal.getmodelfractalentropy((256,256))
        #dlfractal.drawfractal(i)
        dlfractal.minibatchnumber = 4000

        epochcount = 200
        minibatchloss = 0
        for dlfractal.minibatch in range(dlfractal.minibatchnumber):
            print("Minibatch",dlfractal.minibatch)
            epochloss = dlfractal.training.trainmultipleepochs(epochcount, startfromscratch=True)
            print("")
            print("epochloss =",epochloss)
            minibatchloss += epochloss
            if dlfractal.minibatch != 0:
                if dlfractal.minibatch in [1,5,10,15]:
                    dlfractal.imagedimension = (256, 256)
                    fractaltensor = dlfractal.iterateovercomplexspace(splitin=2)
                    dlfractal.draw2dtensor(fractaltensor, i, suffix="mb" + str(dlfractal.minibatch))
                    modeltensor = dlfractal.generatemodeltensor()
                    dlfractal.draw2dtensor(modeltensor, i, suffix="modelmb" + str(dlfractal.minibatch))
                    dlfractal.imagedimension = (1000, 1000)
                if (dlfractal.minibatch % 50 == 0 and dlfractal.minibatch != 100):

                    fractaltensor = dlfractal.iterateovercomplexspace(splitin=8)
                    dlfractal.draw2dtensor(fractaltensor, i,suffix="mb"+str(dlfractal.minibatch))
                    modeltensor = dlfractal.generatemodeltensor()
                    dlfractal.draw2dtensor(modeltensor, i, suffix="modelmb"+ str(dlfractal.minibatch))
                if dlfractal.minibatch  in [100,500,1000,2000]:
                    dlfractal.imagedimension = (4000, 4000)
                    fractaltensor = dlfractal.iterateovercomplexspace(splitin=40)
                    dlfractal.draw2dtensor(fractaltensor, i,suffix="mb" + str(dlfractal.minibatch))
                    dlfractal.generatemultiplehiresmodels(i)
                    dlfractal.imagedimension = (1000, 1000)
            #fractaltensor = dlfractal.iterateovercomplexspace(splitin=8)
            #dlfractal.draw2dtensor(fractaltensor, i)
        print("minibatch loss=",minibatchloss/dlfractal.minibatchnumber)
        #print("Fractal entropy start =",entropystart,"end=",dlfractal.getmodelfractalentropy((256,256)))
        print("saving fractal ",dlfractal.bigbatch)
        # if dlfractal.bigbatch < 6:
        #     fractaltensor = dlfractal.iterateovercomplexspace(splitin=8)
        #     dlfractal.draw2dtensor(fractaltensor, i)
        #     modeltensor = dlfractal.generatemodeltensor()
        #     dlfractal.draw2dtensor(modeltensor, i, suffix="model")
        # if dlfractal.bigbatch % 10 == 0:
        #     dlfractal.imagedimension =(4000,4000)
        #     fractaltensor = dlfractal.iterateovercomplexspace(splitin=40)
        #     dlfractal.draw2dtensor(fractaltensor, i)
        dlfractal.bigbatch += 1
        print("Big batch",dlfractal.bigbatch)
        if recurrentfractaltraining:
            # Regularizing training tensor
            dlfractal.training.trainingtensor = torch.nan_to_num(fractaltensor.flatten())
            dlfractal.training.trainingtensor -= dlfractal.training.trainingtensor.min()
            dlfractal.training.trainingtensor /= dlfractal.training.trainingtensor.max()
            #dlfractal.training.trainingtensor += 2.j

        #dlfractal.dlfractalconf.debugforwarding = True
        modeltensor = dlfractal.generatemodeltensor()
        #dlfractal.dlfractalconf.debugforwarding = False
        #dlfractal.model.printweights()

        dlfractal.draw2dtensor(modeltensor, i,suffix="model")

        #dlfractal.drawfractal(i)
        #dlfractal.model.__init__()
        #print(dlfractal.iterateformula(1.2,.3))
mainloop()