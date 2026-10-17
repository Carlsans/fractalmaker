import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from torch import Tensor
from torch.nn.functional import normalize

from sources import globalvariables
from sources.FractalDepthSamplor import FractalDepthSamplor
from sources.IterationsManager import IterationsManager
from sources.IteratorEvent import IteratorEvent
from sources.fractals import Fractal, FractalParameters
from sources.globalvariables import getbasedirectory, getcenterpredictionmodelpath
from sources.IteratorLogManager import IteratorLogManager
from sources.misc import settensortodevice, removeNan
from sources.models import getmodel


class FractalIterator:
    fractalid:int
    actualposition:(float,float)
    radius:float
    activeid:int
    selectiontype:str
    imageresolution:int
    modelresolution:int
    distributionmatrixresolution:int
    currentiteratorevent:IteratorEvent
    iteratorlogmanager:IteratorLogManager
    iterationsmanager:IterationsManager
    currentfractal:Fractal
    currentfractaltensor:torch.Tensor
    fractalshrinkingratio:float
    display:bool
    debug:bool
    maxiter:int
    fractaldepthsamplor:FractalDepthSamplor
    def __init__(self,iterationsmanager,fractalid,selectiontype,actualposition,radius,fractalresolution,modelresolution,display=False,debug = False,fractaldepthsamplor = None):
        self.maxiter = 200
        self.fractalid = fractalid
        if selectiontype != "random":
            self.selectiontype = selectiontype
        else:
            self.selectiontype = random.choice(["min","max"])

        #self.actualposition = actualposition
        #self.radius = radius
        self.fractalresolution = fractalresolution
        self.modelresolution = modelresolution
        #self.activeid =-1
        self.display=display
        self.debug=debug
        fractalparameters = FractalParameters.defaultparameters()
        fractalparameters.fractaldimensions = (self.fractalresolution,self.fractalresolution)
        fractalparameters.setweightsfromfractalid(self.fractalid)
        self.currentiteratorevent = IteratorEvent(id = -1,parentid=-1,eventposition=actualposition,eventradius=radius)
        fractalparameters.setrangefromcentertoside(self.currentiteratorevent.eventposition,(self.currentiteratorevent.eventradius,self.currentiteratorevent.eventradius))
        self.currentfractal = Fractal(fractalparameters)
        self.currentfractaltensor = self.currentfractal.generatefractalc(returnnormalized=True)
        if self.display:
            self.currentfractal.generatefractalc(display=True)
        # activeid and parentid are the same for the moment
        self.currentiteratorevent.distributionmatrix = self.getfractaldistributionmatrix(self.getfractalpredictionsvectors())
        self.iteratorlogmanager = IteratorLogManager(self.currentiteratorevent,self.fractalid)
        self.iterationsmanager = iterationsmanager
        self.distributionmatrixresolution = self.getfractalpredictionsvectors()[0].shape[0]
        self.fractalshrinkingratio = (self.distributionmatrixresolution-2)/self.distributionmatrixresolution

        self.getfollowingchild()
    def flipselectiontype(self):
        if self.selectiontype == "max":
            self.selectiontype = "min"
        else:
            self.selectiontype = "max"
        print("flipped selection type to "+self.selectiontype)
    def getfollowingchild(self):
        parentevent = self.currentiteratorevent
        distributionmatrix = parentevent.getdistributionmatrix()

        if self.selectiontype == "max":
            # Make sure Nan values are not used
            tempdistributionmatrix = torch.nan_to_num(distributionmatrix.detach(), nan=-1000000000)
            selectionx = torch.argmax(torch.max(tempdistributionmatrix, 1)[0]).item()
            selectiony = torch.argmax(torch.max(tempdistributionmatrix, 0)[0]).item()
            selectionvalue = torch.max(tempdistributionmatrix)
        if self.selectiontype == "min":
            tempdistributionmatrix = torch.nan_to_num(distributionmatrix.detach(), nan=1000000000)
            print(tempdistributionmatrix)
            #test = torch.argmin(torch.min(tempdistributionmatrix,1)[0])

            selectionx = torch.argmin(torch.min(tempdistributionmatrix,1)[0]).item()
            selectiony = torch.argmin(torch.min(tempdistributionmatrix,0)[0]).item()
            selectionvalue = torch.min(tempdistributionmatrix)
            #print(tempdistributionmatrix[selectionx][selectiony],selectionvalue)



        #selectionvalue = distributionmatrix[selectiony][selectionx]
        #print(selectionvalue)
        print(selectionx,selectiony)
        #print(distributionmatrix[selectionx][selectiony])
        # This index of the distribution matrix is Nan and will not be used anymore
        distributionmatrix[selectionx][selectiony] = torch.nan
        print("distributionmatrix",distributionmatrix)
        parentevent.setdistributionmatrix(distributionmatrix)

        #distributionmatrix = self.iteratorlogmanager.geteventbyid(self.activeid).getdistributionmatrix()
        #print("currentevent.distributionmatrix",currentevent.distributionmatrix)
        childnextradius = self.fractalshrinkingratio * parentevent.eventradius
        childnextcenter = self.getnewfractalcenter(selectionx, selectiony)

        self.currentiteratorevent = self.iteratorlogmanager.addnewiteratorevent(parentevent.id,childnextcenter,childnextradius)
        parentevent.addaverredcenter(childid=self.currentiteratorevent.id,center = (selectionx,selectiony))
        #self.activeid = self.currentiteratorevent.id
        self.currentfractal.parameters.setrangefromcentertoside(self.currentiteratorevent.eventposition,(self.currentiteratorevent.eventradius,self.currentiteratorevent.eventradius))
        self.currentfractaltensor = self.currentfractal.generatefractalc(returnnormalized=True)
        if self.display:
            self.currentfractal.generatefractalc(display=True)

        #self.currentfractal.generatefractalc(iteration=self.currentiteratorevent.id,suffix="plunge",subdirectory="plunge1\\"+self.selectiontype+str(self.fractalid),display=False,overridedimensions=(128,128))

        self.currentiteratorevent.distributionmatrix = self.getfractaldistributionmatrix(self.getfractalpredictionsvectors())
        #print("self.iteratorlogmanager", self.iteratorlogmanager)

        self.iterationsmanager.fitnessfunc(self)
        # if instructions:
        #     self.getfollowingchild()

        # print(childnextradius)
        # print("childeventid",childeventid)


    def getnewfractalcenter(self,selectionx,selectiony):
        actualx,actualy = self.currentiteratorevent.eventposition
        resolutionstep = self.currentiteratorevent.eventradius / self.distributionmatrixresolution
        xmod = ((self.distributionmatrixresolution / 2) - selectionx) * resolutionstep
        xnewcenter = actualx - xmod
        ymod = ((self.distributionmatrixresolution / 2) - selectiony) * resolutionstep
        ynewcenter = actualy - ymod
        # xratio = (selectionx + 1) / self.distributionmatrixresolution
        # yratio = (selectiony + 1) / self.distributionmatrixresolution

        return (xnewcenter,ynewcenter)
    def getfractalpredictionsvectors(self,debug=True):
        fractaltensor = torch.unsqueeze(self.currentfractaltensor, 0)
        model = getmodel(getcenterpredictionmodelpath())
        fractaltensor = settensortodevice(fractaltensor)
        outputx = model(fractaltensor)
        maxx, predictedx = torch.max(outputx, 1)
        outputy = model(fractaltensor.permute(0, 1, 3, 2))
        maxy, predictedy = torch.max(outputy, 1)
        if debug:
            print("Predicted Answer = (", predictedx.item(), ",", predictedy.item(),
              ") maximum = (", maxx.item(), ",", maxy.item(), ")")
        # distributionmatrix(outputx[0], outputy[0])
        # if display:
        #     fractal.generatefractalc()
        return (outputx[0],outputy[0])

    def getfractaldistributionmatrix(self,predictionvectors):
        with torch.no_grad():
            outputx,outputy = predictionvectors
            xsortindex = torch.argsort(outputx, descending=True)
            ysortindex = torch.argsort(outputy, descending=True)
            normalizedx = normalize(outputx, p=2.0, dim=0) + 0.5
            normalizedy = normalize(outputy, p=2.0, dim=0) + 0.5
            summatrix = (normalizedy[:, None] + normalizedx) / 2

            #summatrix = summatrix.permute(1,0)
            if self.debug:
                print(summatrix)
            if self.display:
                im = Image.fromarray(np.uint8(cm.Greys(Tensor.cpu(summatrix)) * 255))
                # im = Image.fromarray(np.uint8(cm.Greys([[.2,.8],[.8,.1]]) * 255))
                im = im.resize((128, 128), 0)
                imagepath = globalvariables.getbasedirectory() + str(-1) + "distributiongraph" + ".png"
                im.save(imagepath)
                os.system("powershell -c " + imagepath)
            return summatrix
    def restartfromfirstevent(self):
        self.currentiteratorevent = self.iteratorlogmanager.geteventbyid(-1)
        self.currentiteratorevent.distributionmatrix = self.getfractaldistributionmatrix(self.getfractalpredictionsvectors())
        self.iteratorlogmanager = IteratorLogManager(self.currentiteratorevent,self.fractalid)
    @staticmethod
    def iteratefractalbyfractalid(fractalid, radiuslist,depthfractalresolution,subdirectory):
        fractaldepthsamplor = FractalDepthSamplor(radiuslist, depthfractalresolution=depthfractalresolution,subdirectory=subdirectory)
        iterationsmanager = IterationsManager(maxiter=500, fitnessmode="random",
                                              fractaldepthsamplor=fractaldepthsamplor)
        fractaliterator = FractalIterator(iterationsmanager, fractalid=fractalid, selectiontype="random",
                                          actualposition=(0, 0), radius=3, fractalresolution=128, modelresolution=16,
                                          display=False)
        generatetrainenvironnementbyfractalid(fractalid)

def generatetrainenvironnementbyfractalid(fractalid):
    itlogman = IteratorLogManager(-1,fractalid)
    itlogman = itlogman.loadlogmanager()
    itlogman.generatetrainingenvironnement()

#iteratefractalbyfractalid(2)

#overgenid = itlogman.getparentsidovergenerations(childid=6,generationnumber=4)
#print(itlogman.printdescendancebyid(0))
#test = itlogman.getindividualsfromverticaldescendancethreshold(5)
#print(test)


#fractaliterator.getfollowingchild(selectiontype="min")
# firstfractal = 9821
# for i in range(100):
#     fractaliterator = FractalIterator(fractalid=firstfractal+i,actualposition = (0,0),radius=3,fractalresolution=128,modelresolution=16,display=False)
#     fractaliterator.getfollowingchild(selectiontype="min")
#     fractaliterator = FractalIterator(fractalid=firstfractal+i,actualposition = (0,0),radius=3,fractalresolution=128,modelresolution=16,display=False)
#     fractaliterator.getfollowingchild(selectiontype="max")
# fractaliterator.getfollowingchild(selectiontype="max")
# fractaliterator = FractalIterator(fractalid=20001,actualposition = (0,0),radius=3,fractalresolution=128,modelresolution=16,display=False)
# fractaliterator.getfollowingchild(selectiontype="min")
# fractaliterator.getfollowingchild(selectiontype="max")
#print(fractaliterator.iteratorlogmanager)