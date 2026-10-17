import random

import torch

from sources.FractalDepthSamplor import FractalDepthSamplor


class IterationsManager:
    maxiter:int
    currentiter:int
    fitnessmode:str
    stepbackward:int
    trynumber:int
    wrongtrybackwardsteps:int
    fractaldepthsamplor:FractalDepthSamplor
    def __init__(self,maxiter,fitnessmode,fractaldepthsamplor=None):
        self.maxiter = maxiter
        if fitnessmode != "random":
            self.fitnessmode = fitnessmode
        else:
            self.fitnessmode = random.choice(["getapproximateradiusfromlist","straight","backward"])
        self.currentiter = -1

        self.stepbackward = 1
        self.stepbackwardentropylevel = 1
        self.trynumber = 6
        self.wrongtrybackwardsteps = 9
        self.fractaldepthsamplor = fractaldepthsamplor

    def fitnessfunc(self,iterationmanager):
        self.currentiter +=1
        if self.currentiter > self.maxiter:
            self.enditeration(iterationmanager)
            return False
        if self.fitnessmode == "getapproximateradiusfromlist":
            # Fractal radius list exhausted
            if(self.fractaldepthsamplor.lastiterationfinished()) :
                print("lastiterationfinished")
                self.fractaldepthsamplor.saveallfractalstodisk(iterationmanager)
                self.enditeration(iterationmanager)
                return False
            print("iterationmanager.currentfractal.parameters",iterationmanager.currentfractal.parameters)
            # The asked radius is reached, restart from the start fractal
            if(self.fractaldepthsamplor.testradius(iterationmanager.currentfractal.parameters)):
                print("iterationmanager.currentfractal.parametersafter", iterationmanager.currentfractal.parameters)
                self.fractaldepthsamplor.appendfractalparameters(iterationmanager.currentfractal.parameters)
                iterationmanager.restartfromfirstevent()
                iterationmanager.getfollowingchild()
                return False
            # Else, continue iterating the fractal until next radius is reached
            fractalentropy = iterationmanager.currentfractal.getfractalentropy(resolution=16)
            if fractalentropy > self.stepbackwardentropylevel:
                iterationmanager.getfollowingchild()
            else:
                print("entropy insuffisant, going back...")
                print("currentid", iterationmanager.iteratorlogmanager.lastelementid)
                previouseventid = iterationmanager.iteratorlogmanager.getparentsidovergenerations(
                    childid=iterationmanager.iteratorlogmanager.lastelementid, generationnumber=self.stepbackward)
                backwardevent = iterationmanager.iteratorlogmanager.geteventbyid(previouseventid)
                backwardevent = self.limittrynumber(backwardevent, iterationmanager)
                iterationmanager.currentiteratorevent = backwardevent
                print("backwardeventid", backwardevent.id)
                print("backwardeventDM", backwardevent.distributionmatrix)
                #iterationmanager.flipselectiontype()
                iterationmanager.getfollowingchild()

        if self.fitnessmode == "straight":
            fractalentropy = iterationmanager.currentfractal.getfractalentropy(resolution=16)
            print("Entropy",fractalentropy)
            if fractalentropy > 1:
                iterationmanager.getfollowingchild()
            else:
                print("Entropy has reached inferior limit")
                self.enditeration(iterationmanager)
                return False
        if self.fitnessmode == "backward":
            fractalentropy = iterationmanager.currentfractal.getfractalentropy(resolution=16)
            if fractalentropy > self.stepbackwardentropylevel:
                iterationmanager.getfollowingchild()
            else:
                print("entropy insuffisant, going back...")
                print("currentid",iterationmanager.iteratorlogmanager.lastelementid)
                previouseventid = iterationmanager.iteratorlogmanager.getparentsidovergenerations(childid=iterationmanager.iteratorlogmanager.lastelementid, generationnumber=self.stepbackward)
                backwardevent = iterationmanager.iteratorlogmanager.geteventbyid(previouseventid)
                backwardevent = self.limittrynumber(backwardevent,iterationmanager)
                iterationmanager.currentiteratorevent = backwardevent
                print("backwardeventid", backwardevent.id)
                print("backwardeventDM",backwardevent.distributionmatrix)
                iterationmanager.flipselectiontype()
                iterationmanager.getfollowingchild()
    def limittrynumber(self, event,iterationmanager):
        nancount = torch.numel(torch.nonzero(torch.isnan(event.distributionmatrix.view(-1))))
        if nancount > self.trynumber:
            eventid = iterationmanager.iteratorlogmanager.getparentsidovergenerations(
                childid=event.id, generationnumber=self.wrongtrybackwardsteps)
            event = iterationmanager.iteratorlogmanager.geteventbyid(eventid)
            print("depasse")
        return event



    def enditeration(self,iterationmanager):
        iterationmanager.iteratorlogmanager.savelogmanager()