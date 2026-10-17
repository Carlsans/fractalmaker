import copy

from sources import trainingenvironnement
from sources.IteratorEvent import IteratorEvent
import dill

from sources.fractals import FractalParameters, Fractal
from sources.globalvariables import getbasedirectory


class IteratorLogManager:
    iteratoreventlist: list
    lastelementid:int
    fractalid:int
    def __init__(self,parentiteratorevent,fractalid):
        self.iteratoreventlist = [parentiteratorevent]
        self.lastelementid = -1
        self.fractalid = fractalid
    def append(self, iteratorevent):
        self.iteratoreventlist.append(iteratorevent)
    def addnewiteratorevent(self,parentid,eventposition,eventradius):
        self.lastelementid += 1
        iteratorevent = IteratorEvent(self.lastelementid,parentid,eventposition,eventradius)
        self.append(iteratorevent)
        return iteratorevent

    def geteventbyid(self, id):
        my_filter_iter = filter(lambda x: x.id == id, self.iteratoreventlist)

        return next(my_filter_iter)
    def getchildlistbyid(self,parentid):
        childlist = []
        for i in range(parentid,self.lastelementid):
            child = self.geteventbyid(i)
            if child.parentid == parentid:
                childlist.append(child.id)
        return childlist
    def printdescendancebyid(self,parentid):

        childlist = self.getchildlistbyid(parentid)
        print("parentid",parentid," childlist",childlist)
        for i in childlist:
            self.printdescendancebyid(i)
    def getindividualsfromverticaldescendancethreshold(self,threshold):
        individuals = []
        for i in range(self.lastelementid):
            if self.getverticalchildnumberbyid(i) > threshold:
                individuals.append(i)
        return individuals
    def generatetrainingenvironnement(self):
        selectedindividuals = self.getindividualsfromverticaldescendancethreshold(threshold=5)
        iteration = 0
        for i in selectedindividuals:
            self.generatefractalbyeventid(i,iteration=iteration)
            iteration+=1

    def generatefractalbyeventid(self,eventid,iteration = -1,fractalresolution = 128):
        event = self.geteventbyid(eventid)
        fractalparameters = FractalParameters.defaultparameters()
        fractalparameters.fractaldimensions = (fractalresolution, fractalresolution)
        fractalparameters.setweightsfromfractalid(self.fractalid)
        fractalparameters.setrangefromcentertoside(event.eventposition, (event.eventradius, event.eventradius))
        fractal = Fractal(fractalparameters)
        subdirectory = "plunge1\\train\\"  + str(self.fractalid)
        fractal.generatefractalc(iteration=iteration,
                                             subdirectory=subdirectory,
                                             display=False)
        averredcenter = event.averredcenters[0][1]
        trainingenvironnement.savetrainresultfile(subdirectory=subdirectory,filenumber=iteration,results=(averredcenter[0],averredcenter[1],event.eventradius))

    def getverticalchildnumberbyid(self,parentid,count=0):
        childlist = self.getchildlistbyid(parentid)
        #print("parentid", parentid, " childlist", childlist)
        #print(count)
        if not childlist:
            return count
        for i in childlist:
            return self.getverticalchildnumberbyid(i,count+1)


    def printeveryelementchildlist(self):
        for i in range(self.lastelementid):
            print("Id="+str(i)+" Child list: ")
            print(self.getchildlistbyid(i))
    def savelogmanager(self):
        with open(getbasedirectory(subdirectory="plunge1")+"logmanager"+str(self.fractalid)+".pkl", 'wb') as f:
            dill.dump(self, f)
    def loadlogmanager(self):
        with open(getbasedirectory(subdirectory="plunge1")+"logmanager"+str(self.fractalid)+".pkl", 'rb') as f:
            loadeddill = dill.load(f)
        return loadeddill
    def getparentsidovergenerations(self,childid,generationnumber):
        childiteratorevent = self.geteventbyid(childid)
        for i in range(generationnumber):
            parentid = childiteratorevent.parentid
            childiteratorevent = self.geteventbyid(parentid)

        return parentid
    def __str__(self):
        description = ""

        for i in self.iteratoreventlist:
            description += str(i)
        return description