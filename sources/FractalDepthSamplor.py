import calendar
import copy
import time

from sources.fractals import Fractal


class FractalDepthSamplor:
    approximateradiuslist: list
    currentradius: float
    retainedfractalsparameters: list
    radiuslistindex:int
    depthfractalresolution:(int,int)
    subdirectory:str
    def __init__(self,approximateradiuslist,depthfractalresolution = (256,256),subdirectory=""):
        self.approximateradiuslist = approximateradiuslist
        self.currentradius = self.approximateradiuslist[0]
        self.retainedfractalsparameters = []
        self.radiuslistindex = 0
        self.depthfractalresolution = depthfractalresolution
        self.subdirectory=subdirectory
    def testradius(self,fractalparameters):
        if self.currentradius > fractalparameters.getrangesdimensions()[0] / 2:
            return True
        else:
            return False
    def appendfractalparameters(self,fractalparameters):
        self.retainedfractalsparameters.append(copy.copy(fractalparameters))
        self.radiuslistindex += 1
        if self.radiuslistindex < len(self.approximateradiuslist):
            self.currentradius = self.approximateradiuslist[self.radiuslistindex]
        print("self.retainedfractalsparameters",self.retainedfractalsparameters)
    def lastiterationfinished(self):
        if self.radiuslistindex > (len(self.approximateradiuslist)-1):
            return True
        else:
            return False

    def saveallfractalstodisk(self,fractaliterator):
        n=0
        for fractalparameters in self.retainedfractalsparameters:
            fractal = Fractal(fractalparameters)
            print("thefractalparameters",fractalparameters)
            current_GMT = time.gmtime()
            time_stamp = calendar.timegm(current_GMT)
            fractal.generatefractalc(iteration=fractaliterator.fractalid,suffix="zoom"+str(n)+"ts"+str(time_stamp),overridedimensions=self.depthfractalresolution,subdirectory=self.subdirectory,display=False)

            n += 1
