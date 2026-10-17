from sources.globalvariables import getbasedirectory


class DLFractalConfiguration():
    def __init__(self):
        self.debugforwarding = False
        self.printfullvariablecontent = False
        self.trackmemoryusage = False
        self.modelpath = getbasedirectory("dlfractaltraining") + "fractalNN.pt"

