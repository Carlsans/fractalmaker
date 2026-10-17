import torch


class IteratorEvent:
    id: int
    parentid:int
    eventposition: (float, float)
    eventradius: float
    distributionmatrix:torch.tensor
    note:int
    averredcenters:list
    def __init__(self,id,parentid,eventposition,eventradius):
        self.id = id
        self.parentid = parentid
        self.eventposition = eventposition
        self.eventradius = eventradius
        self.averredcenters = []
    def getdistributionmatrix(self):
        return self.distributionmatrix

    def setdistributionmatrix(self,distributionmatrix):
        self.distributionmatrix = distributionmatrix

    def addaverredcenter(self,childid,center):
        self.averredcenters.append([childid,center])

    def __str__(self):
        description = ""
        description += "Id = "+ str(self.id) + " Parentid = " + str(self.parentid) +"\n"
        description += "Position = " + str(self.eventposition) + " Radius = " + str(self.eventradius) +"\n"
        description += str(self.distributionmatrix) + "\n"
        description += str(self.averredcenters) + "\n"
        return  description
