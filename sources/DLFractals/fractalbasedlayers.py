import random
from collections import OrderedDict

import seaborn as seaborn
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu
class LinearLayer(nn.Module):
    def __init__(self, dlfractalconf,numberoflayers,hiddenlayerssize,outputsize = 2,iscomplex =False):
        super().__init__()
        self.iscomplex = iscomplex

        if iscomplex:
            dtype = torch.complex128
            inputsize = 1
            activationfunction = phase_amplitude_relu()
        else :
            dtype = torch.double
            inputsize = 2
            activationfunction = nn.LeakyReLU()

        layers = []
        dropoutlayer = []
        layers.append(nn.Linear(inputsize, hiddenlayerssize,dtype=dtype))
        layers.append(activationfunction)
        for i in range(numberoflayers-2):
            if i in dropoutlayer:
                layers.append(nn.Dropout(.1))
            layers.append(nn.Linear(hiddenlayerssize, hiddenlayerssize,dtype=dtype))
            layers.append(activationfunction)
        layers.append(nn.Linear(hiddenlayerssize, outputsize,dtype=dtype))
        self.sequentialmodel = nn.Sequential(*layers)
        print(self)
        self.sequentialmodel.apply(self.init_weights)
    def xycoordinatetocomplex(self,x):
        print(x.shape)
        exit()



    def init_weights(self,m):
        #print(m)
        if type(m) == nn.Linear:
            #m.weight.data.fill_(1.0)
            torch.nn.init.xavier_uniform_(m.weight)
            #print(m.weight)
    def forward(self, xvalue,yvalue,returnascomplex=False):
        #print(xvalue.shape,yvalue.shape)
        if self.iscomplex:
            x = xvalue + yvalue * 1j
            x = torch.unsqueeze(x,dim=1)

        else:
            x = torch.stack((xvalue, yvalue)).T

        #print(x.shape)
        x = self.sequentialmodel(x)

        if returnascomplex:
            return x[:,0]+x[:,1]*1j
        #print(x.shape)
        return x

    def printforwarding(self,comment,forwarding):
        if self.dlfractalconf.debugforwarding or self.dlfractalconf.trackmemoryusage:
            print(comment)
            print("shape",forwarding.shape)
            if self.dlfractalconf.printfullvariablecontent:
                print(forwarding)
            if self.dlfractalconf.trackmemoryusage:
                print("Memory allocated (Mb)",int(torch.cuda.memory_allocated()/1_000_000))

    def real_imaginary_relu(self,z):
        #return torch.sigmoid(z)
        return F.relu(z.real) + 1.j * F.relu(z.imag)
    def printweights(self):
        for name, param in self.named_parameters():
            print(name,param)

class MathLayer(nn.Module):
    def __init__(self, dlfractalconf,complexmathfunction):
        super().__init__()
        self.complexmathfunction = complexmathfunction


    def forward(self, xvalue,yvalue,returnascomplex=True):
        #print(xvalue.shape,yvalue.shape)
        x = xvalue + yvalue * 1j
        #print(x.shape)
        x = self.complexmathfunction(x)

        if returnascomplex:
            return x
        return (x.real,x.imag)
class CompositeModel(nn.Module):
    def __init__(self, dlfractalconf):
        super().__init__()
        self.seriemodel1 = LinearMathwithAttentionModel(dlfractalconf)
        self.parallelmodel1 = LinearMathwithAttentionModel(dlfractalconf)
        self.parallelmodel2 = LinearMathwithAttentionModel(dlfractalconf)
        self.summodel1 = LinearMathwithAttentionModel(dlfractalconf)
    def forward(self, xvalue, yvalue, returnascomplex=True,printattentionmean=False):
        #x = self.seriemodel1(xvalue, yvalue)
        #x = self.seriemodel1(x.real, x.imag)

        resultparallel1 = self.parallelmodel1(xvalue, yvalue,printattentionmean=printattentionmean)
        resultparallel1 = self.parallelmodel1(resultparallel1.real, resultparallel1.imag,printattentionmean=printattentionmean)
        resultparallel1 = self.parallelmodel1(resultparallel1.real, resultparallel1.imag,
                                              printattentionmean=printattentionmean)

        resultparallel2 = self.parallelmodel2(xvalue, yvalue,printattentionmean=printattentionmean)
        # resultparallel2 = self.parallelmodel2(resultparallel2.real, resultparallel2.imag,
        #                                       printattentionmean=printattentionmean)
        # resultparallel2 = self.parallelmodel2(resultparallel2.real, resultparallel2.imag,
        #                                       printattentionmean=printattentionmean)
        x = resultparallel1 * resultparallel2
        return x

class LinearMathwithAttentionModel(nn.Module):
    def __init__(self, dlfractalconf):
        super().__init__()
        linearlayerweightcount = 100
        linearlayerlayerscount = 3
        complexlinlayer = False
        self.attentionlayer = LinearLayer(dlfractalconf,3,100,outputsize = 7)
        self.linearlayer1 = LinearLayer(dlfractalconf,linearlayerlayerscount,linearlayerweightcount,iscomplex=complexlinlayer)
        self.linearlayer2 = LinearLayer(dlfractalconf, linearlayerlayerscount, linearlayerweightcount,iscomplex=complexlinlayer)
        self.linearlayer3 = LinearLayer(dlfractalconf, linearlayerlayerscount, linearlayerweightcount,iscomplex=complexlinlayer)
        self.linearlayer4 = LinearLayer(dlfractalconf, linearlayerlayerscount, linearlayerweightcount,iscomplex=complexlinlayer)
        self.linearlayer5 = LinearLayer(dlfractalconf, linearlayerlayerscount, linearlayerweightcount,iscomplex=complexlinlayer)
        self.linearlayer6 = LinearLayer(dlfractalconf, linearlayerlayerscount, linearlayerweightcount,iscomplex=complexlinlayer)


        self.linearlayerstraight = LinearLayer(dlfractalconf, 4, 100)

        self.mathlayer1 = MathLayer(dlfractalconf,torch.sin)
        self.mathlayer2 = MathLayer(dlfractalconf, torch.asin)
        self.mathlayer3 = MathLayer(dlfractalconf, torch.exp)
        self.mathlayer4 = MathLayer(dlfractalconf, torch.log)
        self.mathlayer5 = MathLayer(dlfractalconf, torch.sigmoid)
        self.mathlayer6 = MathLayer(dlfractalconf, torch.square)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xvalue, yvalue, returnascomplex=True,printattentionmean=False):

        attention = self.softmax(self.attentionlayer(xvalue, yvalue))
        if printattentionmean:
            with torch.no_grad():
                print(torch.sum(attention,dim=0)/attention.shape[0])

        result1 = self.linearlayer1(xvalue, yvalue)
        result1 = self.mathlayer1(result1[:,0],result1[:,1])
        #print(result1.shape)
        #print(attention.shape)
        result1 = result1 * attention[:,0]

        result2 = self.linearlayer2(xvalue, yvalue)
        result2 = self.mathlayer2(result2[:, 0], result2[:, 1])
        result2 = result2 * attention[:, 1]

        result3 = self.linearlayer3(xvalue, yvalue)
        result3 = self.mathlayer3(result3[:, 0], result3[:, 1])
        result3 = result3 * attention[:, 2]

        result4 = self.linearlayer4(xvalue, yvalue)
        result4 = self.mathlayer4(result4[:, 0], result4[:, 1])
        result4 = result4 * attention[:, 3]

        result5 = self.linearlayer5(xvalue, yvalue)
        result5 = self.mathlayer5(result5[:, 0], result5[:, 1])
        result5 = result5 * attention[:, 4]

        result6 = self.linearlayer6(xvalue, yvalue)
        result6 = self.mathlayer6(result6[:, 0], result6[:, 1])
        result6 = result6 * attention[:, 5]

        resultstraight = self.linearlayerstraight(xvalue, yvalue,returnascomplex=True)
        #print(resultstraight.shape)
        resultstraight = resultstraight * attention[:,6]

        x = result1 + result2 + result3 + result4 + result5 + result6 + resultstraight
        #x = torch.relu(x.real)+torch.relu(x.imag)*1j
        return x

class phase_amplitude_relu(nn.Module):
    def __init__(self):
        super(phase_amplitude_relu, self).__init__()

    def forward(self, z):
        return relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

    #def phase_amplitude_relu(z):
    #    return relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))
#alinearlayer = LinearLayer(None)
#alinearlayer.printweights()