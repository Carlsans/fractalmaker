import random

import seaborn as seaborn
import torch
import torch.nn.functional as F
import torch.nn as nn
class Linear_Model(nn.Module):
    def __init__(self, dlfractalconf):
        super().__init__()
        self.dlfractalconf = dlfractalconf
        latentdim=5
        self.tcl1 = torch.nn.ConvTranspose1d(2, 3, 2,dtype=torch.double)
        #self.tcl2 = torch.nn.ConvTranspose1d(5, 15, 4, dtype=torch.double)
        self.intermediatelayer = nn.Linear(6, 6, dtype=torch.double)
        self.lastlayer = nn.Linear(6, 2, dtype=torch.double)
        torch.nn.init.xavier_uniform_(self.tcl1.weight)
        #torch.nn.init.xavier_uniform_(self.tcl2.weight)
        torch.nn.init.xavier_uniform_(self.lastlayer.weight)
        # self.lm1 = nn.Linear(1, latentdim,dtype=torch.complex128)
        # self.lm2 = nn.Linear(latentdim, latentdim,dtype=torch.complex128)
        # self.lm3 = nn.Linear(latentdim, latentdim,dtype=torch.complex128)
        # self.lm4 = nn.Linear(latentdim, latentdim, dtype=torch.complex128)
        # self.lm5 = nn.Linear(latentdim, latentdim, dtype=torch.complex128)
        # self.lm6 = nn.Conv1d(latentdim, 128, 15, stride=1,dtype=torch.complex128)
        #
        # self.lm7 = nn.Linear(512, latentdim, dtype=torch.complex128)
        # self.lm8 = nn.Linear(latentdim, latentdim, dtype=torch.complex128)
        # self.lm9 = nn.Linear(128, 1, dtype=torch.complex128)
        # torch.nn.init.xavier_uniform_(self.lm1.weight)
        # torch.nn.init.xavier_uniform_(self.lm2.weight)
        # torch.nn.init.xavier_uniform_(self.lm3.weight)
        # torch.nn.init.xavier_uniform_(self.lm4.weight)
        # torch.nn.init.xavier_uniform_(self.lm5.weight)
        # torch.nn.init.xavier_uniform_(self.lm6.weight)
        # torch.nn.init.xavier_uniform_(self.lm7.weight)
        # torch.nn.init.xavier_uniform_(self.lm8.weight)
        # torch.nn.init.xavier_uniform_(self.lm9.weight)
        # print("self.lm1.weight",self.lm1.weight)
        # print("self.lm2.weight",self.lm2.weight)
        # print("self.lm3.weight",self.lm3.weight)
    def complexdropout(self,vector,probability):
        return vector
        if self.train():
            mask = torch.nn.functional.dropout(torch.ones_like(vector.real),probability)
            return vector * mask
        return vector

    def forward(self, xreal,ximag,returnascomplex=False):

        x = torch.stack((xreal,ximag),dim=1).unsqueeze(2)
        self.printforwarding("x1", x)
        #print("x.shape",x.shape)
        x = self.tcl1(x)
        self.printforwarding("x2:tcl1", x)
        if self.training:
            x = torch.nn.functional.dropout(x,p=.01)
        #x = self.tcl2(torch.relu(x))
        #print("x.shape2", x.shape)
        x = x.reshape(x.size(0),  -1)
        self.printforwarding("x3:reshape", x)
        #x = torch.transpose(x.reshape(x.size(0),x.size(1), -1),0,1)
        #print(x)
        #print("x.size(1)",x.size(1))
        #print("x.shape3", x.shape)
        #x = torch.permute(x, (1, 0))
        activation = nn.LeakyReLU(0.2)
        #x = x[:,0]*x[:,1]+x[:,2],x[:,3]*x[:,4]+x[:,5]
        x = self.intermediatelayer(activation(x))
        x = self.lastlayer(activation(x))
        self.printforwarding("x4:lastlayer", x)
        #print("x.shape4", x.shape)
        if returnascomplex:
            return x[:,0]+x[:,1]*1j

        #print(x[:,0])
        return x[:,0]+x[:,1]
    def printforwarding(self,comment,forwarding):
        if self.dlfractalconf.debugforwarding:
            print(comment)
            print(forwarding)
    def real_imaginary_relu(self,z):
        #return torch.sigmoid(z)
        return F.relu(z.real) + 1.j * F.relu(z.imag)
    def printweights(self):
        for name, param in self.named_parameters():
            print(name,param)


class MultiplicativeModel(nn.Module):
    def __init__(self, dlfractalconf):
        super().__init__()
        self.dlfractalconf = dlfractalconf
        linearlayersoutputs = 70
        self.scaler = 1000000
        self.lnatt = nn.Linear(2, linearlayersoutputs, dtype=torch.double)
        self.ln1 = nn.Linear(2, linearlayersoutputs, dtype=torch.double)
        self.ln1l2 = nn.Linear(linearlayersoutputs, linearlayersoutputs, dtype=torch.double)
        self.ln1l3 = nn.Linear(linearlayersoutputs, linearlayersoutputs, dtype=torch.double)
        self.ln2 = nn.Linear(2, linearlayersoutputs, dtype=torch.double)
        self.ln3 = nn.Linear(40, linearlayersoutputs, dtype=torch.double)
        self.lstm_layer = nn.LSTM(2, 40, 3, batch_first=True, dtype=torch.double)
        #self.scalinglayer = nn.Linear(2, linearlayersoutputs, dtype=torch.double)
        self.powlayer1 = nn.Linear(2, 32, dtype=torch.double)
        self.powlayer2 = nn.Linear(32, 1, dtype=torch.double)

        #self.scalinglayer1 = nn.Linear(2, 5, dtype=torch.double)
        #self.scalinglayer2 = nn.Linear(5, 2, dtype=torch.double)
        self.lastlayer = nn.Linear(linearlayersoutputs, 2, dtype=torch.double)
        torch.nn.init.xavier_uniform_(self.lnatt.weight)
        torch.nn.init.xavier_uniform_(self.ln1.weight)
        torch.nn.init.xavier_uniform_(self.ln2.weight)
        torch.nn.init.xavier_uniform_(self.ln3.weight)
        torch.nn.init.xavier_uniform_(self.lastlayer.weight)
        self.dropout = nn.Dropout(0.1)

        #torch.nn.init.xavier_uniform_(self.scalinglayer.weight)
        torch.nn.init.zeros_(self.powlayer1.weight)
        torch.nn.init.zeros_(self.powlayer2.weight)
        #torch.nn.init.xavier_uniform_(self.powlayer1.weight)
        #torch.nn.init.xavier_uniform_(self.powlayer2.weight)

        #self.tcl1 = torch.nn.ConvTranspose1d(2, 3, 2,dtype=torch.double)
        #self.intermediatelayer = nn.Linear(6, 6, dtype=torch.double)
        #self.lastlayer = nn.Linear(6, 2, dtype=torch.double)
        #torch.nn.init.xavier_uniform_(self.tcl1.weight)
        #torch.nn.init.xavier_uniform_(self.lastlayer.weight)

    def forward(self, xreal,ximag,returnascomplex=False):
        if self.training and random.randrange(0, 1000) == 1:
            torch.set_printoptions(profile="full")
            print("Model parameters print")
            for name, param in self.named_parameters():
                print(name,param)
            torch.set_printoptions(profile="default")
        activation = nn.LeakyReLU(0.2)
        stdrelu = nn.ReLU()
        activation2 = nn.Sigmoid()
        #activation = nn.ReLU()
        #activation = nn.Sigmoid()
        #x = torch.stack((xreal,ximag),dim=-2).unsqueeze(2)
        torch.nan_to_num_(xreal,posinf=0,neginf=0)
        torch.nan_to_num_(ximag,posinf=0,neginf=0)
        x = torch.stack((xreal, ximag), dim=1).unsqueeze(1)

        #seaborn.heatmap(weights.detach().numpy())
        self.printforwarding("entréeshape", x)
        softmax = nn.Softmax(dim=1)
        resultlnatt = softmax(self.lnatt(x))
        self.printforwarding("resultlnatt", resultlnatt)
        self.printforwarding("entréeshape", x)
        resultln1 = activation(self.ln1(x))
        resultln1 = activation(self.ln1l2(resultln1))
        resultln1 = self.dropout(resultln1)
        resultln1 = activation(self.ln1l3(resultln1))
        self.printforwarding("resultln1p1", resultln1)
        resultln1 = resultln1 * resultlnatt


        # print("resultpowlayer1",resultpowlayer)

        self.printforwarding("resultln1p2", resultln1)
        resultln2 = activation(self.ln2(x)) * resultlnatt
        self.printforwarding("resultln2", resultln2)
        resultlstm = self.lstm_layer(x)[0]
        self.printforwarding("resultlstm", x)
        resultln3 = activation(self.ln3(resultlstm)) * resultlnatt
        self.printforwarding("resultln3p1", resultln3)

        #print("resultpowlayer2",resultpowlayer)


        #resultscalinglayer = self.scalinglayer(x)+ resultpowlayer
        #resultscalinglayer = torch.nan_to_num(resultscalinglayer)
        resultpowlayer = torch.sigmoid(self.powlayer1(x))
        resultpowlayer = torch.sigmoid(self.powlayer2(resultpowlayer))
        resultpowlayer = torch.sigmoid(resultpowlayer)
        resultpowlayer = torch.abs(torch.nan_to_num(torch.logit(resultpowlayer)))
        self.printforwarding("resultpowlayer2", resultpowlayer)
        x = (resultln1 + resultln2 + resultln3)  #* torch.abs(resultpowlayer)
        self.printforwarding("xaddition", x)

        x = self.lastlayer(activation(x)) * resultpowlayer
        self.printforwarding("x lastlayer", x)
        #x = x * torch.abs(resultpowlayer)
        #self.printforwarding("x last layer powed", x)
        x = x.reshape(x.size(0), -1)
        self.printforwarding("x reshaped", x)
        torch.nan_to_num_(x, posinf=0, neginf=0)

        if returnascomplex:
            self.printforwarding("returnascomplex", x[:,0]+x[:,1]*1j)
            return x[:,0]+x[:,1]*1j

        self.printforwarding("returnasrealsum", x[:,0]+x[:,1])
        return x[:,0]+x[:,1]
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





class IterativeModel(nn.Module):
    def __init__(self, dlfractalconf):
        super().__init__()
        self.max_iterations = 100
        self.bailoutvalue = 2
        linearlayersoutputs = 30
        self.dlfractalconf = dlfractalconf

        self.ln1 = nn.Linear(1, linearlayersoutputs, dtype=torch.complex128)
        self.ln2 = nn.Linear(linearlayersoutputs, linearlayersoutputs, dtype=torch.complex128)
        self.ln3 = nn.Linear(linearlayersoutputs, linearlayersoutputs, dtype=torch.complex128)
        self.lastlayer = nn.Linear(linearlayersoutputs, 1, dtype=torch.complex128)
        torch.nn.init.xavier_uniform_(self.lastlayer.weight)
        torch.nn.init.xavier_uniform_(self.ln1.weight)
        torch.nn.init.xavier_uniform_(self.ln2.weight)
        torch.nn.init.xavier_uniform_(self.ln3.weight)


    def forward(self, x,returnascomplex=False):
        if self.training and random.randrange(0, 1000) == 1:
            torch.set_printoptions(profile="full")
            print("Model parameters print")
            for name, param in self.named_parameters():
                print(name,param)
            torch.set_printoptions(profile="default")
        activation = nn.LeakyReLU(0.2)
        stdrelu = nn.ReLU()
        x = x.unsqueeze(1)
        self.printforwarding("firstlayer", x)
        y = torch.zeros(x.shape) - self.max_iterations

        for i in range(self.max_iterations):
            print("iteration",i)
            x = self.ln1(x)
            self.printforwarding("ln1", x)
            x = self.ln2(x)
            self.printforwarding("ln2", x)
            x = self.ln3(x)
            self.printforwarding("ln3", x)
            x = self.lastlayer(x)
            self.printforwarding("lastlayer", x)
            y = torch.where((abs(x.real)+abs(x.imag) > 7) , i, y)
            print("Avant filtre",x)
            x = torch.where((abs(x.real) + abs(x.imag) > 7), float('nan'), x)
            print("Resultat pour iteration",i," : ",x.shape)
            print(x)
        x = x.reshape(x.size(0), -1)
        print("y",y)
        # print("resultpowlayer1",resultpowlayer)


        if returnascomplex:
            return x
            #self.printforwarding("returnascomplex", x[:,0]+x[:,1]*1j)
            #return x[:,0]+x[:,1]*1j

        self.printforwarding("returnasrealsum", x[:,0]+x[:,1])
        return x[:,0]+x[:,1]
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