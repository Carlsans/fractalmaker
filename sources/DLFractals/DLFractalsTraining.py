import random

import torch
import winsound
#from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import profile

from sources.DLFractals.EpochDot import EpochDot
from sources.DLFractals.TrainingFunctions import TrainingFunctions
from sources.DLFractals.models import Linear_Model
from sources.globalvariables import getbasedirectory
from torch import nn

class DLFractalsTraining:
    optimizer: torch.optim.SGD
    totalsteps: int
    totalepoch: int

    def __init__(self, dlfractal,trainingtensor, startfromscratch=True, debugmode=False):
        self.dlfractal = dlfractal
        self.dlfractalconf = dlfractal.dlfractalconf

        self.model = dlfractal.model
        self.epochdot = EpochDot()
        self.startfromscratch = startfromscratch
        self.debugmode = debugmode
        self.totalsteps = 0
        self.totalepoch = 0
        self.lostfunctionresolution = 256
        self.last_loss = 0
        self.resolution = 256

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters())
        #self.optimizer = torch.optim.Adadelta(self.model.parameters())
        self.trainingtensor = trainingtensor
    def convergencetest(self):
        loss_list = []
        for t in range(100):
            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            loss_list.append(loss)
            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()
            print('\repoch: {}\tLoss =  {:.3f}'.format(t, loss), end="")
    def customloss(self):
        #result = torch.exp(abs(outputs + values))
        #result = torch.sum(result) / result.shape[0]
        # print(result)
        result = 1 /self.dlfractal.getmodelfractalentropy((self.resolution,self.resolution))
        return result

    def entropyloss(self,output,lastloss):
        lsm = nn.LogSoftmax(dim=0)
        log_probs = lsm(output)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        entropy = -p_log_p.mean()
        return 10 - entropy



    def train_one_epoch(self,epoch):
        #for name, param in self.model.named_parameters():
        #    print('name',name,'param',param)
        self.model.train()
        running_loss = 0.
        #loss_fn = self.entropyloss
        #loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.L1Loss()
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.optimizer.zero_grad()
        #noise = (torch.rand(self.resolution*self.resolution,1,device=self.device,dtype=torch.complex128))*0.01+(torch.rand(self.resolution*self.resolution,1,device=self.device,dtype=torch.complex128))*0.01j

        #inputs = self.getmatrixlinearinterpolation()# + noise
        inputs = self.dlfractal.generate2dcomplexlinspace().flatten()

        inputs = inputs + torch.rand(inputs.shape,device=self.device,dtype=torch.complex128)*.001##
        #print("inputs.real.shape",inputs.real.shape,"inputs.imag.shape",inputs.imag.shape)

        outputs = self.model(torch.flatten(inputs.real),torch.flatten(inputs.imag),returnascomplex=True,printattentionmean=False)
        #formatedinputs = torch.stack((torch.flatten(inputs.real), torch.flatten(inputs.imag))).T

        #outputs = self.model(formatedinputs)

        #print("outputs.shape",outputs.shape)
        #outputs = self.model(noise)

        #target = torch.zeros(outputs.shape,device=self.device,dtype=torch.double)+2
        if self.trainingtensor == []:
            if self.switch:
                target = TrainingFunctions.trainingfunctioncaller(1,inputs)
                self.switch = False
            else:
                target = TrainingFunctions.trainingfunctioncaller(2, inputs)
                self.switch = True

            #print(target)
            #minustarget = TrainingFunctions.trainingfunctioncaller(0,inputs)
            #print("target",target)
            #target = self.dlfractal.generate2dcomplexlinspace().flatten()+torch.pow(self.dlfractal.generate2dcomplexlinspace().flatten(),5)*2.23+1.2j## LOSS =7 ou 8
        else:
            target = self.trainingtensor
        # infinity = 1000000000
        # torch.nan_to_num_(target.real,posinf=infinity,neginf=infinity)
        # torch.nan_to_num_(target.imag, posinf=infinity, neginf=infinity)
        # torch.nan_to_num_(minustarget.real, posinf=infinity, neginf=infinity)
        # torch.nan_to_num_(minustarget.imag, posinf=infinity, neginf=infinity)
        # Conversion vers coordonnée cartésienne au lieu de complexe
        #target =
        #formatedtarget= torch.stack((torch.flatten(target.real), torch.flatten(target.imag))).T
        loss = loss_fn(outputs,target)
        #sig = self.donothing
        #loss = triplet_loss(target,outputs,minustarget)
        #print("outputs",outputs.shape,"target",target.shape,"minustarget",minustarget.shape)

        #if self.dlfractal.bigbatch == 1 and self.dlfractal.minibatch == 0:
        print("loss",loss.item())
        #print("loss", '{:.2e}'.format(loss.item()) ," ",end="")
        self.last_loss = loss
        #with torch.no_grad():
        #    self.last_loss = loss.clone()
        self.optimizer.zero_grad()
        loss.backward()


        # Adjust learning weights
        self.optimizer.step()
        return loss.item()

    def donothing(self,x):
        return x
    def trainmultipleepochs(self, epochcount, startfromscratch, savefile=True):
        self.switch = False
        runningloss = 0
        if not startfromscratch:
            self.model = torch.load(self.dlfractalconf.modelpath)

        for i in range(0, epochcount):
            # winsound.Beep(500, 500)
            self.totalepoch += 1
            if i == 0:
                print("Epoch ",end="")
            self.epochdot.displaydot()
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.model.train(True)
            runningloss += self.train_one_epoch(i)
            self.model.train(False)

            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            if savefile:
                torch.save(self.model, self.dlfractalconf.modelpath)
            # if i % 1 == 0 or i == epochcount-1:
            #     testtrainingset(self.model)

        # reactiver pour debugself.tensorvisualizer.showcurrentmatrix()
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        #winsound.Beep(frequency, duration)
        # UserInteraction.createusersbatchs(batchnumber=10, batchindex=creationbatch)
        # Reload training set after user interaction
        # self.trainingsethandler.loadtrainingset()
        return runningloss/epochcount