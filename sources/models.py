import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, cm
from torch import nn, Tensor
import torch.nn.functional as F
import sources.debugging as debugging
from sources.globalvariables import gettraininghyperparameter, getbasedirectory


class FractalModel(torch.nn.Module):
    visualdebugger: debugging.TensorVisualization
    def __init__(self,visualdebugger):
        super(FractalModel, self).__init__()
        self.visualdebugger = visualdebugger
        self.linear1 = torch.nn.Linear(22, 2000).double()
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2000, 1000).double()
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1000, 1).double()

    def forward(self, x):
        #print("input", x)
        #print("dtype", x.dtype)
        x = x.to(torch.float64)
        #print("dtype", x.dtype)
        #self.visualdebugger.addcol(x, 1)
        x = self.linear1(x)
        #print("before",x.shape)
        #x = self.mutation(x,0.5)
        #print("after",x)
        #self.visualdebugger.addcol(x, 2)
        x = self.activation(x)
        #self.visualdebugger.addcol(x, 3)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)

        return x
    def mutation(self,tensor,mutationrate):
        totalelements = 1
        for i in range(len(tensor.shape)):
            totalelements *= len(tensor[i])
        elementsmutated = int(totalelements * mutationrate)
        print("elementsmutated")
        print(elementsmutated)
        dimen = torch.zeros(len(tensor.shape))
        for i in range(0,elementsmutated):
            j = int(torch.rand(1).item() * len(tensor.shape))
            k = int(tensor.shape[j] * torch.rand(1).item())
            print("j",j,"k",k)
            tensor[j][k] = torch.rand(1)
        return tensor


class VisualLayerModel(nn.Module):
    visualdebugger: debugging.TensorVisualization
    fig:plt.figure
    def __init__(self):
        super().__init__()
        self.i = 0
        self.visualdebugger = debugging.TensorVisualization()
        self.conv1 = nn.Conv2d(3, 6, 3,padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 9, 3,padding=1)

        self.fc1 = nn.Linear(9216, 2304)
        nn.init.normal_(self.fc1.weight)
        self.fc2 = nn.Linear(2304, 576)
        self.fc3 = nn.Linear(576, 144)
        self.fc4 = nn.Linear(144, 16)
    def forward(self, x,batchinfos=(-1,-1),debug = False):
        self.debug = debug
        #print("x0", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print("x1",x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("x2", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("x3", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        self.debugstep(x,batchinfos)
        return x
    def debugstep(self,x,batchinfos):
        if batchinfos == (-1,-1):
            return
        dataloaderbatchsize = gettraininghyperparameter("dataloaderbatchsize")
        batchindex,epochindex = batchinfos
        if self.debug:


            tensorvalue = torch.clone(Tensor.cpu(x)).detach()
            #print(tensorvalue.shape)
            #Conv layer ex.
            #tensorvalue = tensorvalue.view(18*3*dataloaderbatchsize,18,3)
            #im = Image.fromarray(np.uint8(cm.gist_stern(tensorvalue) * 255), mode="RGBA")
            tensorvalue = tensorvalue.view(16,-1)
            #print(tensorvalue.shape)
            # Linear layer ex
            im = Image.fromarray(np.uint8(cm.gist_stern(tensorvalue) * 255))

            im.save(getbasedirectory("testvisual")  + "E" + str(epochindex) + "B" + str(batchindex) + ".png")


    @staticmethod
    def getmodelfromsubdirectory(subdirectory):
        if torch.cuda.is_available():
            model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt", map_location=torch.device('cuda'))

        else:
            model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt", map_location=torch.device('cpu'))
        return model






def getmodel(modelpath):
    if torch.cuda.is_available():
        model = torch.load(modelpath, map_location=torch.device('cuda'))
    else:
        model = torch.load(modelpath, map_location=torch.device('cpu'))
    return model