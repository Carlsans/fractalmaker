import glob
import os
from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import transforms

import sources.debugging as debugging
import sources.globalvariables as globalvariables
from sources.fractals import FractalParameters
from sources.models import FractalModel
import winsound
#from torch.utils.tensorboard import SummaryWriter
import time
from datetime import timedelta

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        imagenumber = 0
        os.chdir(self.img_dir)
        for file in glob.glob("*.txt"):
            imagenumber += 1
        return imagenumber

    def gettxtfilevalue(self, idx):
        #print("gettxtfilevalue", idx)
        with open(globalvariables.getbasedirectory() + str(idx) + '.txt', "r") as myfile:
            data = myfile.read().splitlines()
            if not data[0].isdigit():
                raise Exception("Text file error, file " + globalvariables.getbasedirectory() + str(idx) + '.txt' + "is non digit")
            if int(data[0])<0 or int(data[0])>9:
                raise Exception("Text file error, value must be bound inside 0 and 9 file :" + globalvariables.getbasedirectory() + str(idx) + '.txt')
            return int(data[0])

    def getparamfilevalue(self, idx):
        return torch.load(globalvariables.getbasedirectory() + str(idx) + '.pt')

    def __getitem__(self, idx):
        self.gettxtfilevalue(idx)
        img_path = self.img_dir + str(idx) + ".png"
        image = read_image(img_path)
        label = self.gettxtfilevalue(idx)
        param = self.getparamfilevalue(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, param

@dataclass
class TrainingSetHandler():
    training_set:CustomImageDataset
    training_loader:DataLoader

    def __init__(self):
        self.training_set = CustomImageDataset(img_dir=globalvariables.getbasedirectory())
        self.training_loader = DataLoader(self.training_set, batch_size=64, shuffle=True,num_workers=0)

    def showtrainingset(self, minrange, maxrange):
        figure = plt.figure(figsize=(8, 8))
        rangetotal = maxrange - minrange
        rows = int(rangetotal / 5) + 1
        cols = 5
        for i in range(0, maxrange + 1):
            # sample_idx = torch.randint(i, size=(1,)).item()
            img, label, _ = self.training_set[i + minrange]
            figure.add_subplot(rows, cols, i + 1)
            plt.title(label)
            plt.axis("off")
            plt.imshow(img.squeeze().transpose(0, 2))
        plt.show()

    def loadtrainingset(self):
        self.showtrainingset(minrange=0, maxrange=len(self.training_set)-1)


        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        # Create data loaders for our datasets; shuffle for training, not for validation

        print('Training set has {} instances'.format(len(self.training_set)))
        #print("training_set", self.training_set)
        #print("dataloader", self.training_loader)
        dataiter = iter(self.training_loader)
        images, labels, params = dataiter.next()
        # print(images)
        #print(labels)
        #print(params)

class TrainingParameters:
    #fractalid:int
    optimizertype:int
    learningrate:int
    inputnumber:int
    numberepochs:int

    # Optimizer type constants :
    class Optimizer:
        OPTIMIZERCOUNT=10
        SGD=0
        ASGD=1
        ADADELTA=2
        ADAMAX=3
        RPROP=4
        ADAM=5
        ADAMW=6
        SPARSEADAM=7
        RMSPROP=8
        NADAM=9


        DEFAULTSGDLR= 0.00001


    def __init__(self,optimizertype = 0,learningrate = None,inputnumber = 20, numberepochs = 3):
        self.optimizertype=optimizertype
        self.learningrate=learningrate
        self.inputnumber=inputnumber
        self.numberepochs=numberepochs

        return

    def __str__(self):
        return "Optimizer = " + str(self.optimizertype) + " Learning rate = " + str(self.learningrate) + "\nNumber of inputs = " + str(self.inputnumber) + " Number of epochs = " + str(self.numberepochs)

    def getoptimizer(self,modelparameters):
        if self.optimizertype == self.Optimizer.SGD:
            if self.learningrate != None:
                optimizer = torch.optim.SGD(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.SGD(modelparameters,self.Optimizer.DEFAULTSGDLR)
            return optimizer
        if self.optimizertype == self.Optimizer.ASGD:
            if self.learningrate != None:
                optimizer = torch.optim.ASGD(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.ASGD(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.ADADELTA:
            if self.learningrate != None:
                optimizer = torch.optim.Adadelta(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.Adadelta(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.ADAMAX:
            if self.learningrate != None:
                optimizer = torch.optim.Adamax(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.Adamax(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.RPROP:
            if self.learningrate != None:
                optimizer = torch.optim.Rprop(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.Rprop(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.ADAM:
            if self.learningrate != None:
                optimizer = torch.optim.Adam(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.Adam(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.ADAMW:
            if self.learningrate != None:
                optimizer = torch.optim.AdamW(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.AdamW(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.SPARSEADAM:
            if self.learningrate != None:
                optimizer = torch.optim.SparseAdam(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.SparseAdam(modelparameters)
            return optimizer

        if self.optimizertype == self.Optimizer.RMSPROP:
            if self.learningrate != None:
                optimizer = torch.optim.RMSprop(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.RMSprop(modelparameters)
            return optimizer
        if self.optimizertype == self.Optimizer.NADAM:
            if self.learningrate != None:
                optimizer = torch.optim.NAdam(modelparameters, self.learningrate)
            else:
                optimizer = torch.optim.NAdam(modelparameters)
            return optimizer

    def shiftoptimizer(self):
        if self.optimizertype < self.Optimizer.OPTIMIZERCOUNT - 1:
            self.optimizertype += 1
        else:
            self.optimizertype = 0


class TrainingEvent:
    fractalparameters:FractalParameters
    trainparams: TrainingParameters
    additionalinfo:str
    lastloss:float
    def __init__(self,fractalparameters,trainparams,additionalinfo=""):
        self.fractalparameters=fractalparameters
        self.trainparams=trainparams
        self.additionalinfo=additionalinfo

    def addinfo(self,information):
        self.additionalinfo += '\n' +information


    def __str__(self):
        return str(self.fractalparameters)+"\n"+str(self.trainparams)+"\n"+ str(self.additionalinfo)

class Trainer:
    model:FractalModel
    optimizer:torch.optim.SGD
    trainingsethandler: TrainingSetHandler
    tensorvisualizer: debugging.TensorVisualization
    totalsteps:int
    totalepoch:int
    def __init__(self):
        self.tensorvisualizer = debugging.TensorVisualization()
        self.model = FractalModel(self.tensorvisualizer)
        # Optimizers specified in the torch.optim package
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.totalsteps = 0
        self.totalepoch = 0
    def start(self,epochcount,startfromscratch = True):
        self.trainingsethandler = TrainingSetHandler()
        self.trainingsethandler.loadtrainingset()
        self.trainmultipleepochs(epochcount,startfromscratch)
    def testtimecost(self):
        self.trainingsethandler = TrainingSetHandler()
        self.trainingsethandler.loadtrainingset()
        start = time.time()
        self.trainmultipleepochs(10, startfromscratch = False,savefile=False)
        print(" Time elapsed for 10 epochs", str(timedelta(seconds=time.time() - start)))

    def train_one_epoch(self,epoch_index,writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        lastoutputs=-1

        for i, data in enumerate(self.trainingsethandler.training_loader):
            self.totalsteps += 1
            # Every data instance is an input + label pair
            _, labels, inputs = data
            # print(data)
            # Zero your gradients for every batch!

            self.optimizer.zero_grad()

            # Generate a new line for image debugging purpose
            self.tensorvisualizer.newline()
            # Make predictions for this batch

            outputs = self.model(inputs)

            #self.tensorvisualizer.showcurrentmatrix()

            # Compute the loss and its gradients
            # print('inputs',inputs)
            # print('outputs',outputs)
            labels = (abs(labels - 10.0))
            labels = labels.to(torch.float64)
            print("labels.dtype",labels.dtype)
            # labels = labels.double()
            #print(labels)
            #print(torch.squeeze(outputs, 1))

            # mae_loss = nn.L1Loss()
            # output = mae_loss(input, target)
            # output.backward()

            loss_fn = torch.nn.L1Loss()
            #loss_fn = torch.nn.MSELoss()
            #loss = loss_fn(torch.squeeze(outputs, 1), labels)  # sum(labels)/len(labels)
            print("inputs",inputs)
            print("labels", labels)
            print("outputs",outputs)
            print("inputs.shape",inputs.shape)
            print("labels.shape",labels.shape)
            print("outputs", outputs.shape)
            loss = loss_fn(torch.squeeze(outputs, 1), labels)
            #print("torch.squeeze(outputs, 1)",torch.squeeze(outputs, 1))


            print("loss", loss.item())
            print("outputs",outputs)
            loss.backward()
            # for name,param in self.model.named_parameters():
            #     print("name")
            #     print(name)
            #     print("data")
            #     print(param.data.shape)
            #     self.tensorvisualizer.addcol(param.data,1)
            #     self.tensorvisualizer.addcol(param.grad,3)
            #
            #     print("max")
            #     print(torch.max(param.data).item())
            #     print("min")
            #     print(torch.min(param.data).item())
            #     print("grad")
            #     print(param.grad.shape)
            #     print("max")
            #     print(torch.max(param.grad).item())
            #     print("min")
            #     print(torch.min(param.grad).item())
            # print(self.model.linear1.weight.data)
            # print(self.model.linear1.weight.data.shape)
            # print(self.model.linear1.weight.grad)
            # print(self.model.linear1.weight.grad.shape)

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if self.totalsteps % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                #tb_x = epoch_index * len(self.trainingsethandler.training_loader) + i + 1
                tb_x = self.totalsteps
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return inputs

        print('Total loss for this batch: {}'.format(loss.item()))

    def trainmultipleepochs(self,epochcount, startfromscratch,savefile=True):
        if not startfromscratch:
            self.model = torch.load(globalvariables.getbasedirectory() + "model2.pt")
        writer = SummaryWriter(globalvariables.getbasedirectory() + '/runs/trainer')
        for i in range(0, epochcount):
            self.totalepoch += 1
            print("Epoch",i)
            correctedinputs = self.train_one_epoch(i,writer)
        if savefile:
            torch.save(self.model, globalvariables.getbasedirectory() + "model2.pt")
        #reactiver pour debugself.tensorvisualizer.showcurrentmatrix()
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)
        #UserInteraction.createusersbatchs(batchnumber=10, batchindex=creationbatch)
        # Reload training set after user interaction
        #self.trainingsethandler.loadtrainingset()

