import glob
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import PIL
import numpy as np
import torch
import winsound
from PIL import Image
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt, cm
from torch import Tensor
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import transforms

from sources import debugging, fractals
from sources.globalvariables import getbasedirectory, gettraininghyperparameter
from sources.models import VisualLayerModel
from torch.profiler import profile, ProfilerActivity

from sources.trainingenvironnement import createtrainingfilesystem



if __name__ == "__main__":

    print()
    #createtrainingfilesystem(20000,subdirectory)
    #createtrainingfilesystem(100,testdirectory)
class CustomVisualLayerDataset(Dataset):
    def __init__(self, subdirectory, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = getbasedirectory(subdirectory,nosyncdir=True)
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
        with open(self.img_dir + str(idx) + '.txt', "r") as myfile:
            data = myfile.read().splitlines()
            for i in range(2):
                if not data[i].isdigit():
                    raise Exception("Text file error, file " + self.img_dir + str(idx) + '.txt' + "is non digit")
            return (int(data[0]),int(data[1]),-1)

    # def getparamfilevalue(self, idx):
    #     return torch.load(getbasedirectory() + str(idx) + '.pt')

    def __getitem__(self, idx):
        #self.gettxtfilevalue(idx)
        img_path = self.img_dir + str(idx) + ".png"
        #print(img_path)
        image = read_image(img_path,mode=ImageReadMode.RGB)
        #print("image.shape",image.shape)
        #print("normalized image",image)
        result = self.gettxtfilevalue(idx)
        result = torch.Tensor(result)
        #param = self.getparamfilevalue(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            result = self.target_transform(result)

        return image, result

@dataclass
class VisualLayerTrainingSetHandler():
    training_set:CustomVisualLayerDataset
    training_loader:DataLoader

    def __init__(self,subdirectory,shuffle = True,singleprocess = False):
        #pickle.dumps("")
        num_worker = 6
        if singleprocess:
            num_worker = 0
        if __name__ == "__main__":
            self.training_set = CustomVisualLayerDataset(subdirectory=subdirectory)
            self.training_loader = DataLoader(self.training_set, batch_size=gettraininghyperparameter("dataloaderbatchsize"), shuffle=shuffle,num_workers=num_worker,persistent_workers=False,pin_memory=True)


    def showtrainingset(self, minrange, maxrange):
        figure = plt.figure(figsize=(8, 8))
        rangetotal = maxrange - minrange
        rows = int(rangetotal / 5) + 1
        cols = 5
        for i in range(0, maxrange + 1):
            # sample_idx = torch.randint(i, size=(1,)).item()
            img,  imagecoordinates = self.training_set[i + minrange]
            figure.add_subplot(rows, cols, i + 1)
            plt.title(imagecoordinates)
            plt.axis("off")
            plt.imshow(img.squeeze().transpose(0, 2))
        plt.show()

    def loadtrainingset(self,overriderange=(0,0)):
        if overriderange==(0,0):
            print("Loading training set image disabled")
            #self.showtrainingset(minrange=0, maxrange=len(self.training_set)-1)
        else:
            minrange,maxrange = overriderange
            self.showtrainingset(minrange=minrange, maxrange=maxrange)


        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5,), (0.5,))])
        # Create data loaders for our datasets; shuffle for training, not for validation

        print('Training set has {} instances'.format(len(self.training_set)))
        #print("training_set", self.training_set)
        #print("dataloader", self.training_loader)
        #dataiter = iter(self.training_loader)
        #images, values= dataiter.next()
        # print(images)
        #print(labels)
        #print(params)

def testtrainingset(model=None):
    with torch.no_grad():
        trainingsethandler = VisualLayerTrainingSetHandler(testdirectory,shuffle = False,singleprocess = True)
        #trainingsethandler.loadtrainingset(overriderange=(0,4))
        if model is None:
            if torch.cuda.is_available():
                model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt")

            else:
                model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt",map_location=torch.device('cpu'))

            #model.__init__()
        for i, data in enumerate(trainingsethandler.training_loader):
            # Every data instance is an input + label pair
            inputs, results = data
            if torch.cuda.is_available():
                inputs = inputs.to(torch.device("cuda:0"))
                results = results.to(torch.device("cuda:0"))
            torch.set_printoptions(threshold=10000)

            #print("inputs",inputs)
            inputs = (inputs / 256)
            results = (results[0:results.shape[0], 0] / 8).type(torch.LongTensor)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            debugstep = False
            print("inputs.shape",inputs.shape)
            outputx = model(inputs,debug = debugstep)
            outputy = model(inputs.permute(0,1,3,2),debug = debugstep)
            print(outputx.shape)
            maxx, predictedx = torch.max(outputx, 1)
            maxy, predictedy = torch.max(outputy, 1)

            for i in range(outputx.shape[0]):
                print("Real Answer =", results[i].item()," predicted Answer = (", predictedx[i].item(),",", predictedy[i].item(),") maximum = (", maxx[i].item(),",",maxy[i].item(),")")
            #print("predicted output",output,"real output",results)
            # print("mean difference", torch.mean(abs(output-results)))
            # loss_fn = torch.nn.L1Loss()
            # loss = loss_fn(torch.squeeze(output, 1), results)
            # print("verification set loss",loss)

def distributionmatrix(outputx,outputy):
    with torch.no_grad():
        xsortindex = torch.argsort(outputx, descending=True)
        ysortindex = torch.argsort(outputy, descending=True)
        print(outputx)
        print(xsortindex)
        print(outputy)
        print(ysortindex)
        #depthmatrix = torch.outer(outputx,outputy)
        #depthmatrix = normalize(depthmatrix, p=2.0, dim = 0) + 0.5
        normalizedx = normalize(outputx, p=2.0, dim = 0) + 0.5
        normalizedy = normalize(outputy, p=2.0, dim = 0) + 0.5
        summatrix = (normalizedy[:,None]+normalizedx)/2
        print(summatrix)
        im = Image.fromarray(np.uint8(cm.Greys(Tensor.cpu(summatrix)) * 255))

        #im = Image.fromarray(np.uint8(cm.Greys([[.2,.8],[.8,.1]]) * 255))
        im = im.resize((128,128),0)
        im.show()

def getfractalcentercoordinates(fractalid,display=False):
    fractalparam = fractals.FractalParameters.defaultparameters()
    fractalparam.fractaldimensions = (128,128)
    weighttensor = torch.load(getbasedirectory() + str(fractalid) + '.pt')
    fractalparam.setweighttensorfromparam(weighttensor)
    fractal = fractals.Fractal(fractalparam)
    fractaltensor = torch.unsqueeze(fractal.generatefractalc(returnnormalized=True),0)
    #torch.set_printoptions(profile="full")
    #print(fractaltensor)
    if torch.cuda.is_available():
        model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt", map_location=torch.device('cuda'))
        fractaltensor = fractaltensor.cuda()
    else:
        model = torch.load(getbasedirectory(subdirectory) + "visualmodel.pt", map_location=torch.device('cpu'))

    outputx = model(fractaltensor)
    maxx, predictedx = torch.max(outputx, 1)
    #print(outputx[0])
    #print(torch.argsort(outputx[0], descending=True))
    #print(torch.argsort(outputx[0], descending=False))
    outputy = model(fractaltensor.permute(0,1,3,2))
    maxy, predictedy = torch.max(outputy, 1)
    print("Predicted Answer = (", predictedx.item(), ",", predictedy.item(),
          ") maximum = (", maxx.item(), ",", maxy.item(), ")")
    distributionmatrix(outputx[0], outputy[0])
    if display:
        fractal.generatefractalc()
    return (predictedx.item(),predictedy.item())

def testimage(id):
    img_path = getbasedirectory(subdirectory,nosyncdir=True) + str(id) + ".png"
    image = read_image(img_path) / 256
    image = torch.unsqueeze(image,0)
    print(image.shape)
    if torch.cuda.is_available():
        image = image.cuda()
    else:
        image = image.cpu()
    model = VisualLayerModel.getmodelfromsubdirectory(subdirectory)

    outputx = model(image)
    maxx, predictedx = torch.max(outputx, 1)
    outputy = model(image.permute(0, 1, 3, 2))
    maxy, predictedy = torch.max(outputy, 1)
    print("Predicted Answer = (", predictedx.item(), ",", predictedy.item(),
          ") maximum = (", maxx.item(), ",", maxy.item(), ")")
    distributionmatrix(outputx[0], outputy[0])

if __name__ == "__main__":
    print()
    #testimage(8)
    #print(getfractalcentercoordinates(9546,display = True))
    #print(getfractalcentercoordinates(20000,display = True))#laptop
    #print(getfractalcentercoordinates(9014, display=False))


class VisualLayerTrainer:
    model:VisualLayerModel
    optimizer:torch.optim.SGD
    trainingsethandler: VisualLayerTrainingSetHandler
    tensorvisualizer: debugging.TensorVisualization
    totalsteps:int
    totalepoch:int
    def __init__(self,subdirectory,testdirectory,startfromscratch=True,debugmode = False):
        self.subdirectory = subdirectory
        self.testdirectory = testdirectory
        self.tensorvisualizer = debugging.TensorVisualization()
        self.model = VisualLayerModel()
        self.startfromscratch = startfromscratch
        self.debugmode = debugmode
        self.totalsteps = 0
        self.totalepoch = 0
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def customloss(self,outputs, values):
        result = torch.exp(abs(outputs + values))
        result = torch.sum(result) / result.shape[0]
        #print(result)
        return result
    def start(self,epochcount):
        self.trainingsethandler = VisualLayerTrainingSetHandler(self.subdirectory)
        self.trainingsethandler.loadtrainingset()
        self.trainmultipleepochs(epochcount,self.startfromscratch)
    def testtimecost(self):
        self.trainingsethandler = VisualLayerTrainingSetHandler(self.subdirectory)
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
        #loss_fn = torch.nn.MSELoss()
        #loss_fn = self.customloss
        loss_fn = torch.nn.CrossEntropyLoss()
        # if torch.cuda.is_available():
        #     loss_fn = loss_fn.cuda()
        for i, (inputs, results) in enumerate(self.trainingsethandler.training_loader):
            self.totalsteps += 1
            # Every data instance is an input + label pair

            if torch.cuda.is_available():
                inputs = inputs.to(torch.device("cuda:0"))
                results = results.to(torch.device("cuda:0"))
            inputs = (inputs / 256)
            results = (results[0:results.shape[0], 0]  / 8).type(torch.LongTensor)
            if torch.cuda.is_available():
                results = results.cuda()
            # print("resultsbefore", results.shape)
            # results[0:results.shape[0],0] /= 512
            # results[0:results.shape[0],1] /= 512
            # results[0:results.shape[0],2] /= 256
            # results[:][1] = (results[:][1] / 512)
            # results[:][2] = (results[:][2] / 256)
            print("resultsafter", results)
            # inputs = ((inputs / 256) - 0.5) * 2
            # results[0] = ((results[0] / 512) - 0.5)*2
            # results[1] = ((results[1] / 512) - 0.5)*2
            # results[2] = ((results[2] / 256) - 0.5)*2
            #results = torch.cat((torch.unsqueeze(results[0],1),torch.unsqueeze(results[1],1),torch.unsqueeze(results[2],1)),1)
            #print("results",results)
            # Zero your gradients for every batch!

            self.optimizer.zero_grad()

            # Generate a new line for image debugging purpose
            #self.tensorvisualizer.newline()
            # Make predictions for this batch
            batchinfos = (i,epoch_index)

            outputs = self.model(inputs,batchinfos,debug=self.debugmode)

            #self.tensorvisualizer.showcurrentmatrix()

            # Compute the loss and its gradients
            # print('inputs',inputs)
            # print('outputs',outputs)

            # labels = labels.double()
            #print(labels)
            #print(torch.squeeze(outputs, 1))

            # mae_loss = nn.L1Loss()
            # output = mae_loss(input, target)
            # output.backward()

            #loss_fn = torch.nn.L1Loss()

            #loss_fn = torch.nn.CrossEntropyLoss()
            #loss = loss_fn(torch.squeeze(outputs, 1), labels)  # sum(labels)/len(labels)
            # print("inputs",inputs)
            #
            # print("outputs",outputs)
            # print("inputs.shape",inputs.shape)
            #
            # print("outputs", outputs.shape)
            # print("results",results)
            # print("results.shape", results.shape)
            # #print("results.shape",results.shape)
            # print("torch.squeeze(outputs, 1)",torch.squeeze(outputs, 1))
            loss = loss_fn(torch.squeeze(outputs, 1), results )
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


        print('Total loss for this batch: {}'.format(loss.item()))

    def trainmultipleepochs(self,epochcount, startfromscratch,savefile=True):
        if not startfromscratch:
            self.model = torch.load(getbasedirectory(self.subdirectory) + "visualmodel.pt")
        writer = SummaryWriter(getbasedirectory(self.subdirectory) + '/runs/trainer')
        for i in range(0, epochcount):
            #winsound.Beep(500, 500)
            self.totalepoch += 1
            print("Epoch",i)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                self.train_one_epoch(i,writer)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            if savefile:
                torch.save(self.model, getbasedirectory(self.subdirectory) + "visualmodel.pt")
            # if i % 1 == 0 or i == epochcount-1:
            #     testtrainingset(self.model)

        #reactiver pour debugself.tensorvisualizer.showcurrentmatrix()
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)
        #UserInteraction.createusersbatchs(batchnumber=10, batchindex=creationbatch)
        # Reload training set after user interaction
        #self.trainingsethandler.loadtrainingset()
# print(__name__)
if __name__ == "__main__":
    #createtrainingfilesystem(100000, subdirectory)
    subdirectory = "plunge1\\trainmerged"
    testdirectory = "plunge1\\trainmergedtestdir"
    # visuallayertrainer = VisualLayerTrainer(subdirectory=subdirectory,testdirectory=testdirectory,startfromscratch=True)
    # visuallayertrainer.start(epochcount=500)

#     for i in range(2000):
#         createtrainingfilesystem(10000, subdirectory)
#         visuallayertrainer = VisualLayerTrainer(subdirectory=subdirectory, testdirectory=testdirectory,
#                                                 startfromscratch=False)
#         visuallayertrainer.start(epochcount=5)
#testtrainingset()
