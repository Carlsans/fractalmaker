import numpy as np
import torch
import torch.nn.functional as F
import torchvision
#from PIL.Image import Image
from PIL import Image
from matplotlib import pyplot as plt, cm


class TensorVisualization:
    currentcolumn:torch.Tensor
    currentrow:torch.Tensor

    def __init__(self):
        self.currentcolumn=torch.empty(3,0,0)
        self.currentrow=torch.empty(3,0,0)

    def addtensor2d(self,tensor,axis,origintensor=torch.empty(3, 0, 0)):
        if len(tensor.shape) != 3:
            raise Exception("Tensor shape not adequate",tensor.shape)
        if axis < 1 or axis > 2:
            raise Exception("axis must be 1 or 2")
        if origintensor.shape[2]==0 and origintensor.shape[1]==0 or origintensor.shape == (3,0,0) :
            #print( tensor.shape)
            return tensor
        shapex = origintensor.shape[1] - tensor.shape[1]
        shapey = origintensor.shape[2] - tensor.shape[2]
        #print("x= "+str(shapex)+"y= "+str(shapey))
        if axis == 1:
            if origintensor.shape[2] < tensor.shape[2]:
                origintensor= F.pad(origintensor, (0,abs(shapey),0,0), "constant", 0)
            else:
                tensor = F.pad(tensor, (0, abs(shapey),0,0), "constant", 0)
        else:
            if origintensor.shape[1] < tensor.shape[1]:
                origintensor = F.pad(origintensor, (0,0,0, abs(shapex)), "constant", 0)
            else:
                tensor = F.pad(tensor, (0,0,0,abs(shapex)), "constant", 0)
        #print(origintensor.shape)
        #print(tensor.shape)
        return torch.concat((origintensor, tensor), axis=axis)



    def addcol(self,col,colorchannel = 0):
        #print("addcol",col)
        # Normalize for proper display
        # min = torch.min(col[0])
        # max = torch.max(col[0])
        colcopy = torch.clone(col)
        colcopy -= colcopy.min(1, keepdim=True)[0]
        colcopy /= colcopy.max(1, keepdim=True)[0]
        #print("min", min, "max", max)
        colcopy = self.formattensorshape(colcopy,colorchannel)
        self.currentcolumn = self.addtensor2d(colcopy, 2,self.currentcolumn)
    def addcol2dnumpy(self,numpymatrix,maxvalue = 100):
        column = self.formattensorshape(torch.from_numpy(numpymatrix))
        self.currentcolumn = self.addtensor2d(column, 2, self.currentcolumn)

    def newline(self):
        self.currentrow = self.addtensor2d(self.currentcolumn, 1,self.currentrow)
        self.currentcolumn = torch.torch.empty(3,0,0)
    def showcurrentmatrix(self):
        #self.newline()
        self.showmatrix(self.currentrow)
        self.reset()
    def reset(self):
        self.currentcolumn = torch.empty(3, 0, 0)
        self.currentrow = torch.empty(3, 0, 0)
    def showmatrix(self,amatrix):
        # The matrix set by default has 3:x:y shape, bring it to 1:x:y
        amatrix = torch.unsqueeze(amatrix[0,:,:],0)
        # Cast to a numpy array
        amatrix = amatrix.numpy()
        # Color map accept only x,y,1 shape so roll axis
        amatrix = np.rollaxis(amatrix, 0, 3)
        print("amatrix",amatrix)
        print(amatrix.shape)
        # Normalize matrix
        npimg = amatrix / 100
        # Apply color map to unnormalised matrix
        npimg = np.uint8(cm.gist_stern(npimg) * 255)
        # display image
        im = Image.fromarray(npimg, mode="RGBA")
        im.show()
        #npimg = torchvision.utils.make_grid(amatrix).detach().numpy()
        #print("transposition",np.transpose(npimg, (1, 2, 0)).shape)
        #plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        #plt.imshow(np.transpose(npimg, (1, 2, 0)),cmap="Purples")
        #plt.show()
    def togray(self,img):
        img = img.mean(dim=0)
        return img

    def formattensorshape(self,atensor,colorchannel=0):
        if len(atensor.shape) == 2:
            returntensor = torch.unsqueeze(torch.detach(atensor),0).expand(3,-1,-1)
        if len(atensor.shape) == 1:
            returntensor = torch.unsqueeze(torch.unsqueeze(torch.detach(atensor),0),0).expand(3,-1,-1)
        if len(atensor.shape) == 3:
            returntensor = atensor
        if colorchannel > 0:
            modifiedreturntensor = torch.zeros(0,returntensor.shape[1],returntensor.shape[2])
            selectedcolor = torch.unsqueeze(returntensor[0][:][:],0)
            emptytensor = torch.zeros(1,returntensor.shape[1],returntensor.shape[2])

            for i in range(1,4):
                if i == colorchannel:
                    modifiedreturntensor = torch.concat((modifiedreturntensor, selectedcolor), axis=0)
                else:
                    modifiedreturntensor = torch.concat((modifiedreturntensor, emptytensor), axis=0)
            #print(modifiedreturntensor)
            return modifiedreturntensor
        else:
            return returntensor

        return atensor