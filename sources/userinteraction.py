import os
import random
import copy
import torch
from matplotlib import image as mpimg, pyplot as plt
import sources.globalvariables as globalvariables
import sources.fractals as fractals


class UserInteraction:
    @staticmethod
    def createusersbatchs(batchnumber=10, batchindex=0, revisedparameterstensor=[]):
        startparameters = fractals.FractalParameters(xrange=(-2, 1), yrange=(-2, 2), max_iterations=40,
                                                        fractalbias=0.1 + 0.7j, fractalexponential=3)
        parameters=copy.copy(startparameters)
        #print(startparameters)
        for i in range(0, batchnumber):

            # if revisedparameterstensor != []:
            #     #print(revisedparameterstensor)
            #     parameters.setweighttensorfromparam(revisedparameterstensor[i])

            # create a randomnized batch
            maxrangevariation = 1.5
            maxbiasvariation = 1
            maxexpvariation = 3
            parameters.xrange = (
                startparameters.xrange[0] + random.uniform(-maxrangevariation, maxrangevariation),
                startparameters.xrange[1] + random.uniform(-maxrangevariation, maxrangevariation))
            parameters.yrange = (
                startparameters.yrange[0] + random.uniform(-maxrangevariation, maxrangevariation),
                startparameters.yrange[1] + random.uniform(-maxrangevariation, maxrangevariation))
            parameters.fractalbias = startparameters.fractalbias + random.uniform(-maxbiasvariation,
                                                                                  maxbiasvariation) + random.uniform(
                -maxbiasvariation, maxbiasvariation) * 1j
            parameters.fractalexponential = startparameters.fractalexponential + random.uniform(-maxexpvariation,
                                                                                                maxexpvariation)

            #print(revisedparameterstensor)
            #print(revisedparameterstensor[i])
            #print("parameters")
            #print(parameters)
            # if revisedparameterstensor != []:
            #     parameters.setweighttensorfromparam(revisedparameterstensor[i])

            print(parameters)
            mandelbrot_set = fractals.Fractal(parameters)
            mandelbrot_set.generatefractal(iteration=i + batchindex * batchnumber, display=False, verbose=False)
            # plt.imread(globalvariables.getbasedirectory()+"image"+str(i)+".png")
            img = mpimg.imread(globalvariables.getbasedirectory() + "image" + str(i + batchindex * batchnumber) + ".png")
            imgplot = plt.imshow(img)
            plt.show()
            note = input()
            while (True):
                if not note.isdigit():
                    if note == "h":
                        oldfractdim = mandelbrot_set.parameters.fractaldimensions
                        mandelbrot_set.parameters.fractaldimensions = (1024, 1024)
                        mandelbrot_set.generatefractal(iteration=i + batchindex * batchnumber, display=True,
                                                       verbose=True,
                                                       suffix="hres")
                        mandelbrot_set.parameters.fractaldimensions = oldfractdim
                        print("saved")
                    print("enter digit")
                    note = input()
                else:
                    if int(note)<0 or int(note) > 9:
                        print("Enter digit between 0 and 9 included")
                        note = input()
                if note.isdigit() and int(note) >= 0 and int(note) <= 9:
                    break

            FileManagement.saveclassificationtextfile(i + batchindex * batchnumber, note, parameters)
            print(note)
            parameters = copy.copy(startparameters)


class FileManagement:
    @staticmethod
    def saveclassificationtextfile(index, note, fractalparameters):
        file = globalvariables.getbasedirectory() + str(index) + ".txt"
        if os.path.exists(file):
            os.remove(file)
        with open(file, 'w') as f:
            f.write(str(note))
            # f.write(str(fractalparameters.getweighttensorfromparam()))
        torch.save(fractalparameters.getweighttensorfromparam(), globalvariables.getbasedirectory() + str(index) + '.pt')
        # and torch.load('file.pt')
# Create a fractals file system with an image of a fractal and a text file.
# The text file contain "0" by default. The user can after change it to provide output
class CreateFileSystem:
    fileindex:int
    # startfileindex represent the first file id of the suite
    def __init__(self, startfileindex):
        self.fileindex = startfileindex
    # Generate a new file according to the fractal parameters supplied.
    # File index is incremented each time to provide a suite of files from 0 to n
    def savefile(self,fractalparameters, suffix=""):

        FileManagement.saveclassificationtextfile(self.fileindex, '0', fractalparameters)
        mandelbrot_set = fractals.Fractal(fractalparameters)
        mandelbrot_set.generatefractalc(iteration=self.fileindex,display=False,suffix=suffix)
        #mandelbrot_set.generatefractal(iteration=self.fileindex,display=False,verbose=False, usemathplot=False,suffix=suffix)
        self.fileindex+=1

