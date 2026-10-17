import os

from sources.globalvariables import getbasedirectory
from glob import glob
import shutil


def fusetrainingfolders(sourcefolder,destinationfolder):
    folderlist = os.listdir(sourcefolder)
    fileindex = 0
    for folder in folderlist:
        os.chdir(sourcefolder+folder)
        numberoffiles = len(glob("*.txt"))
        print(folder)
        for i in range(numberoffiles):
            sourcetxtfile = str(i)+".txt"
            destinationtxtfile = str(fileindex)+".txt"
            shutil.copy(sourcefolder+folder+"\\"+sourcetxtfile, destinationfolder+ "\\"+ destinationtxtfile)
            sourcepngfile = str(i) + ".png"
            destinationpngfile = str(fileindex) + ".png"
            shutil.copy(sourcefolder+folder+"\\"+sourcepngfile, destinationfolder+ "\\"+ destinationpngfile)
            fileindex+=1


fusetrainingfolders(sourcefolder=getbasedirectory(subdirectory = "plunge1\\train",nosyncdir=True),destinationfolder=getbasedirectory(subdirectory = "plunge1\\trainmerged",nosyncdir=True))