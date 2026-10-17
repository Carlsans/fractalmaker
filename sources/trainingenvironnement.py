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
import secrets
from sources.globalvariables import getbasedirectory, gettraininghyperparameter

def generateRGBrandomcolor():
    return (secrets.randbelow(256),secrets.randbelow(256),secrets.randbelow(256))
def generaterandomcenter(resolution):
    return (secrets.randbelow(resolution),secrets.randbelow(resolution))
def savetrainresultfile(subdirectory,filenumber,results,nosyncdir=True):
    file = getbasedirectory(subdirectory,nosyncdir=nosyncdir) + str(filenumber) + ".txt"
    x, y, r = results
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'w') as f:
        f.write(str(x) + "\n")
        f.write(str(y) + "\n")
        f.write(str(r) + "\n")

def createtrainingfilesystem(trainingelements,subdirectory,internellipses=True,simpleellipse = True,speckles = True,rectangles = True,regularpolygon = True,deleteoldfiles = True):
    secretsGenerator = secrets.SystemRandom()

    if deleteoldfiles:
        [f.unlink() for f in Path(getbasedirectory(subdirectory,nosyncdir=True)).glob("*") if f.is_file()]
        print("files deleted")

    for i in range(trainingelements):
        imageresolution = 128
        maxradius = imageresolution / 2
        innerellipsesmaxnumber = 20
        x, y = generaterandomcenter(imageresolution)
        r = secrets.randbelow(int(maxradius))+2
        results = (x,y,r)
        image = PIL.Image.new('RGB', (imageresolution, imageresolution),color=generateRGBrandomcolor())
        draw = PIL.ImageDraw.Draw(image)
        variationnumber = secrets.randbelow(sum([simpleellipse,internellipses,rectangles,regularpolygon]))
        print(variationnumber)
        p = 0
        if regularpolygon:
            if p == variationnumber :
                p = - 1

                draw.regular_polygon((x,y,r),secrets.randbelow(7)+3, fill=generateRGBrandomcolor())

            else:
                p += 1
        if simpleellipse:
            if p == variationnumber :
                p = - 1
                leftUpPoint = (x - r, y - r)
                rightDownPoint = (x + r, y + r)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.ellipse(twoPointList, fill=generateRGBrandomcolor())
            else:
                p += 1

        if internellipses:
            if p == variationnumber:
                p = - 1
                ellipsenumber = secrets.randbelow(innerellipsesmaxnumber)
                for j in range(ellipsenumber,0,-1):
                    jnormalisedx = (j / ellipsenumber) * secretsGenerator.uniform(0,1)
                    jnormalisedy = (j / ellipsenumber) * secretsGenerator.uniform(0,1)
                    leftUpPoint = (x - r*jnormalisedx, y - r * jnormalisedy)
                    rightDownPoint = (x + r*jnormalisedx, y + r * jnormalisedy)
                    twoPointList = [leftUpPoint, rightDownPoint]
                    draw.ellipse(twoPointList, fill=generateRGBrandomcolor())
            else:
                p += 1

        if rectangles:
            if p == variationnumber:
                p = - 1
                rvariation = secretsGenerator.uniform(0,1.25)
                leftUpPoint = (x - r*rvariation, y - r*rvariation)
                rightDownPoint = (x + r*rvariation, y + r*rvariation)
                twoPointList = [leftUpPoint, rightDownPoint]

                draw.rectangle(twoPointList, outline=generateRGBrandomcolor(),
                               width=int(secretsGenerator.uniform(0,0.1)*imageresolution))
            else:
                p += 1
        if speckles:
            pixelsnumber = imageresolution * imageresolution
            specklesnumber = secrets.randbelow(int(pixelsnumber))
            for j in range(specklesnumber):
                image.putpixel((secrets.randbelow(imageresolution), secrets.randbelow(imageresolution)), (secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256)))


        image.save(getbasedirectory(subdirectory,nosyncdir=True) + str(i) + ".png", "PNG")
        savetrainresultfile(subdirectory, i, results)
subdirectory = "testvisual"
testdirectory = "testvisualverification"
#createtrainingfilesystem(20000,subdirectory)
#createtrainingfilesystem(100,testdirectory)