import copy
import math
import random
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import ImageTk
from PIL import Image

from sources.fractals import FractalParameters


class UserEventManager:
    def __init__(self,explorationwindow):
        self.explorationwindow = explorationwindow
    def getKeyEvent(self,key):
        if key=='h' or key=='H':
            self.explorationwindow.loadfractalfromhistory()
        if key=='p'or key=='P':
            self.displayColorMapHelper()
        if key == 'f':
            self.explorationwindow.isfractaldouble = not self.explorationwindow.isfractaldouble
            if self.explorationwindow.isfractaldouble:
                print("Using double variables for fractal generation")
            else:
                print("Using 16 bit float variables for fractal generation")
            print("Waiting")
            self.explorationwindow.reloadfractal()
            print("Done")
        if key == 'F':
            if self.explorationwindow.isfractaldouble:
                print("Currently using double variables for fractal generation")
            else:
                print("Currently using 16 bit float variables for fractal generation")
        if key == 'r':
            self.zoomfractal(2)
        if key == 'R':
            self.zoomfractal(8)
        if key == 't':
            self.zoomfractal(0.5)
        if key == 'T':
            self.zoomfractal(0.125)
        if key == 'i':
            self.setfractaliteration(100)
        if key == 'I':
            self.setfractaliteration(100,increment=True)
        if key == 'o':
            self.setfractaliteration(350)
        if key == 'O':
            self.setfractaliteration(20)
        if key=='Q':
            exit()
        print(key, 'is pressed')
    def populateuserinterface(self):
        ui = self.explorationwindow.userinterface
        historybutton = tk.Button(self.explorationwindow.userinterface,text="Historique",command=self.explorationwindow.loadfractalfromhistory)
        palettebutton = tk.Button(ui, text="Palette",command=self.displayColorMapHelper)
        labelzoom = tk.Label(self.explorationwindow.userinterface, text="Zoom :")
        zoomx2 = tk.Button(self.explorationwindow.userinterface,text="X2",command=lambda: self.zoomfractal(1/2))
        zoomx8 = tk.Button(self.explorationwindow.userinterface, text="X8",
                                command=lambda: self.zoomfractal(1 / 8))
        unzoomx2 = tk.Button(self.explorationwindow.userinterface, text="-X2",
                                command=lambda: self.zoomfractal(2))
        unzoomx8 = tk.Button(self.explorationwindow.userinterface, text="-X8",
                                command=lambda: self.zoomfractal(8))
        labelincrement = tk.Label(self.explorationwindow.userinterface, text="Increment :")
        incrementiteration = tk.Button(self.explorationwindow.userinterface, text="+ 100",
                                command=lambda: self.setfractaliteration(100,increment=True))
        incrementiteration2 = tk.Button(self.explorationwindow.userinterface, text="= 40",
                                       command=lambda: self.setfractaliteration(40, increment=False))
        incrementiteration3 = tk.Button(self.explorationwindow.userinterface, text="= 100",
                                        command=lambda: self.setfractaliteration(100, increment=False))
        incrementiteration4 = tk.Button(self.explorationwindow.userinterface, text="= 256",
                                        command=lambda: self.setfractaliteration(256, increment=False))
        floatvariablesizebutton = tk.Button(self.explorationwindow.userinterface, text="Float size",
                                        command=self.flipfractalvariablesize)
        labelasyncres = tk.Label(self.explorationwindow.userinterface, text="Async cap=")

        resolution4000x4000button = tk.Button(self.explorationwindow.userinterface, text="4000x4000",
                                            command=lambda: self.explorationwindow.savehiresolutionfractal((4000,4000)))
        resolution2000x2000button = tk.Button(self.explorationwindow.userinterface, text="2000x2000",
                                              command=lambda: self.explorationwindow.savehiresolutionfractal(
                                                  (2000, 2000)))
        resolution8000x8000button = tk.Button(self.explorationwindow.userinterface, text="8000x8000",
                                              command=lambda: self.explorationwindow.savehiresolutionfractal(
                                                  (8000, 8000)))
        historybutton.grid(row=0, column=0)
        palettebutton.grid(row=0, column=1)
        labelzoom.grid(row=1, column=0)
        zoomx2.grid(row=1, column=1)
        zoomx8.grid(row=1, column=2)
        unzoomx2.grid(row=1, column=3)
        unzoomx8.grid(row=1, column=4)
        labelincrement.grid(row=2, column=0)
        incrementiteration.grid(row=2, column=1)
        incrementiteration2.grid(row=2, column=2)
        incrementiteration3.grid(row=2, column=3)
        incrementiteration4.grid(row=2, column=4)
        floatvariablesizebutton.grid(row=3, column=0)
        labelasyncres.grid(row=4, column=0)
        resolution2000x2000button.grid(row=4, column=1)
        resolution4000x4000button.grid(row=4, column=2)
        resolution8000x8000button.grid(row=4, column=3)

    def flipfractalvariablesize(self):
        self.explorationwindow.isfractaldouble = not self.explorationwindow.isfractaldouble

        print("Waiting")
        self.explorationwindow.reloadfractal()
        print("Done")
    def setfractaliteration(self,iterations,increment=False):
        if increment:
            self.explorationwindow.fractal.parameters.max_iterations += iterations
        else:
            self.explorationwindow.fractal.parameters.max_iterations = iterations
        print("Max iterations =",self.explorationwindow.fractal.parameters.max_iterations)
        self.explorationwindow.reloadfractal()
    def zoomfractal(self,zoomfactor):
        xyranges = self.explorationwindow.fractal.parameters.getfractalscaleddimensions(zoomfactor)
        print(xyranges)
        self.explorationwindow.fractal.parameters.xrange = (xyranges[0], xyranges[1])
        self.explorationwindow.fractal.parameters.yrange = (xyranges[2], xyranges[3])
        print("New range : X=", self.explorationwindow.fractal.parameters.xrange, "Y=",
              self.explorationwindow.fractal.parameters.yrange)
        self.explorationwindow.setfractalradiusandrange()
        self.explorationwindow.reloadfractal()
        self.explorationwindow.appendfractaltohistory()
    def displayColorMapHelper(self):
        self.root = tk.Tk()
        self.root.title("Colormap Palette (actual palette :"+ self.explorationwindow.fractalpalette.name +")")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        #root.geometry('5x5')
        self.frame = tk.Frame(self.root)
        #self.frame = tk.Frame()
        self.frame.pack(expand=True)
        self.frame.focus_force()
        self.imagelist = []

        # Display currently in use palette
        currentpaletteimage = self.getTKinterimagefromcolormapname(self.explorationwindow.fractalpalette.name,
                                                                   colormapimagewidth=256*4, nuancesnumber=256,
                                                                   bandwidth=20)
        currentpalettelabel = tk.Label(self.frame, image=currentpaletteimage, padx=1, pady=1,
                         text="In use : " + self.explorationwindow.fractalpalette.name, compound='center')
        currentpalettelabel.grid(row=0, column=0,columnspan=4)

        colormaplist = FractalParameters.getcompletecmaparray()
        col,row = 0, 0
        maxrow = 20
        for j, colormapname in enumerate(colormaplist):
            self.imagelist.append(self.getTKinterimagefromcolormapname(colormapname))
            label = tk.Label(self.frame, image=self.imagelist[-1],padx=1,pady=1,text=colormapname,compound='center')

            if row > maxrow:
                col +=1
                row = 0
            label.grid(row=row+1, column=col)

            label.bind('<Button-1>', lambda e,n=colormapname: self.selectcolormap(e,n))


            #self.root.update()
            row += 1

        self.root.mainloop()
    def getTKinterimagefromcolormapname(self,colormapname,colormapimagewidth = 256,nuancesnumber = 32,bandwidth = 15):
        colormap = cm.get_cmap(colormapname)
        color_array = np.zeros((1, nuancesnumber, 4), dtype=np.uint8)
        for anuance in range(nuancesnumber):
            nuancecolor = colormap(anuance / nuancesnumber)
            #print(nuancecolor)
            color_array[0, anuance] = np.array(nuancecolor) * 255

        im = Image.fromarray(color_array)
        im = im.resize(size=(colormapimagewidth, bandwidth))
        return ImageTk.PhotoImage(im, master=self.frame)
    def selectcolormap(self,e,n):
        print(n)
        self.explorationwindow.changecolormapandreloadfractal(n)
        self.root.destroy()


    def rgbtohex(self,r, g, b):
        return f'#{r:02x}{g:02x}{b:02x}'