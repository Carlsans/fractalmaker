import copy
import math
import multiprocessing
import threading
import tkinter as tk
from threading import Thread

import PIL
from PIL import Image, ImageTk
from matplotlib import cm

from sources import misc
from sources.ManualFractalExploration.InterfaceInformationsManager import InterfaceInformationsManager
from sources.ManualFractalExploration.UserEventManager import  UserEventManager
from sources.fractals import FractalParameters, Fractal
from sources.globalvariables import getbasedirectory
from collections import deque

class ExplorationWindow:

    def __init__(self,fractalid):
        print("__name__", __name__)
        self.usereventmanager = UserEventManager(self)
        self.fractal = None
        self.WIDTH, self.HEIGHT = 512, 512
        self.userinterfacewindowwidth = 300
        self.fractalcenterx = 0
        self.fractalcentery = 0
        self.fractalradiusx = 4
        self.fractalradiusy = 4
        self.isselecting = False
        self.isfractaldouble = False
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.oldtopx, self.oldtopy, self.oldbotx, self.oldboty = 0, 0, 0, 0
        self.fractalid = fractalid
        #self.fractalpalette = FractalParameters.getrandomcmap()
        self.fractalpalette = cm.get_cmap('gist_stern')
        self.rect_id = None
        self.currentfractalimage = None
        self.trainingimagesdirectory = getbasedirectory(subdirectory="viewer\\trainingimages",nosyncdir=True)
        self.history = deque()
        self.interfaceinformations = None
        self.initwindow()

    def get_mouse_posn(self, event):
        self.topx, self.topy = event.x, event.y
        self.botx,self.boty = 0, 0
        self.isselecting = True
        #print(topx, topy)
    def update_sel_rect(self, event):
        self.botx, self.boty = event.x, event.x + self.topy - self.topx
        #print("event.x",event.x)
        self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)  # Update selection rect.
        #print(self.topx, self.topy, self.botx, self.boty)
        self.calculatefractalbounds(printonly=True)

    def mousebuttonreleased(self, event):
        if self.isselecting:
            # Faux mouvement
            if event.x - self.topx < 5:
                return
            #self.botx, self.boty = event.x, event.y
            self.oldtopx, self.oldtopy, self.oldbotx, self.oldboty = self.topx, self.topy, self.botx, self.boty
            self.botx, self.boty = event.x, event.x + self.topy - self.topx

            print("Relaché ", self.botx, self.boty)
            #print(self.topx/self.WIDTH, self.topy/self.HEIGHT, self.botx/self.WIDTH, self.boty/self.HEIGHT)
            self.isselecting = False
            self.calculatefractalbounds()

    def loadfractalfromhistory(self):

        if len(self.history) == 0:
            print("Premier fractal de l'historique !")
            return
        print("Retour vers image précédente")
        self.loadfractal(loadfromhistory=True)
        self.reloadfractalwindow()
    def changecolormapandreloadfractal(self,colormapname):
        self.fractalpalette = cm.get_cmap(colormapname)
        #self.loadfractal
        self.fractal.parameters.setpalette(self.fractalpalette)
        self.generatefractal()
        self.reloadfractalwindow()
    def generatefractal(self):

        self.currentfractalimage = self.fractal.generatefractalmultithread(8, returnpillowimage=True,
                                                                           doubleprecision=self.isfractaldouble)
    def reloadfractal(self):
        self.generatefractal()

        self.interfaceinformations.populatecanvas()
        self.reloadfractalwindow()
    def appendfractaltohistory(self):
        self.history.append(copy.deepcopy(self.fractal))
        print("Append to history :",self.history)
    def key_press(self,event):
        key = event.char
        self.usereventmanager.getKeyEvent(key)


    def savehiresolutionfractal(self,dimensions):
        print("Fractal haute densité en sauvegarde",dimensions)
        hiresfractal = copy.deepcopy(self.fractal)
        p = multiprocessing.Process(target=hiresfractal.generatefractalc,  kwargs={'iteration':self.fractalid,'subdirectory': "viewer\\hd", 'suffix': "_"+misc.get_timestamp(), 'display':False, 'overridedimensions':dimensions,'doubleprecision':self.isfractaldouble,'blackandwhitetoo':True})
        p.start()



    def calculatefractalbounds(self,printonly=False):
        radiusx = (self.botx - self.topx) / 2
        radiusy = (self.boty - self.topy) / 2
        centerx = self.topx + radiusx
        centery = self.topy + radiusy
        newradiusx = (radiusx / self.WIDTH) * self.fractalradiusx #* 2
        newradiusy = (radiusy / self.HEIGHT) * self.fractalradiusy #* 2
        newcenterx = ((-self.WIDTH / 2 + centerx)/self.WIDTH) * 2 * self.fractalradiusx + self.fractalcenterx
        newcentery = ((-self.HEIGHT / 2 + centery) / self.HEIGHT)* 2 * self.fractalradiusy + self.fractalcentery
        print("centerx",centerx,"centery",centery,"radiusx",radiusx,"radiusy",radiusy)
        print("newradiusx",newradiusx,"newradiusy",newradiusy,"newcenterx",newcenterx,"newcentery",newcentery)
        if printonly:
            return
        self.fractalradiusx, self.fractalradiusy = newradiusx,newradiusy
        self.fractalcenterx, self.fractalcentery = newcenterx,newcentery
        self.loadfractal()
        self.reloadfractalwindow()

    def reloadfractalwindow(self):
        img = ImageTk.PhotoImage(self.currentfractalimage)
        self.canvas.img = img  # Keep reference in case this code is put into a function.
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        # Create selection rectangle (invisible since corner points are equal).
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.rect_id = None
        self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                                    dash=(2, 2), fill='', outline='white')

    def initwindow(self):
        self.window = tk.Tk()
        self.window.title("Select Area")
        self.window.geometry('%sx%s' % (self.WIDTH+self.userinterfacewindowwidth, self.HEIGHT))
        self.window.configure(background='grey')
        self.window.protocol("WM_DELETE_WINDOW", exit)
        self.loadfractal()

        #img = ImageTk.PhotoImage(Image.open(self.path))
        img = ImageTk.PhotoImage(self.currentfractalimage)
        #img = ImageTk.PhotoImage(fractalimage.resize((self.WIDTH, self.HEIGHT), Resampling.LANCZOS))
        self.canvas = tk.Canvas(self.window, width=img.width(), height=img.height(),
                           borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="left")
        self.canvas.img = img  # Keep reference in case this code is put into a function.
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)

        # Create selection rectangle (invisible since corner points are equal).
        self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                          dash=(2, 2), fill='', outline='white')
        self.canvas.focus_force()
        self.canvas.bind('<Button-1>', self.get_mouse_posn)
        self.canvas.bind('<B1-Motion>', self.update_sel_rect)
        self.canvas.bind('<ButtonRelease-1>', self.mousebuttonreleased)
        self.canvas.bind('<KeyPress>', self.key_press)
        self.canvas.bind('<F2>', self.savehiresolutionfractal)
        self.userinterface = tk.Frame(self.window, width=self.userinterfacewindowwidth, height=self.HEIGHT,
                           borderwidth=0, highlightthickness=0,background="red")

        self.usereventmanager.populateuserinterface()
        self.userinterface.pack(side="top")
        self.interfaceinformations = InterfaceInformationsManager(self,tk.Frame(self.window, width=self.userinterfacewindowwidth, height=self.HEIGHT,
                           borderwidth=0, highlightthickness=0,background="blue"))

        self.interfaceinformations.getCanvas().pack(side="bottom")
        self.window.mainloop()

    def loadfractal(self,loadfromhistory = False):
        if loadfromhistory:
            if len(self.history)>1:
                self.history.pop()
                self.fractal = self.history[-1]
                self.interfaceinformations.populatecanvas()
                print(self.fractal.parameters)
                self.setfractalradiusandrange()
                self.generatefractal()
            else:
                print("Nothing left in history")
        else:
            self.savetravelimagesfortraining()
            if self.fractal == None:
                fractalparameters = FractalParameters.defaultparameters()
                fractalparameters.max_iterations = 100
                fractalparameters.fractaldimensions = (self.WIDTH, self.HEIGHT)
                fractalparameters.setweightsfromfractalid(self.fractalid)
            else:
                fractalparameters = self.fractal.parameters
            fractalparameters.setrangefromcentertoside((self.fractalcentery,self.fractalcenterx), (self.fractalradiusy, self.fractalradiusx))
            fractalparameters.setpalette(self.fractalpalette)
            self.fractal = Fractal(fractalparameters)
            if self.interfaceinformations:
                self.interfaceinformations.populatecanvas()
            self.appendfractaltohistory()
            print("history",self.history)
            #subdirectory = "plunge1\\train\\" + str(self.fractalid)
            self.generatefractal()
            #self.currentfractalimage = self.fractal.generatefractalc(display=False,returnpillowimage=True,doubleprecision=self.isfractaldouble)

    def setfractalradiusandrange(self):
        self.fractalcentery, self.fractalcenterx = self.fractal.parameters.getfractalcenter()
        fractalranges = self.fractal.parameters.getfractalrangelength()
        self.fractalradiusy, self.fractalradiusx = fractalranges[0] / 2, fractalranges[1] / 2


    def savetravelimagesfortraining(self):
        if self.currentfractalimage == None:
            return

        lastfilesavedid = misc.getlastfilenumber(self.trainingimagesdirectory,"txt")
        currentfileid = lastfilesavedid+1
        file = self.trainingimagesdirectory + str(currentfileid) + ".txt"
        self.currentfractalimage.save(self.trainingimagesdirectory + str(currentfileid) + ".png")

        #if os.path.exists(file):
        #    os.remove(file)

        centerx = math.ceil((self.oldbotx - self.oldtopx) / 2) + self.oldtopx
        centery = math.ceil((self.oldboty - self.oldtopy) / 2) + self.oldtopy
        radius = math.ceil((self.oldbotx - self.oldtopx) / 2)
        print("centerx",centerx,"centery",centery,"radius",radius)
        with open(file, 'w') as f:
            f.write(str(centerx) + "/" + str(self.WIDTH) + "\n")
            f.write(str(centery) + "/" + str(self.HEIGHT) + "\n")
            f.write(str(radius) + "/" + str(self.WIDTH) + "\n")
        print(self.currentfractalimage)


