import tkinter as tk

class InterfaceInformationsManager:
    def __init__(self,explorationwindow, canvas):
        self.explorationwindow = explorationwindow
        self.canvas = canvas
        self.populatecanvas()
    def populatecanvas(self):
        labelrangex = tk.Label(self.canvas, text="Range X=[" + str(self.explorationwindow.fractal.parameters.xrange[0])+":"+ str(self.explorationwindow.fractal.parameters.xrange[1])+"]")
        labelrangey = tk.Label(self.canvas, text="Y=["+ str(self.explorationwindow.fractal.parameters.yrange[0])+":"+ str(self.explorationwindow.fractal.parameters.yrange[1])   +"]")

        if self.explorationwindow.isfractaldouble :
            labelvariablesize = tk.Label(self.canvas, text="Variable is double")
        else:
            labelvariablesize = tk.Label(self.canvas, text="Variable is float")
        labelnumiterations = tk.Label(self.canvas, text=",Iterations="+str(self.explorationwindow.fractal.parameters.max_iterations))
        labelhistorylen = tk.Label(self.canvas, text="History length="+str(len(self.explorationwindow.history)))
        labelrangex.grid(row=0, column=0,columnspan=2)
        labelrangey.grid(row=1, column=0, columnspan=2)
        labelvariablesize.grid(row=2, column=0)
        labelnumiterations.grid(row=2, column=1)
        labelhistorylen.grid(row=3, column=0)
    def getCanvas(self):
        return self.canvas