class EpochDot:
    def __init__(self):
        self.currentcolumn = 0
        self.maxcolumn = 120
    def displaydot(self):
        if self.currentcolumn > self.maxcolumn:
            print(".")
            self.currentcolumn = 0
            return
        print(".",end="")
        self.currentcolumn +=1
    def reset(self):
        self.currentcolumn = 0
