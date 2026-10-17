from sources.globalvariables import getbasedirectory

def formatfloattostr(afloat):
    return "{:.2f}".format(afloat)

def genperfboard(maxx,maxy,separations):
    xlineseparation = maxx/separations
    ylineseparation = maxy / separations
    with open(getbasedirectory() + 'perfboard.gcode', 'w') as f:

        f.write("G1 Z12\n")
        f.write("G1 X0 Y0\n")
        # X lines
        for i in range(separations+1):
            f.write("G1 Z12\n")
            f.write("G1 Y0\n")
            f.write("G1 X"+formatfloattostr(i*xlineseparation) +"\n")
            f.write("G1 Z0\n")
            f.write("G1 Y"+ formatfloattostr(maxy) +"\n")

        for i in range(separations+1):
            f.write("G1 Z12\n")
            f.write("G1 X0\n")
            f.write("G1 Y"+formatfloattostr(i*ylineseparation) +"\n")
            f.write("G1 Z0\n")
            f.write("G1 X"+ formatfloattostr(maxx) +"\n")
        f.write("G1 Z12\n")

genperfboard(maxx = 914,maxy = 875,separations= 20)
