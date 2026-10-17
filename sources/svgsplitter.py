import math
import os
from xml.dom import minidom
from sources.globalvariables import getbasedirectory
import glob

def fusegcode(split=False):
    gcodedirectory = getbasedirectory(subdirectory="svgtests\\gcode")
    print(gcodedirectory)
    if os.path.exists(gcodedirectory+"output.gcode"):
        os.remove(gcodedirectory+"output.gcode")

    os.chdir(gcodedirectory)
    data = ""
    filenumber = 0
    splitin = 2
    upmovement = 'G1 Z15 F30000; S0'
    for file in glob.glob("*.gcode"):
        data += "\n;File number = " + str(filenumber) + "\n"
        data += "\nG1 Z25\nG1 X450 Y430\nG1 Z60\nM226 P121\nG1 Z20\nM226 P121\n "
        gcode_file = open(gcodedirectory+file, "r")
        if split:
            linesplitedfile = gcode_file.read().splitlines()
            splittedlinestart = int(len(linesplitedfile) / splitin)
            alinecount = 0
            findnextupmovement = False
            for aline in linesplitedfile:
                alinecount +=1
                data += aline + "\n"
                if alinecount == splittedlinestart:
                    findnextupmovement = True

                if findnextupmovement:
                    if aline == upmovement:
                        findnextupmovement = False
                        data += "\nG1 Z25\nG1 X480 Y430\nG1 Z20\nM226 P121\n "


        else:
            data += gcode_file.read() + "\n"
        gcode_file.close()
        filenumber+=1
    output_file = open(gcodedirectory+"output.gcode", "w")

    output_file.write(data)
    output_file.close()
    print(data)


fusegcode(split=False)
# Non fonctionnel !
def removeduplicate():
    gcodedirectory = getbasedirectory(subdirectory="svgtests\\gcode")
    outfilename = gcodedirectory + "outputnodup.gcode"
    infilename = gcodedirectory + "output.gcode"
    lines_seen = set()  # holds lines already seen
    outfile = open(outfilename, "w")
    for line in open(infilename, "r"):
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
        else:
            print(line)
    outfile.close()
#removeduplicate()
def splitsvg(subfilesnumber = 8,getlengthbypos = False,dellist=None):
    sodipodi ="<sodipodi:namedview id=\"base\" pagecolor=\"#ffffff\" bordercolor=\"#666666\" borderopacity=\"1.0\" inkscape:pageopacity=\"0.0\" inkscape:pageshadow=\"2\" "
    sodipodi += "\ninkscape:zoom=\"0.35\" inkscape:cx=\"400\" inkscape:cy=\"560\" inkscape:document-units=\"mm\" inkscape:current-layer=\"layer1\" showgrid=\"false\" inkscape:window-width=\"1570\" inkscape:window-height=\"1035\" inkscape:window-x=\"-8\" inkscape:window-y=\"-8\" inkscape:window-maximized=\"1\" />"

    svg_file = getbasedirectory(subdirectory="svgtests") +"test.svg"


    doc = minidom.parse(svg_file)  # parseString also exists


    widthmm = 900
    heightmm = 865
    def getdlength(elem):
        return len(elem[0])
    def getidvalue(elem):
        return int(elem[2][4:])
    pathlist = []
    isother = True
    for path in doc.getElementsByTagName('path'):
        # Selectionne seulement un path sur 2 (évite de repasser 2 fois partout à la même place)
        if isother:
            pathlist.append([path.getAttribute('d'),path.getAttribute('style'),path.getAttribute('id'),path.getAttribute('inkscape:connector-curvature')])
        ##isother = not isother
    pathlist = sorted(pathlist, key=getdlength,reverse=True)
    if getlengthbypos:
        n = 0
        for apath in pathlist :
            print(str(getdlength(apath)) + " " + str(n))
            n += 1
        exit()
    if dellist is not None:
        pathlist = [j for i, j in enumerate(pathlist) if i not in dellist]

    #pathlist.sort(key=getdlength,reverse=True)
    #for i in pathlist:
    #    print(i[2],i[1])
    maxrange = math.ceil(len(pathlist) / subfilesnumber)
    pathindex = 0
    for filenumber in range(subfilesnumber):
        print("Starting batch " + str(filenumber))
        pathbatch = []
        n = filenumber
        while n < len(pathlist):
            pathbatch.append(pathlist[n])
            n += subfilesnumber
        pathbatch = sorted(pathbatch, key=getidvalue, reverse=True)

        for t in pathbatch:
            print(t[2],t[1])
        with open(getbasedirectory(subdirectory="svgtests")+"result" + str(filenumber)+".svg", 'w') as file:
            file.write("<svg width=\""+str(widthmm)+"mm\" height=\""+str(heightmm)+"mm\" viewBox=\"0 0 "+str(widthmm)+" "+str(heightmm)+ "\">")
            file.write(sodipodi)

            for i in pathbatch:
                    #file.write(";id="+str(i[2])+" pathindex="+str(pathindex)+" filenumber "+str(filenumber))
                    file.write("<path \n")
                    file.write("style=\""+i[1]+"\"\n")
                    file.write("d=\""+i[0]+"\"\n")
                    file.write("id=\""+i[2] + "\"\n")
                    file.write("inkscape:connector-curvature=\""+i[3] + "\"\n")
                    file.write(" />")

            file.write("</svg>")
    print("Finished !")
    path_elements = doc.getElementsByTagName('path')
    doc.unlink()
#splitsvg(subfilesnumber=10)
#splitsvg(dellist=[1,2,3,5,7,8,10,12,14,15,16,17,19,20,21,22,24,26,27,30,31,32,33,36,37,39,40,41,43,45,47,49,50,52])