import socket
from pathlib import Path

#print(socket.gethostname())
def getbasedirectory(subdirectory="",nosyncdir=False):
    #laptop
    if socket.gethostname() == "DESKTOP-JUS0GDK":
        if nosyncdir:
            BASEDIRECTORY = "C:\\localtraindirnosync\\test10\\"
        else:
            BASEDIRECTORY = "C:\\localtraindir\\test10\\"
    #principal
    elif socket.gethostname() =="DESKTOP-2KUTAU8":
        if nosyncdir:
            BASEDIRECTORY = "G:\\localtraindirnosync\\test10\\"
        else:
            BASEDIRECTORY = "G:\\localtraindir\\test10\\"
    else:
        raise Exception("Enter a base directory for this machine in globalvariables.py : " + socket.gethostname())
    if subdirectory != "":
        BASEDIRECTORY += subdirectory + "\\"
    # Generate directory if not exist
    Path(BASEDIRECTORY).mkdir(parents=True, exist_ok=True)
    return BASEDIRECTORY
def getcenterpredictionmodelpath():
    #return getbasedirectory("testvisual") + "visualmodel - fonctionnel512x512.pt"
    #return getbasedirectory("testvisual") + "visualmodel.pt"
    return getbasedirectory("testvisual") + "visualmodelcomment.pt"

def gettraininghyperparameter(parametername):
    # laptop
    if socket.gethostname() == "DESKTOP-JUS0GDK":
        if parametername == "dataloaderbatchsize":
            return 50



    # principal
    elif socket.gethostname() == "DESKTOP-2KUTAU8":
        if parametername == "dataloaderbatchsize":
            return 100
    else:
        raise Exception("Enter a valid hyperparameter for this machine in globalvariables.py : " + socket.gethostname())

