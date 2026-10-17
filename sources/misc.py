import glob
import os
from pathlib import Path

#import torch
import time
from sources import globalvariables


def createdir(dirname):
    dirpath = globalvariables.getbasedirectory() + dirname
    if dirpath[-1] != "\\":
        dirpath += "\\"
    # Create if dir not exist
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    return dirpath
def settensortodevice(atensor):
    if torch.cuda.is_available():
        atensor = atensor.cuda()
    else:
        atensor = atensor.cpu()
    return atensor
def removeNan(tensor):
    # Flatten:
    shape = tensor.shape
    tensor_reshaped = tensor.reshape(shape[0], -1)
    # Drop all rows containing any nan:
    tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
    # Reshape back:
    tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0], *shape[1:])
def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def getlastfilenumber(path,extension):
    print(path)
    list_of_files = glob.glob(path + '*.'+extension)
    print(list_of_files)
    names = [int(os.path.basename(x)[0:-4]) for x in list_of_files]
    if names == []:
        return 0
    print(names)
    print(max(names))
    
    return max(names)