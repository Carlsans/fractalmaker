import os
import sys
from pathlib import Path

#from ManualFractalExploration.ManualFractalExploration import ExplorationWindow


def main():
    path = sys.argv[1]
    filename = Path(path).stem
    print(filename)
    # Si le fichier n'est pas un fractal à ouvrir, il s'agit d'une image ordinaire
    if not filename.isnumeric():
        command = '"' + "C:\\Program Files\\ImageGlass\\ImageGlass.exe"+'" ' + path
        print(command)
        os.system(command)
        exit()
    from sources.ManualFractalExploration.ManualFractalExploration import ExplorationWindow
    filenumber = int(filename)
    print(sys.path)
    test = ExplorationWindow(fractalid=filenumber)
    input()

if __name__ == "__main__":
    main()