import glob
import os
import numpy as np


folders = glob.glob("MIST_v1.2*")

for folder in folders:
    filelist = glob.glob(folder + os.sep + "*track.eep")
    mass = []
    age = []
    for i in filelist:
        data = np.loadtxt(i)
        mass.append(float(i.split("M.track")[0].split(os.sep)[-1]) / 100.0)
        age.append(data[:, 0][-1])
    print(folder)
    for i, j in zip(mass, age):
        print(str(i) + ", " + str(j))
    x = input("Press Enter for next folder.")
