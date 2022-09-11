import glob
import os
import numpy as np


folders = glob.glob("Z0.0*Y0.*")

for folder in folders:
    filelist = glob.glob(folder + os.sep + "*0.DAT")
    hb_filelist = glob.glob(folder + os.sep + "*0.HB.DAT")
    mass = []
    age = []
    mass_hb = []
    age_hb = []
    for i in filelist:
        data = np.loadtxt(i, skiprows=1)
        mass.append(float(i.split("_F7_M")[1].split(".DAT")[0]))
        age.append(data[:, 2][-1])
    for i in hb_filelist:
        data = np.loadtxt(i, skiprows=1)
        mass_hb.append(float(i.split("_F7_M")[1].split(".HB.DAT")[0]))
        age_hb.append(data[:, 2][-1])
    print(folder)
    for j, k in zip(mass, age):
        mask = np.argwhere(j == np.array(mass_hb))
        if mask.size > 0:
            extra_age = np.array(age_hb)[mask[0]][0]
        else:
            extra_age = 0.0
        print(str(j) + ", " + str(k + extra_age))
    x = input("Press Enter for next folder.")
