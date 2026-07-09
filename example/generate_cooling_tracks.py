#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the cooling tracks imported"""

import os

from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from example_utils import get_example_output_dir


try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))

OUTPUT_DIR = get_example_output_dir()

atm = AtmosphereModelReader()

# Default passband is G3
G = atm.interp_am()
BP = atm.interp_am(dependent="G3_BP")
RP = atm.interp_am(dependent="G3_RP")

logg = np.arange(7.0, 9.5, 0.5)
Mbol = np.arange(0.0, 20.0, 0.1)

plt.figure(1, figsize=(8, 8))
for i in logg:
    logg_i = np.ones_like(Mbol) * i
    plt.plot(
        BP(logg_i, Mbol) - RP(logg_i, Mbol),
        G(logg_i, Mbol),
        label=rf"$\log(g) = {i}$",
    )

plt.ylim(20.0, 6.0)
plt.grid()
plt.legend()
plt.xlabel("(BP - RP) / mag")
plt.ylabel("G / mag")
plt.title("DA Cooling tracks")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "DA_cooling_tracks.png"))
