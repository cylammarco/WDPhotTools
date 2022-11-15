#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute and plot multiple WDLFs with different SFH"""

import os

from matplotlib import pyplot as plt
import numpy as np

from WDPhotTools import theoretical_lf

try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))

wdlf = theoretical_lf.WDLF()
wdlf.set_ifmr_model("C08")
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0, 20.0, 0.1)
age_list = 1e9 * np.arange(2, 15, 2)

fig1, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, sharey=True, figsize=(10, 15)
)

for i, age in enumerate(age_list):

    # Constant SFR
    wdlf.set_sfr_model(mode="constant", age=age)
    _, constant_density = wdlf.compute_density(Mag=Mag)
    ax1.plot(
        Mag, np.log10(constant_density), label="{:.2f} Gyr".format(age / 1e9)
    )

    # Burst SFR
    wdlf.set_sfr_model(mode="burst", age=age, duration=1e9)
    _, burst_density = wdlf.compute_density(Mag=Mag, passband="G3")
    ax2.plot(
        Mag, np.log10(burst_density), label="{:.2f} Gyr".format(age / 1e9)
    )

    # Exponential decay SFR
    wdlf.set_sfr_model(mode="decay", age=age)
    _, decay_density = wdlf.compute_density(Mag=Mag, passband="G3")
    ax3.plot(
        Mag, np.log10(decay_density), label="{:.2f} Gyr".format(age / 1e9)
    )

ax1.legend()
ax1.grid()
ax1.set_xlim(7.5, 20)
ax1.set_ylim(-5, 0)
ax1.set_title("Star Formation History: Constant")

ax2.grid()
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_title("Star Formation History: 1 Gyr Burst")

ax3.grid()
ax3.set_xlabel(r"G$_{DR3}$ / mag")
ax3.set_title(r"Star Formation History: Exponential Decay ($\tau=3$)")

plt.savefig(
    os.path.join(
        HERE,
        "example_output",
        "wdlf_compare_sfr.png",
    )
)
