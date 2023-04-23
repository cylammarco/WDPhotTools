#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute WDLFs with burst and exponential SFH"""

import os

import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import theoretical_lf


try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))


wdlf = theoretical_lf.WDLF()

wdlf.set_ifmr_model("C08")

wdlf.compute_cooling_age_interpolator()

mag = np.arange(0, 20.0, 0.2)

age_list = 1e9 * np.arange(8, 15, 1)

fig1 = plt.figure(1, figsize=(12, 8))
ax1 = plt.gca()

fig2 = plt.figure(2, figsize=(12, 8))
ax2 = plt.gca()

for i, age in enumerate(age_list):
    # Burst SFR
    wdlf.set_sfr_model(mode="burst", age=age, duration=1e9)
    _, burst_density = wdlf.compute_density(
        mag=mag,
        passband="G3",
        save_csv=True,
        folder=os.path.join(HERE, "example_output"),
    )
    ax1.plot(mag, np.log10(burst_density), label=f"{age / 1e9:.2f} Gyr")

    # Exponential decay SFR
    wdlf.set_sfr_model(mode="decay", age=age)
    wdlf.compute_cooling_age_interpolator()

    _, decay_density = wdlf.compute_density(
        mag=mag,
        passband="G3",
        save_csv=True,
        folder=os.path.join(HERE, "example_output"),
    )
    ax2.plot(mag, np.log10(decay_density), label=f"{age / 1e9:.2f} Gyr")

ax1.legend()
ax1.grid()
ax1.set_xlabel(r"G$_{DR3}$ / mag")
ax1.set_ylabel("log(arbitrary number density)")
ax1.set_xlim(7.5, 20)
ax1.set_ylim(-5, 0)
ax1.set_title("Star Formation History: 1 Gyr Burst")
fig1.savefig(
    os.path.join(
        HERE,
        "example_output",
        "burst_C16_C08_montreal_co_da_20_"
        "montreal_co_da_20_montreal_co_da_20.png",
    )
)

ax2.legend()
ax2.grid()
ax2.set_xlabel(r"G$_{DR3}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(7.5, 20)
ax2.set_ylim(-5, 0)
ax2.set_title("Star Formation History: Exponential Decay")
fig2.savefig(
    os.path.join(
        HERE,
        "example_output",
        "decay_C16_C08_montreal_co_da_20_"
        "montreal_co_da_20_montreal_co_da_20.png",
    )
)
