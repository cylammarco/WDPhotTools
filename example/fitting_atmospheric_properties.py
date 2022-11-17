#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fitting PSOJ1801p6254"""

import os

from WDPhotTools.fitter import WDfitter


try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))


ftr = WDfitter()

# Fitting for logg and Mbol with 11 filters for both DA and DB
ftr.fit(
    filters=[
        "g_ps1",
        "r_ps1",
        "i_ps1",
        "z_ps1",
        "y_ps1",
        "G3",
        "G3_BP",
        "G3_RP",
        "J_mko",
        "H_mko",
        "K_mko",
    ],
    mags=[
        21.1437,
        19.9678,
        19.4993,
        19.2981,
        19.1478,
        20.0533,
        20.7883,
        19.1868,
        19.45 - 0.91,
        19.96 - 1.39,
        20.40 - 1.85,
    ],
    mag_errors=[
        0.0321,
        0.0229,
        0.0083,
        0.0234,
        0.0187,
        0.006322,
        0.118615,
        0.070880,
        0.05,
        0.03,
        0.05,
    ],
    independent=["Teff", "logg"],
    distance=71.231,
    distance_err=2.0,
    initial_guess=[4000.0, 7.5],
    kwargs_for_minimize={"method": "Nelder-Mead"},
)
print(ftr.results["H"])
print(ftr.results["He"])
ftr.show_best_fit(
    display=False,
    savefig=True,
    folder=os.path.join(HERE, "example_output"),
    filename="PSOJ1801p6254",
)


# fitting with least_squares

ftr = WDfitter()

# Fitting for logg and Mbol with 11 filters for both DA and DB
ftr.fit(
    filters=[
        "g_ps1",
        "r_ps1",
        "i_ps1",
        "z_ps1",
        "y_ps1",
        "G3",
        "G3_BP",
        "G3_RP",
        "J_mko",
        "H_mko",
        "K_mko",
    ],
    mags=[
        21.1437,
        19.9678,
        19.4993,
        19.2981,
        19.1478,
        20.0533,
        20.7883,
        19.1868,
        19.45 - 0.91,
        19.96 - 1.39,
        20.40 - 1.85,
    ],
    mag_errors=[
        0.0321,
        0.0229,
        0.0083,
        0.0234,
        0.0187,
        0.006322,
        0.118615,
        0.070880,
        0.05,
        0.03,
        0.05,
    ],
    independent=["Teff", "logg"],
    distance=71.231,
    distance_err=2.0,
    initial_guess=[4000.0, 7.5],
    method="least_squares",
)
print(ftr.results["H"])
print(ftr.results["He"])
ftr.show_best_fit(
    display=False,
    savefig=True,
    folder=os.path.join(HERE, "example_output"),
    filename="PSOJ1801p6254_least_squares",
)


# Fitting for logg and Mbol with 11 filters for both DA and DB
ftr = WDfitter()
ftr.fit(
    atmosphere="H",
    filters=[
        "g_ps1",
        "r_ps1",
        "i_ps1",
        "z_ps1",
        "y_ps1",
        "G3",
        "G3_BP",
        "G3_RP",
        "J_mko",
        "H_mko",
        "K_mko",
    ],
    mags=[
        21.1437,
        19.9678,
        19.4993,
        19.2981,
        19.1478,
        20.0533,
        20.7883,
        19.1868,
        19.45 - 0.91,
        19.96 - 1.39,
        20.40 - 1.85,
    ],
    mag_errors=[
        0.0321,
        0.0229,
        0.0083,
        0.0234,
        0.0187,
        0.006322,
        0.118615,
        0.070880,
        0.05,
        0.03,
        0.05,
    ],
    independent=["Teff", "logg"],
    initial_guess=[4000.0, 7.5],
    atmosphere_interpolator="CT",
    distance=71.231,
    distance_err=2.0,
    method="emcee",
    nwalkers=100,
    nsteps=1000,
    nburns=100,
)
print(ftr.results["H"])
ftr.show_best_fit(
    atmosphere="H",
    display=False,
    savefig=True,
    folder=os.path.join(HERE, "example_output"),
    filename="PSOJ1801p6254_emcee",
)
ftr.show_corner_plot(
    figsize=(10, 10),
    display=True,
    savefig=True,
    folder=os.path.join(HERE, "example_output"),
    filename="PSOJ1801p6254_emcee_corner",
    kwarg={
        "quantiles": [0.158655, 0.5, 0.841345],
        "show_titles": True,
        "truths": [3550, 7.45],
    },
)
