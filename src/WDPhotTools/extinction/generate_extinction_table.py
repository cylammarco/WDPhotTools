#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Precompute the extinction tables"""


import glob
import os

import extinction
import numpy as np
from spectres import spectres
from astropy.modeling import models
from astropy import units as u


def _fitzpatrick99(wavelength, Rv):
    """
    Parameters
    ----------
    wavelegnth: array
        Wavelength in Angstrom
    Rv: float
        The extinction in V per unit of A_V, i.e. A_V/E(B − V)

    Return
    ------
    The extinction at the given wavelength per unit of E(B − V),
    i.e. A/E(B - V)

    """

    if isinstance(wavelength, (float, int)):
        wavelength = np.array([wavelength])

    return extinction.fitzpatrick99(wavelength, 1.0, Rv) * Rv


atm_key = np.array(
    [
        "U",
        "B",
        "V",
        "R",
        "I",
        "J",
        "H",
        "Ks",
        "Y_mko",
        "J_mko",
        "H_mko",
        "K_mko",
        "W1",
        "W2",
        "W3",
        "W4",
        "S36",
        "S45",
        "S58",
        "S80",
        "u_sdss",
        "g_sdss",
        "r_sdss",
        "i_sdss",
        "z_sdss",
        "g_ps1",
        "r_ps1",
        "i_ps1",
        "z_ps1",
        "y_ps1",
        "G2",
        "G2_BP",
        "G2_RP",
        "G3",
        "G3_BP",
        "G3_RP",
        "FUV",
        "NUV",
    ]
)
filter_key = np.array(
    [
        "Generic_Johnson.U",
        "Generic_Johnson.B",
        "Generic_Johnson.V",
        "Generic_Cousins.R",
        "Generic_Cousins.I",
        "2MASS_2MASS.J",
        "2MASS_2MASS.H",
        "2MASS_2MASS.Ks",
        "UKIRT_WFCAM.Y_filter",
        "UKIRT_WFCAM.J_filter",
        "UKIRT_WFCAM.H_filter",
        "UKIRT_WFCAM.K",
        "WISE_WISE.W1",
        "WISE_WISE.W2",
        "WISE_WISE.W3",
        "WISE_WISE.W4",
        "Spitzer_IRAC.I1",
        "Spitzer_IRAC.I2",
        "Spitzer_IRAC.I3",
        "Spitzer_IRAC.I4",
        "SLOAN_SDSS.u",
        "SLOAN_SDSS.g",
        "SLOAN_SDSS.r",
        "SLOAN_SDSS.i",
        "SLOAN_SDSS.z",
        "PAN-STARRS_PS1.g",
        "PAN-STARRS_PS1.r",
        "PAN-STARRS_PS1.i",
        "PAN-STARRS_PS1.z",
        "PAN-STARRS_PS1.y",
        "GAIA_GAIA2r.G",
        "GAIA_GAIA2r.Gbp",
        "GAIA_GAIA2r.Grp",
        "GAIA_GAIA3.G",
        "GAIA_GAIA3.Gbp",
        "GAIA_GAIA3.Grp",
        "GALEX_GALEX.FUV",
        "GALEX_GALEX.NUV",
    ]
)

filter_name_mapping = {}
for i, j in zip(filter_key, atm_key):
    filter_name_mapping[i] = j

model_filelist = glob.glob(".." + os.sep + "koester_model" + os.sep + "*")
filter_filelist = glob.glob(".." + os.sep + "filter_response" + os.sep + "*")

A_1um_21 = _fitzpatrick99(10000.0, 2.1)
A_1um_26 = _fitzpatrick99(10000.0, 2.6)
A_1um_31 = _fitzpatrick99(10000.0, 3.1)
A_1um_36 = _fitzpatrick99(10000.0, 3.6)
A_1um_41 = _fitzpatrick99(10000.0, 4.1)
A_1um_46 = _fitzpatrick99(10000.0, 4.6)
A_1um_51 = _fitzpatrick99(10000.0, 5.1)

LIMIT = 1e-3

# normalisation factor of the exponent
# 0.78 comes from Shlafly et al. 2010
# 1.32 also comes from them, it is the O'Donnell extinction at 1 micron
NORM = 0.78 * 1.32 / 2.5 * LIMIT

# Get the temperature and logg
teff = []
logg = []
filters = []
rv21 = []
rv26 = []
rv31 = []
rv36 = []
rv41 = []
rv46 = []
rv51 = []
for j, model in enumerate(model_filelist):
    print(str(j + 1) + " of " + str(len(model_filelist)))
    temp = model.split("\\da")[-1].split("_")
    s_wave, s_flux = np.loadtxt(model).T
    t = float(temp[0])
    g = float(temp[1].split(".")[0]) / 100.0
    bb_wave = np.arange(s_wave[-1], 300000)
    bb = models.BlackBody(temperature=t * u.K)
    bb_flux = bb(bb_wave * u.AA)
    bb_flux /= bb_flux[0]
    bb_flux *= s_flux[-1]
    total_wave = np.concatenate((s_wave, bb_wave[1:]))
    total_flux = np.concatenate((s_flux, bb_flux[1:]))
    # get all the filters
    for i in filter_filelist:
        f_wave, f_response = np.loadtxt(i).T
        wave_bin = np.zeros_like(f_wave)
        wave_diff = np.diff(f_wave) / 2.0
        wave_bin[:-1] = wave_diff
        wave_bin[1:] += wave_diff
        wave_bin[0] += wave_diff[0]
        wave_bin[-1] += wave_diff[-1]
        filters.append(filter_name_mapping[i.split("\\")[-1][:-4]])
        teff.append(t)
        logg.append(g)
        # * 5.03411250E+07 / (3.08568e19)**2. for converting from flux to
        # photon is not needed because they cancel each other in the
        # NORMalisation. The s_wave is the only non-linear multiplier
        total_flux_resampled = spectres(
            f_wave,
            total_wave,
            total_flux * total_wave,
            fill=0.0,
            verbose=False,
        )
        # source flux convolves with filter response
        SxW = total_flux_resampled * f_response * wave_bin
        rv21.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 2.1) / A_1um_21 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv26.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 2.6) / A_1um_26 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv31.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 3.1) / A_1um_31 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv36.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 3.6) / A_1um_36 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv41.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 4.1) / A_1um_41 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv46.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 4.6) / A_1um_46 * NORM)
                )
                / np.sum(SxW)
            )
        )
        rv51.append(
            -2.5
            * np.log10(
                np.sum(
                    SxW
                    * 10.0 ** (-_fitzpatrick99(f_wave, 5.1) / A_1um_51 * NORM)
                )
                / np.sum(SxW)
            )
        )

teff = np.array(teff)
logg = np.array(logg)
filters = np.array(filters)

# np.lexsort can sort based on several columns simultaneously. The columns that
# you want to sort by need to be passed in reverse. That means
# np.lexsort((col_b,col_a)) first sorts by col_a, and then by col_b:
mask_sort = np.lexsort((filters, teff, logg))

rv21 = np.array(rv21)[mask_sort] / LIMIT
rv26 = np.array(rv26)[mask_sort] / LIMIT
rv31 = np.array(rv31)[mask_sort] / LIMIT
rv36 = np.array(rv36)[mask_sort] / LIMIT
rv41 = np.array(rv41)[mask_sort] / LIMIT
rv46 = np.array(rv46)[mask_sort] / LIMIT
rv51 = np.array(rv51)[mask_sort] / LIMIT

teff = teff[mask_sort]
logg = logg[mask_sort]
filters = filters[mask_sort]

for i in atm_key:
    mask = filters == i
    output = np.column_stack(
        (
            rv21[mask],
            rv26[mask],
            rv31[mask],
            rv36[mask],
            rv41[mask],
            rv46[mask],
            rv51[mask],
        )
    )
    np.savetxt(f"{i}.csv", output, delimiter=",")
