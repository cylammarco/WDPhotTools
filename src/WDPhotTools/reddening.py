#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Handling interstellar reddening"""

import itertools
import os

import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator


folder_path = os.path.dirname(os.path.abspath(__file__))


# Interpolating with the custom-build (extra-)interpolator
def reddening_vector_interpolated(**kwargs):
    """
    This generates an interpolation using the pre-computed table from
    Schlafly et al. 2012 for a 7000K, log_Z = -1, and log_g = 4.5 source.

    """

    filepath = os.path.join(folder_path, "extinction", "schlafly12.csv")

    # Load the reddening vectors from file
    data = np.loadtxt(filepath, delimiter=",")

    _xy = np.array(
        [
            i
            for i in itertools.product(
                data[:, 0], np.array([2.1, 3.1, 4.1, 5.1])
            )
        ]
    )
    _z = data[:, 1:].flatten()

    temp = RBFInterpolator(_xy, _z, **kwargs)

    def _RBFInterpolator(*x):
        _x = np.array(
            [
                i
                for i in itertools.product(
                    np.array(x[0]).reshape(-1), np.array(x[1]).reshape(-1)
                )
            ],
            dtype="object",
        )
        return temp(_x.reshape(len(_x), 2))

    return _RBFInterpolator


def reddening_vector_filter(filter_name):
    """
    This generate an interpolation over the parameter space where the models
    from Koester, D. 2010; MSAI 81, 921 and Trembley & Bergeron 2010;
    ApJ696, 1755 cover. The extinction is computed for each filter available
    from the Montreal photometry grid by convolving their filter profile with
    the publicly available spectra at each temperature from the Koester models.

    See http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=koester2,
    WDPhotTools/extinction/generate_extinction_table.py and
    WDPhotTools/filter_response/

    """

    filepath = os.path.join(folder_path, "extinction", filter_name + ".csv")

    # Load the reddening vectors from file
    data = np.loadtxt(filepath, delimiter=",")

    Teff = np.array(
        [
            5000.0,
            5250.0,
            5500.0,
            5750.0,
            6000.0,
            6250.0,
            6500.0,
            6750.0,
            7000.0,
            7250.0,
            7500.0,
            7750.0,
            8000.0,
            8250.0,
            8500.0,
            8750.0,
            9000.0,
            9250.0,
            9500.0,
            9750.0,
            10000.0,
            10250.0,
            10500.0,
            10750.0,
            11000.0,
            11250.0,
            11500.0,
            11750.0,
            12000.0,
            12250.0,
            12500.0,
            12750.0,
            13000.0,
            13250.0,
            13500.0,
            13750.0,
            14000.0,
            14250.0,
            14500.0,
            14750.0,
            15000.0,
            15250.0,
            15500.0,
            15750.0,
            16000.0,
            16250.0,
            16500.0,
            16750.0,
            17000.0,
            17250.0,
            17500.0,
            17750.0,
            18000.0,
            18250.0,
            18500.0,
            18750.0,
            19000.0,
            19250.0,
            19500.0,
            19750.0,
            20000.0,
            21000.0,
            22000.0,
            23000.0,
            24000.0,
            25000.0,
            26000.0,
            27000.0,
            28000.0,
            29000.0,
            30000.0,
            32000.0,
            34000.0,
            35000.0,
            36000.0,
            38000.0,
            40000.0,
            45000.0,
            50000.0,
            60000.0,
            70000.0,
            80000.0,
        ]
    )
    logg = np.array(
        [6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5]
    )
    Rv = np.array([2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1])

    data = data.reshape(len(logg), len(Teff), len(Rv))

    # fill_value is set to None to allow extrapolation.
    # The scipy default is Nan the otherwise.
    return RegularGridInterpolator(
        (logg, Teff, Rv),
        data,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
