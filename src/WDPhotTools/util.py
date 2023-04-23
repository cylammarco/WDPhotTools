#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Some utility class/functions"""

import glob
import pkg_resources

import numpy as np
from scipy import interpolate


def get_uncertainty_least_squares(res):
    """
    Get the 1 standard deviation uncertainty of the results returned by
    least_squares().

    """

    _, _s, _vh = np.linalg.svd(res.jac, full_matrices=False)
    tol = np.finfo(float).eps * _s[0] * max(res.jac.shape)
    _w = _s > tol
    cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # robust covariance matrix
    stdev = np.sqrt(np.diag(cov))

    return stdev


def get_uncertainty_emcee(samples):
    """
    Get the 15.8655 & 84.1345 percentile of the samples returned by
    emcee.

    """

    percentiles = np.percentile(samples, [0.158655, 0.5, 0.841345])
    stdevs = np.array(
        [percentiles[1] - percentiles[0], percentiles[2] - percentiles[1]]
    )

    return stdevs


def load_ms_lifetime_datatable(filename):
    """
    Load the MS lifetime CSV files
    """

    datatable = np.loadtxt(
        glob.glob(
            pkg_resources.resource_filename(
                "WDPhotTools", f"ms_lifetime/{filename}"
            )
        )[0],
        delimiter=",",
    )
    return datatable
