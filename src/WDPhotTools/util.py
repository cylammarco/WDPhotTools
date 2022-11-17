#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Some utility class/functions"""

import glob
import pkg_resources

import numpy as np
from scipy import interpolate


class GlobalSpline2D(interpolate.interp2d):
    """
    Taken from
    https://github.com/pig2015/mathpy/blob/master/polation/globalspline.py
    which extends the base interp2d to extrapolate.
    """

    def __init__(self, _x, _y, _z, kind="linear"):

        if kind == "linear":

            if len(_x) < 2 or len(_y) < 2:

                raise self.get_size_error(2, kind)

        elif kind == "cubic":

            if len(_x) < 4 or len(_y) < 4:

                raise self.get_size_error(4, kind)

        elif kind == "quintic":

            if len(_x) < 6 or len(_y) < 6:

                raise self.get_size_error(6, kind)

        else:

            raise ValueError("unidentifiable kind of spline")

        super().__init__(_x, _y, _z, kind=kind)

        self.extrap_fd_based_x_s = self._linspace_10(
            self.x_min, self.x_max, -4
        )
        self.extrap_bd_based_x_s = self._linspace_10(self.x_min, self.x_max, 4)
        self.extrap_fd_based_y_s = self._linspace_10(
            self.y_min, self.y_max, -4
        )
        self.extrap_bd_based_y_s = self._linspace_10(self.y_min, self.y_max, 4)

    @staticmethod
    def get_size_error(size, spline_kind):
        """if not enough data points"""
        return ValueError(
            f"Length of x and y must be larger or at least equal "
            f"to {size} when applying {spline_kind} spline, assign array_s "
            f"with length no less than {size}."
        )

    @staticmethod
    def _extrap_1d(x_s, y_s, tar_x):

        if isinstance(x_s, np.ndarray):

            x_s = np.ndarray.flatten(x_s)

        if isinstance(y_s, np.ndarray):

            y_s = np.ndarray.flatten(y_s)

        assert len(x_s) >= 4
        assert len(x_s) == len(y_s)

        _f = interpolate.InterpolatedUnivariateSpline(x_s, y_s)

        return _f(tar_x)

    @staticmethod
    def _linspace_10(p_1, p_2, cut=None):

        _ls = list(np.linspace(p_1, p_2, 10))

        if cut is None:

            return _ls

        else:

            cut = int(cut)

            assert cut <= 10

            return _ls[-cut:] if cut < 0 else _ls[:cut]

    def _get_extrap_based_points(self, axis, extrap_p):

        if axis == "x":

            return (
                self.extrap_fd_based_x_s
                if extrap_p > self.x_max
                else self.extrap_bd_based_x_s
                if extrap_p < self.x_min
                else []
            )

        elif axis == "y":

            return (
                self.extrap_fd_based_y_s
                if extrap_p > self.y_max
                else self.extrap_bd_based_y_s
                if extrap_p < self.y_min
                else []
            )

        assert False, "axis unknown"

    def __call__(self, _x, _y, **kwargs):

        x_s = np.atleast_1d(_x)
        y_s = np.atleast_1d(_y)

        if x_s.ndim != 1 or y_s.ndim != 1:

            raise ValueError("x and y should both be 1-D array_s")

        p_z_yqueue = []

        for y_i in y_s:

            extrap_based_y_s = self._get_extrap_based_points("y", y_i)

            p_z_xqueue = []

            for x_i in x_s:

                extrap_based_x_s = self._get_extrap_based_points("x", x_i)

                if not extrap_based_x_s and not extrap_based_y_s:

                    # inbounds
                    p_z = super().__call__(x_i, y_i, **kwargs)[0]

                elif extrap_based_x_s and extrap_based_y_s:

                    # both x, y atr outbounds
                    # allocate based_z from x, based_y_s
                    extrap_based_zs = self.__call__(
                        x_i, extrap_based_y_s, **kwargs
                    )

                    # allocate z of x, y from based_y_s, based_zs
                    p_z = self._extrap_1d(
                        extrap_based_y_s, extrap_based_zs, y_i
                    )

                elif extrap_based_x_s:

                    # only x outbounds
                    extrap_based_zs = super().__call__(
                        extrap_based_x_s, y_i, **kwargs
                    )
                    p_z = self._extrap_1d(
                        extrap_based_x_s, extrap_based_zs, x_i
                    )

                else:

                    # only y outbounds
                    extrap_based_zs = super().__call__(
                        x_i, extrap_based_y_s, **kwargs
                    )
                    p_z = self._extrap_1d(
                        extrap_based_y_s, extrap_based_zs, y_i
                    )

                p_z_xqueue.append(p_z)

            p_z_yqueue.append(p_z_xqueue)

        zss = p_z_yqueue

        if len(zss) == 1:

            zss = zss[0]

        return np.array(zss)


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
