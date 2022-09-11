import numpy as np
from scipy import interpolate
from numpy import linalg


# Taken from
# https://github.com/pig2015/mathpy/blob/master/polation/globalspline.py
class GlobalSpline2D(interpolate.interp2d):
    def __init__(self, x, y, z, kind="linear"):

        if kind == "linear":

            if len(x) < 2 or len(y) < 2:

                raise self.get_size_error(2, kind)

        elif kind == "cubic":

            if len(x) < 4 or len(y) < 4:

                raise self.get_size_error(4, kind)

        elif kind == "quintic":

            if len(x) < 6 or len(y) < 6:

                raise self.get_size_error(6, kind)

        else:

            raise ValueError("unidentifiable kind of spline")

        super().__init__(x, y, z, kind=kind)

        self.extrap_fd_based_xs = self._linspace_10(self.x_min, self.x_max, -4)
        self.extrap_bd_based_xs = self._linspace_10(self.x_min, self.x_max, 4)
        self.extrap_fd_based_ys = self._linspace_10(self.y_min, self.y_max, -4)
        self.extrap_bd_based_ys = self._linspace_10(self.y_min, self.y_max, 4)

    @staticmethod
    def get_size_error(size, spline_kind):

        return ValueError(
            "length of x and y must be larger or at least equal "
            "to {} when applying {} spline, assign arrays with "
            "length no less than "
            "{}".format(size, spline_kind, size)
        )

    @staticmethod
    def _extrap1d(xs, ys, tar_x):

        if isinstance(xs, np.ndarray):

            xs = np.ndarray.flatten(xs)

        if isinstance(ys, np.ndarray):

            ys = np.ndarray.flatten(ys)

        assert len(xs) >= 4
        assert len(xs) == len(ys)

        f = interpolate.InterpolatedUnivariateSpline(xs, ys)

        return f(tar_x)

    @staticmethod
    def _linspace_10(p1, p2, cut=None):

        ls = list(np.linspace(p1, p2, 10))

        if cut is None:

            return ls

        assert cut <= 10

        return ls[-cut:] if cut < 0 else ls[:cut]

    def _get_extrap_based_points(self, axis, extrap_p):

        if axis == "x":

            return (
                self.extrap_fd_based_xs
                if extrap_p > self.x_max
                else self.extrap_bd_based_xs
                if extrap_p < self.x_min
                else []
            )

        elif axis == "y":

            return (
                self.extrap_fd_based_ys
                if extrap_p > self.y_max
                else self.extrap_bd_based_ys
                if extrap_p < self.y_min
                else []
            )

        assert False, "axis unknown"

    def __call__(self, x_, y_, **kwargs):

        xs = np.atleast_1d(x_)
        ys = np.atleast_1d(y_)

        if xs.ndim != 1 or ys.ndim != 1:

            raise ValueError("x and y should both be 1-D arrays")

        pz_yqueue = []

        for y in ys:

            extrap_based_ys = self._get_extrap_based_points("y", y)

            pz_xqueue = []

            for x in xs:

                extrap_based_xs = self._get_extrap_based_points("x", x)

                if not extrap_based_xs and not extrap_based_ys:

                    # inbounds
                    pz = super().__call__(x, y, **kwargs)[0]

                elif extrap_based_xs and extrap_based_ys:

                    # both x, y atr outbounds
                    # allocate based_z from x, based_ys
                    extrap_based_zs = self.__call__(
                        x, extrap_based_ys, **kwargs
                    )

                    # allocate z of x, y from based_ys, based_zs
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y)

                elif extrap_based_xs:

                    # only x outbounds
                    extrap_based_zs = super().__call__(
                        extrap_based_xs, y, **kwargs
                    )
                    pz = self._extrap1d(extrap_based_xs, extrap_based_zs, x)

                else:

                    # only y outbounds
                    extrap_based_zs = super().__call__(
                        x, extrap_based_ys, **kwargs
                    )
                    pz = self._extrap1d(extrap_based_ys, extrap_based_zs, y)

                pz_xqueue.append(pz)

            pz_yqueue.append(pz_xqueue)

        zss = pz_yqueue

        if len(zss) == 1:

            zss = zss[0]

        return np.array(zss)


def get_uncertainty_least_squares(res):
    """
    Get the 1 standard deviation uncertainty of the results returned by
    least_squares().

    """

    _, s, Vh = linalg.svd(res.jac, full_matrices=False)
    tol = np.finfo(float).eps * s[0] * max(res.jac.shape)
    w = s > tol
    cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
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
