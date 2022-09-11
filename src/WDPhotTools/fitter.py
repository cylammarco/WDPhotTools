import copy
import corner
import emcee
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import optimize
import time

from .atmosphere_model_reader import AtmosphereModelReader
from .reddening import reddening_vector_filter, reddening_vector_interpolated
from .util import get_uncertainty_least_squares, get_uncertainty_emcee


plt.rc("font", size=18)
plt.rc("legend", fontsize=12)


class WDfitter(AtmosphereModelReader):
    """
    This class provide a set of methods to fit white dwarf properties
    photometrically.

    """

    def __init__(self):

        super(WDfitter, self).__init__()
        self.interpolator = {"H": {}, "He": {}}
        self.fitting_params = None
        # Only used if minimize or least_squares are the fitting method
        self.results = {"H": {}, "He": {}}
        self.best_fit_params = {"H": {}, "He": {}}
        self.best_fit_mag = {"H": [], "He": []}
        # Only used if emcee is the fitting method
        self.sampler = {"H": [], "He": []}
        self.samples = {"H": [], "He": []}
        self.extinction_convolved = None
        # Note this is the extinction Rv, not radial velocity RV.
        self.rv = None

    def _interp_am(
        self,
        dependent,
        atmosphere,
        independent,
        logg,
        interpolator,
        kwargs_for_RBF,
        kwargs_for_CT,
    ):
        """
        Internal method to interpolate the atmosphere grid models using
        the atmosphere_model_reader.

        """

        _interpolator = self.interp_am(
            dependent=dependent,
            atmosphere=atmosphere,
            independent=independent,
            logg=logg,
            interpolator=interpolator,
            kwargs_for_RBF=kwargs_for_RBF,
            kwargs_for_CT=kwargs_for_CT,
        )

        return _interpolator

    def interp_reddening(
        self, filters, extinction_convolved=True, kind="cubic"
    ):

        if extinction_convolved:

            self.extinction_convolved = True
            self.rv = [reddening_vector_filter(i) for i in filters]

        else:

            self.extinction_convolved = False
            rv_itp = reddening_vector_interpolated(kind=kind)
            wavelength = np.array(
                [self.column_wavelengths[i] for i in filters]
            )
            self.rv = [partial(rv_itp, w) for w in wavelength]

    def _diff2(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value
        (for scipy.optimize.least_squares).

        """

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        mag = np.asarray(mag).reshape(-1)

        err2 = (
            errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
        ) * 1.1788231063225867

        diff2 = (
            (
                10.0 ** ((obs - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_red(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value
        (for scipy.optimize.least_squares).

        """

        if not self.extinction_convolved:

            # Does not require _diff2_red_interpolated_fixed_logg because
            # it is already taken care of when generating the interpolators
            # as the extinction from SFD12 table 6 has no dependency on
            # temperature and logg
            diff2, err2 = self._diff2_red_interpolated(
                x,
                obs,
                errors,
                distance,
                distance_err,
                interpolator_filter,
                Rv,
                ebv,
                True,
            )

        else:

            if "logg" in self.fitting_params["independent"]:

                logg_pos = int(
                    np.argwhere(
                        np.array(self.fitting_params["independent"]) == "logg"
                    )
                )
                diff2, err2 = self._diff2_red_filter(
                    x,
                    obs,
                    errors,
                    distance,
                    distance_err,
                    interpolator_filter,
                    interpolator_teff,
                    logg_pos,
                    Rv,
                    ebv,
                    True,
                )

            else:

                diff2, err2 = self._diff2_red_filter_fixed_logg(
                    x,
                    obs,
                    errors,
                    distance,
                    distance_err,
                    interpolator_filter,
                    interpolator_teff,
                    logg,
                    Rv,
                    ebv,
                    True,
                )

        if return_err:

            return diff2, err2

        else:

            return diff2

    def _diff2_red_interpolated(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value.

        """

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        Av = np.array([i(Rv) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = (
            errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
        ) * 1.1788231063225867

        diff2 = (
            (
                10.0
                ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_red_filter(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg_pos,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value.

        """

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        teff = float(interpolator_teff(x))

        if not np.isfinite(teff):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        logg = x[logg_pos]
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = (
            errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
        ) * 1.1788231063225867

        diff2 = (
            (
                10.0
                ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_red_filter_fixed_logg(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value.

        """

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        teff = float(interpolator_teff(x))
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = (
            errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
        ) * 1.1788231063225867

        diff2 = (
            (
                10.0
                ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs)

    def _diff2_distance_fixed_logg(
        self, x, obs, errors, interpolator_filter, logg, return_err
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.least_squares). And
        when log(g) is fixed.

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[0]))

        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (10.0 ** ((obs - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5) - 1.0)
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance(self, x, obs, errors, interpolator_filter, return_err):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.least_squares).

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:-1]))

        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (10.0 ** ((obs - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5) - 1.0)
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance_red_interpolated(
        self, x, obs, errors, interpolator_filter, Rv, ebv, return_err
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        Av = np.array([i(Rv) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (
                10.0 ** ((obs - Av - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance_red_filter(
        self,
        x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg_pos,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        teff = float(interpolator_teff(x[:2]))
        logg = x[logg_pos]
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (
                10.0 ** ((obs - Av - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance_red_interpolated_fixed_logg(
        self, x, obs, errors, interpolator_filter, Rv, ebv, return_err
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[0]))

        Av = np.array([i(Rv) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (
                10.0 ** ((obs - Av - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance_red_filter_fixed_logg(
        self,
        x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        """

        if (x[-1] <= 0.0) or (x[-1] > 10000.0):

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[0]))

        teff = float(interpolator_teff(x[0]))
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1)
        err2 = errors**2.0

        diff2 = (
            (
                10.0 ** ((obs - Av - mag - 5.0 * np.log10(x[-1]) + 5.0) / 2.5)
                - 1.0
            )
            ** 2.0
        ) / err2

        if np.isfinite(diff2).all():

            if return_err:

                return diff2, err2

            else:

                return diff2

        else:

            if return_err:

                return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

            else:

                return np.ones_like(obs) * np.inf

    def _diff2_distance_red(
        self,
        x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.least_squares).

        """

        if not self.extinction_convolved:

            if logg is None:

                diff2, err2 = self._diff2_distance_red_interpolated(
                    x, obs, errors, interpolator_filter, Rv, ebv, True
                )

            else:

                diff2, err2 = self._diff2_distance_red_interpolated_fixed_logg(
                    x, obs, errors, interpolator_filter, Rv, ebv, True
                )

        else:

            if logg is None:

                logg_pos = int(
                    np.argwhere(
                        np.array(self.fitting_params["independent"]) == "logg"
                    )
                )
                diff2, err2 = self._diff2_distance_red_filter(
                    x,
                    obs,
                    errors,
                    interpolator_filter,
                    interpolator_teff,
                    logg_pos,
                    Rv,
                    ebv,
                    True,
                )

            else:

                diff2, err2 = self._diff2_distance_red_filter_fixed_logg(
                    x,
                    obs,
                    errors,
                    interpolator_filter,
                    interpolator_teff,
                    logg,
                    Rv,
                    ebv,
                    True,
                )

        if return_err:

            return diff2, err2

        else:

            return diff2

    def _diff2_summed(
        self, x, obs, errors, distance, distance_err, interpolator_filter
    ):
        """
        Internal method for computing the ch2-squared value
        (for scipy.optimize.minimize).

        """

        diff2 = self._diff2(
            x, obs, errors, distance, distance_err, interpolator_filter, False
        )

        return np.sum(diff2)

    def _diff2_red_summed(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value
        (for scipy.optimize.minimize).

        """

        diff2, err2 = self._diff2_red(
            x,
            obs,
            errors,
            distance,
            distance_err,
            interpolator_filter,
            interpolator_teff,
            logg,
            Rv,
            ebv,
            True,
        )

        if return_err:

            return np.sum(diff2), 1.0 / np.sum(1.0 / err2)

        else:

            return np.sum(diff2)

    def _diff2_distance_summed(
        self, x, obs, errors, interpolator_filter, logg, return_error
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.minimize).

        """

        if logg is None:

            diff2, err2 = self._diff2_distance(
                x, obs, errors, interpolator_filter, True
            )

        else:

            diff2, err2 = self._diff2_distance_fixed_logg(
                x, obs, errors, interpolator_filter, logg, True
            )

        if return_error:

            return np.sum(diff2), 1.0 / np.sum(1.0 / err2)

        else:

            return np.sum(diff2)

    def _diff2_distance_red_summed(
        self,
        x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
        return_err,
    ):
        """
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.minimize).

        """

        diff2, err2 = self._diff2_distance_red(
            x,
            obs,
            errors,
            interpolator_filter,
            interpolator_teff,
            logg,
            Rv,
            ebv,
            True,
        )

        if return_err:

            if np.isfinite(diff2).all():

                return np.sum(diff2), 1.0 / np.sum(1.0 / err2)

            else:

                return np.inf, np.inf

        else:

            if np.isfinite(diff2).all():

                return np.sum(diff2)

            else:

                return np.inf

    def _log_likelihood(
        self, x, obs, errors, distance, distance_err, interpolator_filter
    ):
        """
        Internal method for computing the ch2-squared value (for emcee).

        """

        diff2, err2 = self._diff2(
            x, obs, errors, distance, distance_err, interpolator_filter, True
        )

        if np.isfinite(diff2).all():

            return -0.5 * np.sum(diff2 + np.log(2 * np.pi * err2))

        else:

            return -np.inf

    def _log_likelihood_red(
        self,
        x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
    ):
        """
        Internal method for computing the log-likelihood value (for emcee).

        """

        diff2, err2 = self._diff2_red(
            x,
            obs,
            errors,
            distance,
            distance_err,
            interpolator_filter,
            interpolator_teff,
            logg,
            Rv,
            ebv,
            True,
        )

        if np.isfinite(diff2).all():

            return -0.5 * np.sum(diff2 + np.log(2.0 * np.pi * err2))

        else:

            return -np.inf

    def _log_likelihood_distance(
        self, x, obs, errors, interpolator_filter, logg
    ):
        """
        Internal method for computing the log-likelihood value in cases when
        the distance is not provided (for emcee).

        """

        if logg is None:

            diff2, err2 = self._diff2_distance(
                x, obs, errors, interpolator_filter, True
            )

        else:

            diff2, err2 = self._diff2_distance_fixed_logg(
                x, obs, errors, interpolator_filter, logg, True
            )

        if np.isfinite(diff2).all():

            return -0.5 * np.nansum(diff2 + np.log(2.0 * np.pi * err2))

        else:

            return -np.inf

    def _log_likelihood_distance_red(
        self,
        x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg,
        Rv,
        ebv,
    ):
        """
        Internal method for computing the log-likelihood value in cases when
        the distance is not provided (for emcee).

        """

        diff2, err2 = self._diff2_distance_red(
            x,
            obs,
            errors,
            interpolator_filter,
            interpolator_teff,
            logg,
            Rv,
            ebv,
            True,
        )

        if np.isfinite(diff2).all():

            return -0.5 * np.sum(diff2 + np.log(2.0 * np.pi * err2))

        else:

            return -np.inf

    def fit(
        self,
        atmosphere=["H", "He"],
        filters=["G3", "G3_BP", "G3_RP"],
        mags=[None, None, None],
        mag_errors=[1.0, 1.0, 1.0],
        allow_none=False,
        distance=None,
        distance_err=None,
        extinction_convolved=True,
        kind="cubic",
        Rv=0.0,
        ebv=0.0,
        independent=["Mbol", "logg"],
        initial_guess=[10.0, 8.0],
        logg=8.0,
        atmosphere_interpolator="CT",
        reuse_interpolator=False,
        method="minimize",
        nwalkers=100,
        nsteps=1000,
        nburns=100,
        progress=True,
        refine=False,
        refine_bounds=[5.0, 95.0],
        kwargs_for_RBF={},
        kwargs_for_CT={},
        kwargs_for_minimize={},
        kwargs_for_least_squares={},
        kwargs_for_emcee={},
    ):
        """
        The method to execute a photometric fit. Pure hydrogen and helium
        atmospheres fitting are supported. See `atmosphere_model_reader` for
        more information. Set allow_none to True so that `mags` can be
        provided in None to Default non-detection, it is not used in the fit
        but it allows the fitter to be reused over a large dataset where
        non-detections occur occasionally. In practice, one can add the full
        list of filters and set None for all the non-detections, however this
        is highly inefficent in memory usage: most of the interpolated grid is
        not used, and masking takes time.

        Parameters
        ----------
        atmosphere: list of str (Default: ['H', 'He'])
            Choose to fit with pure hydrogen atmosphere model and/or pure
            helium atmosphere model.
        filters: list/array of str (Default: ['G3', 'G3_BP', 'G3_RP'])
            Choose the filters to be fitted with.
        mags: list/array of float (Default: [None, None, None])
            The magnitudes in the chosen filters, in their respective
            magnitude system. None can be provided as non-detection, it does
            not contribute to the fitting.
        mag_errors: list/array of float (Default: [1., 1., 1.])
            The uncertainties in the magnitudes provided.
        allow_none: bool (Default: False)
            Set to True to detect None in the `mags` list to create a mask,
            this check requires extra run-time.
        distance: float (Default: None)
            The distance to the source, in parsec. Set to None if the
            distance is to be fitted simultanenous. Provide an initial
            guess in the `initial_guess`, or it will be initialised at
            10.0 pc.
        distance_err: float (Default: None)
            The uncertainty of the distance.
        extinction_convolved: bool (Default: True)
            When False, the A_b/E(B-V) values for filter b from Table 6 of
            Schlafly et al. 2011 are interpolated over the broadband filters.
            When False, the the A_b/E(B-V) values are from integrating
            the convolution of the response function of the filters with
            the DA spectra from Koester et al. 2010 using the equation
            provided in Schlafly et al. 2011.
        kind: str (Default: 'cubic')
            The kind of interpolation of the extinction curve.
        Rv: float (Default: None)
            The choice of Rv, only used if a numerical value is provided.
        ebv: float (Default: None)
            The magnitude of the E(B-V).
        independent: list of str (Default: ['Mbol', 'logg']
            Independent variables to be interpolated in the atmosphere model,
            these are parameters to be fitted for.
        initial_guess: list of float (Default: [10.0, 8.0])
            Starting coordinates of the minimisation. Provide an additional
            value if distance is to be fitted, it would be initialise as
            50.0 pc if not provided.
        logg: float (Default: 8.0)
            Only used if 'logg' is not included in the `independent` argument.
        atmosphere_interpolator: str (Default: 'RBF')
            Choose between 'RBF' and 'CT'.
        reuse_interpolator: bool (Default: False)
            Set to use the existing interpolated grid, it should be set to
            True if the same collection of data is fitted in the same set of
            filters with occasional non-detection (with allow_none=False).
        method: str (Default: 'minimize')
            Choose from 'minimize', 'least_squares' and 'emcee' for using the
            `scipy.optimize.minimize`, `scipy.optimize.least_squares` or the
            `emcee` respectively.
        nwalkers: int (Default: 100)
            Number of walkers (emcee method only).
        nsteps: int (Default: 500)
            Number of steps each walker walk (emcee method only).
        nburns: int (Default: 50)
            Number of steps is discarded as burn-in (emcee method only).
        progress: bool (Default: True)
            Show the progress of the emcee sampling (emcee method only).
        refine: cool (Default: True)
            Set to True to refine the minimum with `scipy.optimize.minimize`.
        refine_bounds: str (Default: [5, 95])
            The bounds of the minimizer are definited by the percentiles of
            the samples.
        kwargs_for_RBF: dict (Default: {})
            Keyword argument for the interpolator. See
            `scipy.interpolate.RBFInterpolator`.
        kwargs_for_CT: dict (Default: {})
            Keyword argument for the interpolator. See
            `scipy.interpolate.CloughTocher2DInterpolator`.
        kwargs_for_minimize: dict (Default:
            {'method': 'Powell', 'options': {'xtol': 0.001}})
            Keyword argument for the minimizer, see `scipy.optimize.minimize`.
        kwargs_for_least_squares: dict (Default: {})
            keywprd argument for the minimizer,
            see `scipy.optimize.least_squares`.
        kwargs_for_emcee: dict (Default: {})
            Keyword argument for the emcee walker.

        """
        _kwargs_for_RBF = {
            "neighbors": None,
            "smoothing": 0.0,
            "kernel": "thin_plate_spline",
            "epsilon": None,
            "degree": None,
        }
        _kwargs_for_CT = {
            "fill_value": float("-inf"),
            "tol": 1e-10,
            "maxiter": 100000,
            "rescale": True,
        }
        _kwargs_for_minimize = {"method": "Powell", "options": {"tol": 0.001}}
        _kwargs_for_least_squares = {
            "method": "lm",
        }
        _kwargs_for_emcee = {}

        _kwargs_for_RBF.update(**kwargs_for_RBF)
        _kwargs_for_CT.update(**kwargs_for_CT)
        _kwargs_for_minimize.update(**kwargs_for_minimize)
        _kwargs_for_least_squares.update(**kwargs_for_least_squares)
        _kwargs_for_emcee.update(**kwargs_for_emcee)

        # Put things into list if necessary
        if isinstance(atmosphere, str):

            atmosphere = [atmosphere]

        if isinstance(filters, str):

            filters = [filters]

        if isinstance(independent, str):

            independent = [independent]

        if isinstance(initial_guess, (float, int)):

            initial_guess = [initial_guess]

        if isinstance(initial_guess, np.ndarray):

            initial_guess = list(initial_guess.reshape(-1))

        if (
            ((Rv >= 0.0) and (self.rv is None))
            or (self.extinction_convolved != extinction_convolved)
            or (len(self.interpolator[atmosphere[0]]) - 4 != len(filters))
        ):

            self.interp_reddening(
                filters=filters,
                extinction_convolved=extinction_convolved,
                kind=kind,
            )

        if isinstance(distance, (float, int, np.float32, np.float64)):

            if not isinstance(
                distance_err, (float, int, np.float32, np.float64)
            ):

                distance_err = np.sqrt(distance)

        if distance is None:

            if len(initial_guess) == len(independent):

                initial_guess = initial_guess + [50.0]

        # Reuse the interpolator if instructed or possible
        # The +4 is to account for ['Teff', 'mass', 'Mbol', 'age']
        if (
            reuse_interpolator
            & (self.interpolator[atmosphere[0]] != [])
            & (len(self.interpolator[atmosphere[0]]) == (len(filters) + 4))
        ):

            pass

        else:

            self.interpolator = {"H": {}, "He": {}}

            for j in atmosphere:

                for i in list(filters) + ["Teff", "mass", "Mbol", "age"]:

                    # Organise the interpolators by atmosphere type
                    # and filter, note that the logg is not used
                    # if independent list contains 'logg'
                    self.interpolator[j][i] = self._interp_am(
                        dependent=i,
                        atmosphere=j,
                        independent=independent,
                        logg=logg,
                        interpolator=atmosphere_interpolator,
                        kwargs_for_RBF=_kwargs_for_RBF,
                        kwargs_for_CT=_kwargs_for_CT,
                    )

        # Mask the data and interpolator if set to detect None
        if allow_none:

            # element-wise comparison with None, so using !=
            mask = np.array(mags) != np.array([None])
            mags = np.array(mags, dtype=float)[mask]
            mag_errors = np.array(mag_errors, dtype=float)[mask]
            filters = np.array(filters)[mask]

        else:

            mags = np.array(mags, dtype=float)
            mag_errors = np.array(mag_errors, dtype=float)
            filters = np.array(filters)

        # Store the fitting params
        self.fitting_params = {
            "atmosphere": atmosphere,
            "filters": filters,
            "mags": mags,
            "mag_errors": mag_errors,
            "distance": distance,
            "distance_err": distance_err,
            "independent": independent,
            "initial_guess": initial_guess,
            "logg": logg,
            "extinction_convolved": extinction_convolved,
            "kind": kind,
            "Rv": Rv,
            "ebv": ebv,
            "reuse_interpolator": reuse_interpolator,
            "method": method,
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "nburns": nburns,
            "progress": progress,
            "refine": refine,
            "refine_bounds": refine_bounds,
            "kwargs_for_RBF": _kwargs_for_RBF,
            "kwargs_for_CT": _kwargs_for_CT,
            "kwargs_for_minimize": _kwargs_for_minimize,
            "kwargs_for_least_squares": _kwargs_for_least_squares,
            "kwargs_for_emcee": _kwargs_for_emcee,
        }

        interpolator_teff = None

        # If using the scipy.optimize.minimize()
        if method == "minimize":

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if extinction_convolved:

                    interpolator_teff = self.interpolator[j]["Teff"]

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv <= 0.0:

                        if "logg" in independent:

                            self.results[j] = optimize.minimize(
                                self._diff2_distance_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    None,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                        else:

                            self.results[j] = optimize.minimize(
                                self._diff2_distance_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    logg,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                    else:

                        if "logg" in independent:
                            self.results[j] = optimize.minimize(
                                self._diff2_distance_red_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                        else:
                            self.results[j] = optimize.minimize(
                                self._diff2_distance_red_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                # If distance is provided, fit here.
                else:

                    if Rv <= 0.0:
                        self.results[j] = optimize.minimize(
                            self._diff2_summed,
                            initial_guess,
                            args=(
                                mags,
                                mag_errors,
                                distance,
                                distance_err,
                                [self.interpolator[j][i] for i in filters],
                            ),
                            **_kwargs_for_minimize,
                        )

                    else:

                        if "logg" in independent:
                            self.results[j] = optimize.minimize(
                                self._diff2_red_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                        else:
                            self.results[j] = optimize.minimize(
                                self._diff2_red_summed,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_minimize,
                            )

                # Store the chi2
                self.best_fit_params[j]["chi2"] = self.results[j].fun
                self.best_fit_params[j]["chi2dof"] = (
                    self.results[j].fun.size - self.results[j].x.size
                )

                # Save the best fit results
                if len(independent) == 1:

                    self.best_fit_params[j][independent[0]] = self.results[
                        j
                    ].x[0]
                    self.best_fit_params[j][independent[0] + "_err"] = np.nan
                    self.best_fit_params[j]["logg"] = logg

                else:

                    for k in range(len(independent)):

                        self.best_fit_params[j][independent[k]] = self.results[
                            j
                        ].x[k]
                        self.best_fit_params[j][
                            independent[k] + "_err"
                        ] = np.nan

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    if "logg" in independent:

                        # the [:2] is to separate the distance from the filters
                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x[:2])
                        )

                    else:

                        # the [:2] is to separate the distance from the filters
                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x[0])
                        )

                    if distance is None:

                        self.best_fit_params[j]["distance"] = self.results[
                            j
                        ].x[-1]

                    else:

                        self.best_fit_params[j]["distance"] = distance

                    self.best_fit_params[j]["dist_mod"] = 5.0 * (
                        np.log10(self.best_fit_params[j]["distance"]) - 1
                    )

        # If using scipy.optimize.least_squares
        elif method == "least_squares":

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if extinction_convolved:

                    interpolator_teff = self.interpolator[j]["Teff"]

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv <= 0.0:

                        self.results[j] = optimize.least_squares(
                            self._diff2_distance,
                            initial_guess,
                            args=(
                                mags,
                                mag_errors,
                                [self.interpolator[j][i] for i in filters],
                                False,
                            ),
                            **_kwargs_for_least_squares,
                        )

                    else:

                        if "logg" in independent:

                            self.results[j] = optimize.least_squares(
                                self._diff2_distance_red,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_least_squares,
                            )

                        else:

                            self.results[j] = optimize.least_squares(
                                self._diff2_distance_red,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_least_squares,
                            )

                # If distance is provided, fit here.
                else:

                    if Rv <= 0.0:

                        self.results[j] = optimize.least_squares(
                            self._diff2,
                            initial_guess,
                            args=(
                                mags,
                                mag_errors,
                                distance,
                                distance_err,
                                [self.interpolator[j][i] for i in filters],
                                False,
                            ),
                            **_kwargs_for_least_squares,
                        )

                    else:

                        if "logg" in independent:

                            self.results[j] = optimize.least_squares(
                                self._diff2_red,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_least_squares,
                            )

                        else:

                            self.results[j] = optimize.least_squares(
                                self._diff2_red,
                                initial_guess,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                    False,
                                ),
                                **_kwargs_for_least_squares,
                            )

                # Store the chi2
                self.best_fit_params[j]["chi2"] = np.sum(self.results[j].fun)
                self.best_fit_params[j]["chi2dof"] = (
                    self.results[j].fun.size - self.results[j].x.size
                )
                # rescaled the uncertainty by the reduced_chi2
                _stdev = get_uncertainty_least_squares(self.results[j])

                # Save the best fit results
                if len(independent) == 1:

                    self.best_fit_params[j][independent[0]] = self.results[
                        j
                    ].x[0]
                    self.best_fit_params[j][independent[0] + "_err"] = _stdev
                    self.best_fit_params[j]["logg"] = logg

                else:

                    for k in range(len(independent)):

                        self.best_fit_params[j][independent[k]] = self.results[
                            j
                        ].x[k]
                        self.best_fit_params[j][
                            independent[k] + "_err"
                        ] = _stdev[k]

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    if "logg" in independent:

                        # the [:2] is to separate the distance from the filters
                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x[:2])
                        )

                    else:

                        # the [:2] is to separate the distance from the filters
                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x[0])
                        )

                    if distance is None:

                        self.best_fit_params[j]["distance"] = self.results[
                            j
                        ].x[-1]

                    else:

                        self.best_fit_params[j]["distance"] = distance

                    self.best_fit_params[j]["dist_mod"] = 5.0 * (
                        np.log10(self.best_fit_params[j]["distance"]) - 1
                    )

        # If using emcee
        elif method == "emcee":

            _initial_guess = np.array(initial_guess)
            ndim = len(_initial_guess)
            nwalkers = int(nwalkers)
            pos = (np.random.random((nwalkers, ndim)) - 0.5) * np.sqrt(
                _initial_guess
            ) + _initial_guess

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if extinction_convolved:

                    interpolator_teff = self.interpolator[j]["Teff"]

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv <= 0.0:

                        if "logg" in independent:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    None,
                                ),
                                **_kwargs_for_emcee,
                            )

                        else:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    logg,
                                ),
                                **_kwargs_for_emcee,
                            )

                    else:

                        if "logg" in independent:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance_red,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                ),
                                **_kwargs_for_emcee,
                            )

                        else:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance_red,
                                args=(
                                    mags,
                                    mag_errors,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                ),
                                **_kwargs_for_emcee,
                            )

                # If distance is provided, fit here.
                else:

                    if Rv <= 0.0:

                        self.sampler[j] = emcee.EnsembleSampler(
                            nwalkers,
                            ndim,
                            self._log_likelihood,
                            args=(
                                mags,
                                mag_errors,
                                distance,
                                distance_err,
                                [self.interpolator[j][i] for i in filters],
                            ),
                            **_kwargs_for_emcee,
                        )

                    else:

                        if "logg" in independent:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_red,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    None,
                                    Rv,
                                    ebv,
                                ),
                                **_kwargs_for_emcee,
                            )

                        else:

                            # Fixed logg is handled in the _log_likelihood_red
                            # The logg provided here is a variable as passed
                            # from the function one layer above.
                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_red,
                                args=(
                                    mags,
                                    mag_errors,
                                    distance,
                                    distance_err,
                                    [self.interpolator[j][i] for i in filters],
                                    interpolator_teff,
                                    logg,
                                    Rv,
                                    ebv,
                                ),
                                **_kwargs_for_emcee,
                            )

                self.sampler[j].run_mcmc(pos, nsteps, progress=progress)
                self.samples[j] = self.sampler[j].get_chain(
                    discard=nburns, flat=True
                )

                # Save the best fit results
                if len(independent) == 1:

                    _stdev = get_uncertainty_emcee(self.samples[j])
                    self.best_fit_params[j][independent[0]] = np.percentile(
                        self.samples[j][:, 0], 50.0
                    )
                    self.best_fit_params[j][independent[0] + "_err"] = np.mean(
                        _stdev
                    )
                    self.best_fit_params[j][independent[0] + "_16"] = _stdev[0]
                    self.best_fit_params[j][independent[0] + "_84"] = _stdev[1]

                    self.best_fit_params[j]["logg"] = logg

                else:

                    for k in range(len(independent)):

                        _stdev = get_uncertainty_emcee(self.samples[j][:, k])
                        self.best_fit_params[j][
                            independent[k]
                        ] = np.percentile(self.samples[j][:, k], 50.0)
                        self.best_fit_params[j][
                            independent[k] + "_err"
                        ] = np.mean(_stdev)
                        self.best_fit_params[j][
                            independent[k] + "_16"
                        ] = _stdev[0]
                        self.best_fit_params[j][
                            independent[k] + "_84"
                        ] = _stdev[1]

                if refine:

                    kwargs = copy.deepcopy(_kwargs_for_minimize)
                    kwargs["bounds"] = np.percentile(
                        self.samples[j], refine_bounds, axis=0
                    ).T

                    print("Refining")

                    _initial_guess = np.percentile(
                        self.samples[j], 50.0, axis=0
                    )

                    if distance is None:

                        # setting distance to infinity so that it will be
                        # turned back to None after the line appending to the
                        # intial_guess when distance has to be found
                        self.fit(
                            filters=filters,
                            mags=mags,
                            mag_errors=mag_errors,
                            allow_none=allow_none,
                            atmosphere=atmosphere,
                            logg=logg,
                            independent=independent,
                            reuse_interpolator=True,
                            method="minimize",
                            distance=None,
                            distance_err=None,
                            initial_guess=_initial_guess,
                            Rv=Rv,
                            ebv=ebv,
                            kwargs_for_minimize=kwargs,
                        )

                    else:

                        self.fit(
                            filters=filters,
                            mags=mags,
                            mag_errors=mag_errors,
                            allow_none=allow_none,
                            atmosphere=atmosphere,
                            logg=logg,
                            independent=independent,
                            reuse_interpolator=True,
                            method="minimize",
                            distance=distance,
                            distance_err=distance_err,
                            initial_guess=_initial_guess,
                            Rv=Rv,
                            ebv=ebv,
                            kwargs_for_minimize=kwargs,
                        )

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    if len(independent) == 1:

                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](
                                self.best_fit_params[j][independent[0]]
                            )
                        )

                    else:

                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](
                                self.best_fit_params[j][independent[0]],
                                self.best_fit_params[j][independent[1]],
                            )
                        )

                    if distance is None:

                        self.best_fit_params[j]["distance"] = np.percentile(
                            self.samples[j][:, -1], 50.0
                        )

                    else:

                        self.best_fit_params[j]["distance"] = distance

                    self.best_fit_params[j]["dist_mod"] = 5.0 * (
                        np.log10(self.best_fit_params[j]["distance"]) - 1
                    )

        else:

            ValueError(
                "Unknown method. Please choose from minimize, "
                "least_squares and emcee."
            )

        # Save the pivot wavelength and magnitude for each filter
        self.pivot_wavelengths = []
        for i in self.fitting_params["filters"]:

            self.pivot_wavelengths.append(self.column_wavelengths[i])

        for j in atmosphere:

            self.best_fit_mag[j] = []

            for i in self.fitting_params["filters"]:

                self.best_fit_mag[j].append(self.best_fit_params[j][i])

            for name in ["Teff", "mass", "Mbol", "age"]:

                if len(independent) == 1:

                    self.best_fit_params[j][name] = float(
                        self.interpolator[j][name](
                            self.best_fit_params[j][independent[0]]
                        )
                    )

                else:

                    self.best_fit_params[j][name] = float(
                        self.interpolator[j][name](
                            self.best_fit_params[j][independent[0]],
                            self.best_fit_params[j][independent[1]],
                        )
                    )

                if Rv > 0.0:

                    if self.extinction_convolved:

                        Av = (
                            np.array(
                                [
                                    i(
                                        [
                                            self.best_fit_params[j]["logg"],
                                            self.best_fit_params[j]["Teff"],
                                            Rv,
                                        ]
                                    )
                                    for i in self.rv
                                ],
                                dtype=np.float64,
                            ).reshape(-1)
                            * ebv
                        )

                    else:

                        Av = (
                            np.array(
                                [i(Rv) for i in self.rv], dtype=np.float64
                            ).reshape(-1)
                            * ebv
                        )

                    Av[np.isnan(Av)] = 0.0

                else:

                    Av = np.zeros(len(self.rv)).reshape(-1)

                for i, f in enumerate(self.fitting_params["filters"]):

                    self.best_fit_params[j]["Av_" + f] = Av[i]

    def show_corner_plot(
        self,
        figsize=(8, 8),
        display=True,
        savefig=False,
        folder=None,
        filename=None,
        ext=["png"],
        return_fig=True,
        kwarg={},
    ):
        """
        Generate the corner plot(s) of this fit. Only if fitting with `emcee`.

        Parameters
        ----------
        figsize: array of size 2 (Default: (8, 6))
            Set the dimension of the figure.
        display: bool (Default: True)
            Set to display the figure.
        savefig: bool (Default: False)
            Set to save the figure.
        folder: str (Default: None)
            The relative or absolute path to destination, the current working
            directory will be used if None.
        filename: str (Default: None)
            The filename of the figure. The Default filename will be used
            if None.
        ext: str (Default: ['png'])
            Image type to be saved, multiple extensions can be provided. The
            supported types are those available in `matplotlib.pyplot.savefig`.
        return_fig: bool (Default: True)
            Set to return the Figure object.
        **kwarg: dict (Default: {
            'quantiles': [0.158655, 0.5, 0.841345],
            'show_titles': True,
            'range': [0.95] * len(self.fitting_params["independent"],})
            Keyword argument for the corner.corner().

        Return
        ------
        fig: list of matplotlib.figure.Figure object
            Return if return_fig is set the True.

        """

        _kwarg = {
            "quantiles": [0.158655, 0.5, 0.841345],
            "show_titles": True,
            "range": [0.95] * len(self.fitting_params["independent"]),
        }

        _kwarg.update(**kwarg)

        if "labels" in kwarg:

            labels = _kwarg["labels"]

        else:

            labels = self.fitting_params["independent"]

            if self.fitting_params["distance"] is None:

                labels = labels + ["distance"]

        fig = []
        for i, j in enumerate(self.fitting_params["atmosphere"]):

            if self.best_fit_params[j] == {}:

                continue

            fig.append(
                corner.corner(
                    self.samples[j],
                    fig=plt.figure(figsize=figsize),
                    labels=labels,
                    titles=labels,
                    **_kwarg,
                )
            )
            plt.tight_layout()

            if savefig:

                if isinstance(ext, str):

                    ext = [ext]

                if folder is None:

                    _folder = os.getcwd()

                else:

                    _folder = os.path.abspath(folder)

                    if not os.path.exists(_folder):

                        os.makedirs(_folder)

                # Loop through the ext list to save figure into each image type
                for e in ext:

                    if filename is None:

                        _filename = "corner_plot_{}_atmosphere_{}.{}".format(
                            j, time.time(), e
                        )

                    elif isinstance(filename, (list, np.ndarray)):

                        _filename = filename[i] + "." + e

                    elif isinstance(filename, str):

                        _filename = filename + "." + e

                    else:

                        raise TypeError(
                            "Please provide the filename as a "
                            "string or a list/array of string."
                        )

                    plt.savefig(os.path.join(_folder, _filename))

        if display:

            plt.show()

        if return_fig:

            return fig

    def show_best_fit(
        self,
        figsize=(8, 6),
        atmosphere=["H", "He"],
        color=["red", "blue"],
        title=None,
        display=True,
        savefig=False,
        folder=None,
        filename=None,
        ext=["png"],
        return_fig=True,
    ):
        """
        Generate a figure with the given and fitted photometry.

        Parameters
        ----------
        figsize: array of size 2 (Default: (8, 6))
            Set the dimension of the figure.
        atmosphere: list of str (Default: ['H', 'He'])
            Choose the atmosphere type to be plotted.
        color: list of str (Default: ['red', 'blue'])
            Set the colour for the respective atmosphere type.
        title: str (Default: None)
            Set the title of the figure.
        display: bool (Default: True)
            Set to display the figure.
        savefig: bool (Default: False)
            Set to save the figure.
        folder: str (Default: None)
            The relative or absolute path to destination, the current working
            directory will be used if None.
        filename: str (Default: None)
            The filename of the figure. The Default filename will be used
            if None.
        ext: str (Default: ['png'])
            Image type to be saved, multiple extensions can be provided. The
            supported types are those available in `matplotlib.pyplot.savefig`.
        return_fig: bool (Default: True)
            Set to return the Figure object.

        Return
        ------
        fig: matplotlib.figure.Figure object
            Return if return_fig is set the True.

        """

        if isinstance(color, str):

            color = [color]

        if isinstance(atmosphere, str):

            atmosphere = [atmosphere]

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        # Plot the photometry provided
        ax.errorbar(
            self.pivot_wavelengths,
            self.fitting_params["mags"],
            yerr=self.fitting_params["mag_errors"],
            linestyle="None",
            capsize=3,
            fmt="s",
            color="black",
            label="Observed",
        )

        # Plot the fitted photometry
        for j, k in enumerate(atmosphere):

            if self.best_fit_params[k] == {}:

                continue

            reddening = [
                self.best_fit_params[k]["Av_" + f]
                for f in self.fitting_params["filters"]
            ]

            ax.scatter(
                np.array(self.pivot_wavelengths),
                np.array(self.best_fit_mag[k])
                + self.best_fit_params[k]["dist_mod"]
                + np.array(reddening),
                label="Best-fit {}".format(k),
                color=color[j],
                zorder=15,
            )

        # Other decorative stuff
        ax.legend()
        ax.invert_yaxis()
        ax.grid()

        ax.set_xlabel("Wavelength / A")
        ax.set_ylabel("Magnitude / mag")

        # Configure the title
        if title is None:

            if len(self.fitting_params["atmosphere"]) == 1:

                ax.set_title(
                    "Best-fit {} atmosphere with {}".format(
                        self.fitting_params["atmosphere"][0],
                        self.fitting_params["method"],
                    )
                )

            else:

                ax.set_title(
                    "Best-fit {} atmosphere with {}".format(
                        "H & He", self.fitting_params["method"]
                    )
                )

        else:

            ax.set_title(title)

        plt.tight_layout()

        if savefig:

            if isinstance(ext, str):

                ext = [ext]

            if folder is None:

                _folder = os.getcwd()

            else:

                _folder = os.path.abspath(folder)

                if not os.path.exists(_folder):

                    os.makedirs(_folder)

            # Loop through the ext list to save figure into each image type
            for e in ext:

                if filename is None:

                    _filename = "best_fit_wd_solution_{}.{}".format(
                        time.time(), e
                    )

                else:

                    _filename = filename + "." + e

                plt.savefig(os.path.join(_folder, _filename))

        if display:

            plt.show()

        if return_fig:

            return fig
