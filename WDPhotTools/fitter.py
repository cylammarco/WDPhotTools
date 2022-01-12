import corner
import emcee
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import optimize
import time

from .atmosphere_model_reader import atm_reader
from .reddening import reddening_vector_filter, reddening_vector_interpolated

plt.rc('font', size=18)
plt.rc('legend', fontsize=12)


class WDfitter:
    '''
    This class provide a set of methods to fit white dwarf properties
    photometrically.

    '''
    def __init__(self):

        self.atm = atm_reader()
        self.interpolator = {'H': {}, 'He': {}}
        self.fitting_params = None
        self.results = {'H': {}, 'He': {}}
        self.best_fit_params = {'H': {}, 'He': {}}
        self.best_fit_mag = {'H': [], 'He': []}
        self.sampler = {'H': [], 'He': []}
        self.samples = {'H': [], 'He': []}
        self.interpolated = None
        self.rv = None

    def _interp_atm(self, dependent, atmosphere, independent, logg, **kwargs):
        '''
        Internal method to interpolate the atmosphere grid models using
        the atmosphere_model_reader.

        '''

        _interpolator = self.atm.interp_atm(dependent=dependent,
                                            atmosphere=atmosphere,
                                            independent=independent,
                                            logg=logg,
                                            **kwargs)

        return _interpolator

    def interp_reddening(self, filters, interpolated=False, kind='cubic'):

        if interpolated:

            self.interpolated = True
            rv_itp = reddening_vector_interpolated(kind=kind)
            wavelength = np.array(
                [self.atm.column_wavelengths[i] for i in filters])
            self.rv = [partial(rv_itp, w) for w in wavelength]

        else:

            self.interpolated = False
            self.rv = [reddening_vector_filter(i) for i in filters]

    def _chi2_minimization(self, x, obs, errors, distance, distance_err,
                           interpolator_filter):
        '''
        Internal method for computing the ch2-squared value
        (for scipy.optimize.least_square).

        '''

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        mag = np.asarray(mag).reshape(-1) + dist_mod

        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - obs)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_summed(self, x, obs, errors, distance, distance_err,
                                  interpolator_filter):
        '''
        Internal method for computing the ch2-squared value
        (for scipy.optimize.minimize).

        '''

        chi2 = self._chi2_minimization(x, obs, errors, distance, distance_err,
                                       interpolator_filter)

        return np.sum(chi2)

    def _log_likelihood(self, x, obs, errors, distance, distance_err,
                        interpolator_filter):
        '''
        Internal method for computing the ch2-squared value (for emcee).

        '''

        return -0.5 * self._chi2_minimization_summed(
            x, obs, errors, distance, distance_err, interpolator_filter)

    def _chi2_minimization_red_interpolated(self, x, obs, errors, distance,
                                            distance_err, interpolator_filter,
                                            Rv, ebv):
        '''
        Internal method for computing the ch2-squared value.

        '''

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        Av = np.array([i(Rv) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_red_filter(self, x, obs, errors, distance,
                                      distance_err, interpolator_filter,
                                      interpolator_teff, logg_pos, Rv, ebv):
        '''
        Internal method for computing the ch2-squared value.

        '''

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        teff = float(interpolator_teff(x))

        if not np.isfinite(teff):

            return np.ones_like(obs) * np.inf

        logg = x[logg_pos]
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_red_filter_fixed_logg(self, x, obs, errors,
                                                 distance, distance_err,
                                                 interpolator_filter,
                                                 interpolator_teff, logg, Rv,
                                                 ebv):
        '''
        Internal method for computing the ch2-squared value.

        '''

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x))

        teff = float(interpolator_teff(x))
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_red(self, x, obs, errors, distance, distance_err,
                               interpolator_filter, interpolator_teff, logg,
                               Rv, ebv):
        '''
        Internal method for computing the ch2-squared value
        (for scipy.optimize.least_square).

        '''

        if self.interpolated:

            chi2 = self._chi2_minimization_red_interpolated(
                x, obs, errors, distance, distance_err, interpolator_filter,
                Rv, ebv)

        else:

            if logg is None:

                logg_pos = int(
                    np.argwhere(
                        np.array(self.fitting_params['independent']) ==
                        'logg'))
                chi2 = self._chi2_minimization_red_filter(
                    x, obs, errors, distance, distance_err,
                    interpolator_filter, interpolator_teff, logg_pos, Rv, ebv)

            else:

                chi2 = self._chi2_minimization_red_filter_fixed_logg(
                    x, obs, errors, distance, distance_err,
                    interpolator_filter, interpolator_teff, logg, Rv, ebv)

        return chi2

    def _chi2_minimization_red_summed(self, x, obs, errors, distance,
                                      distance_err, interpolator_filter,
                                      interpolator_teff, logg, Rv, ebv):
        '''
        Internal method for computing the ch2-squared value
        (for scipy.optimize.minimize).

        '''

        chi2 = self._chi2_minimization_red(x, obs, errors, distance,
                                           distance_err, interpolator_filter,
                                           interpolator_teff, logg, Rv, ebv)

        return np.sum(chi2)

    def _log_likelihood_red(self, x, obs, errors, distance, distance_err,
                            interpolator_filter, interpolator_teff, logg, Rv,
                            ebv):
        '''
        Internal method for computing the log-likelihood value (for emcee).

        '''

        return -0.5 * self._chi2_minimization_red_summed(
            x, obs, errors, distance, distance_err, interpolator_filter,
            interpolator_teff, logg, Rv, ebv)

    def _chi2_minimization_distance(self, x, obs, errors, interpolator_filter):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.least_square).

        '''

        if (x[-1] <= 0.):

            return np.ones_like(obs) * np.inf

        dist_mod = 5. * (np.log10(x[-1]) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = errors**2.

        chi2 = (mag - obs)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_distance_summed(self, x, obs, errors,
                                           interpolator_filter):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.minimize).

        '''

        chi2 = self._chi2_minimization_distance(x, obs, errors,
                                                interpolator_filter)

        return np.sum(chi2)

    def _log_likelihood_distance(self, x, obs, errors, interpolator_filter):
        '''
        Internal method for computing the log-likelihood value in cases when
        the distance is not provided (for emcee).

        '''

        return -0.5 * self._chi2_minimization_distance_summed(
            x, obs, errors, interpolator_filter)

    def _chi2_minimization_distance_red_interpolated(self, x, obs, errors,
                                                     interpolator_filter, Rv,
                                                     ebv):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        '''

        if (x[-1] <= 0.):

            return np.ones_like(obs) * np.inf

        dist_mod = 5. * (np.log10(x[-1]) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        Av = np.array([i(Rv) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = errors**2.

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_distance_red_filter(self, x, obs, errors,
                                               interpolator_filter,
                                               interpolator_teff, logg_pos, Rv,
                                               ebv):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        '''

        if (x[-1] <= 0.):

            return np.ones_like(obs) * np.inf

        dist_mod = 5. * (np.log10(x[-1]) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        teff = float(interpolator_teff(x[:2]))
        logg = x[logg_pos]
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = errors**2.

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_distance_red_filter_fixed_logg(
            self, x, obs, errors, interpolator_filter, interpolator_teff, logg,
            Rv, ebv):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        '''

        if (x[-1] <= 0.):

            return np.ones_like(obs) * np.inf

        dist_mod = 5. * (np.log10(x[-1]) - 1.)

        mag = []

        for interp in interpolator_filter:

            mag.append(interp(x[:2]))

        teff = float(interpolator_teff(x))
        Av = np.array([i([logg, teff, Rv]) for i in self.rv]).reshape(-1) * ebv
        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = errors**2.

        chi2 = (mag - obs + Av)**2. / errors_squared

        if np.isfinite(chi2).all():

            return chi2

        else:

            return np.ones_like(obs) * np.inf

    def _chi2_minimization_distance_red(self, x, obs, errors,
                                        interpolator_filter, interpolator_teff,
                                        logg, Rv, ebv):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.least_square).

        '''

        if self.interpolated:

            chi2 = self._chi2_minimization_distance_red_interpolated(
                x, obs, errors, interpolator_filter, Rv, ebv)

        else:

            if logg is None:

                logg_pos = int(
                    np.argwhere(
                        np.array(self.fitting_params['independent']) ==
                        'logg'))
                chi2 = self._chi2_minimization_distance_red_filter(
                    x, obs, errors, interpolator_filter, interpolator_teff,
                    logg_pos, Rv, ebv)

            else:

                chi2 = self._chi2_minimization_distance_red_filter_fixed_logg(
                    x, obs, errors, interpolator_filter, interpolator_teff,
                    logg, Rv, ebv)

        return chi2

    def _chi2_minimization_distance_red_summed(self, x, obs, errors,
                                               interpolator_filter,
                                               interpolator_teff, logg, Rv,
                                               ebv):
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided (for scipy.optimize.minimize).

        '''

        chi2 = self._chi2_minimization_distance_red(x, obs, errors,
                                                    interpolator_filter,
                                                    interpolator_teff, logg,
                                                    Rv, ebv)

        return np.sum(chi2)

    def _log_likelihood_distance_red(self, x, obs, errors, interpolator_filter,
                                     interpolator_teff, logg, Rv, ebv):
        '''
        Internal method for computing the log-likelihood value in cases when
        the distance is not provided (for emcee).

        '''

        return -0.5 * self._chi2_minimization_distance_red_summed(
            x, obs, errors, interpolator_filter, interpolator_teff, logg, Rv,
            ebv)

    def list_atmosphere_parameters(self):
        '''
        List all the parameters from the atmosphere models using the
        atmosphere_model_reader.

        '''

        return self.atm.list_atmosphere_parameters()

    def fit(self,
            atmosphere=['H', 'He'],
            filters=['G3', 'G3_BP', 'G3_RP'],
            mags=[None, None, None],
            mag_errors=[1., 1., 1.],
            allow_none=False,
            distance=None,
            distance_err=None,
            interpolated=False,
            kind='cubic',
            Rv=None,
            ebv=None,
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 8.0],
            logg=8.0,
            reuse_interpolator=False,
            method='minimize',
            nwalkers=20,
            nsteps=500,
            nburns=50,
            progress=True,
            refine=True,
            refine_bounds=[5., 95.],
            kwargs_for_interpolator={},
            kwargs_for_minimize={
                'method': 'Powell',
                'options': {
                    'xtol': 0.001
                }
            },
            kwargs_for_least_square={
                'method': 'lm',
            },
            kwargs_for_emcee={}):
        '''
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
            Set to True to detect None in the `mags` list to create a mask.
        distance: float (Default: None)
            The distance to the source, in parsec. Set to None if the
            distance is to be fitted simultanenous. Provide an initial
            guess in the `initial_guess`, or it will be initialised at
            10.0 pc.
        distance_err: float (Default: None)
            The uncertainty of the distance.
        interpolated: bool (Default: False)
            When True, the A_b/E(B-V) values for filter b from Table 6 of
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
            10.0 pc if not provided.
        logg: float (Default: 8.0)
            Only used if 'logg' is not included in the `independent` argument.
        reuse_interpolator: bool (Default: False)
            Set to use the existing interpolated grid, it should be set to
            True if the same collection of data is fitted in the same set of
            filters with occasional non-detection.
        method: str (Default: 'minimize')
            Choose from 'minimize', 'least_square' and 'emcee' for using the
            `scipy.optimize.minimize`, `scipy.optimize.least_square` or the
            `emcee` respectively.
        nwalkers: int (Default: 50)
            Number of walkers (emcee method only).
        nsteps: int (Default: 500)
            Number of steps each walker walk (emcee method only).
        nburns: int (Default: 50)
            Number of steps is discarded as burn-in (emcee method only).
        progress: bool (Default: True)
            Show the progress of the emcee sampling (emcee method only).
        refine: bool (Default: True)
            Set to refine the minimum with `scipy.optimize.minimize`.
        refine_bounds: str (Default: [5, 95])
            The bounds of the minimizer are definited by the percentiles of
            the samples.
        kwargs_for_interpolator: dict (Default: {})
            Keyword argument for the interpolator. See
            `scipy.interpolate.CloughTocher2DInterpolator`.
        kwargs_for_minimize: dict (Default:
            {'method': 'Powell', 'options': {'xtol': 0.001}})
            Keyword argument for the minimizer, see `scipy.optimize.minimize`.
        kwargs_for_least_square: dict (Default: {})
            keywprd argument for the minimizer,
            see `scipy.optimize.least_square`.
        kwargs_for_emcee: dict (Default: {})
            Keyword argument for the emcee walker.

        '''

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

            initial_guess = list(initial_guess)

        if (Rv is not None) and (self.rv is None) and (self.interpolated !=
                                                       interpolated):

            self.interp_reddening(filters=filters,
                                  interpolated=interpolated,
                                  kind=kind)

        if distance is None:

            initial_guess = initial_guess + [10.]

        if distance is np.inf:

            distance = None

        # Reuse the interpolator if instructed or possible
        # The +4 is to account for ['Teff', 'mass', 'Mbol', 'age']
        if reuse_interpolator & (self.interpolator[atmosphere[0]] != []) & (
                len(self.interpolator[atmosphere[0]]) == (len(filters) + 4)):

            pass

        else:

            for j in atmosphere:

                for i in list(filters) + ['Teff', 'mass', 'Mbol', 'age']:

                    # Organise the interpolators by atmosphere type
                    # and filter, note that the logg is not used
                    # if independent list contains 'logg'
                    self.interpolator[j][i] = self._interp_atm(
                        dependent=i,
                        atmosphere=j,
                        independent=independent,
                        logg=logg,
                        **kwargs_for_interpolator)

        # Mask the data and interpolator if set to detect None
        if allow_none:

            # element-wise comparison with None, so using !=
            mask = (np.array(mags) != np.array([None]))
            mags = np.array(mags, dtype=float)[mask]
            mag_errors = np.array(mag_errors, dtype=float)[mask]
            filters = np.array(filters)[mask]

        else:

            mags = np.array(mags, dtype=float)
            mag_errors = np.array(mag_errors, dtype=float)
            filters = np.array(filters)

        # Store the fitting params
        self.fitting_params = {
            'atmosphere': atmosphere,
            'filters': filters,
            'mags': mags,
            'mag_errors': mag_errors,
            'distance': distance,
            'distance_err': distance_err,
            'independent': independent,
            'initial_guess': initial_guess,
            'logg': logg,
            'interpolated': interpolated,
            'kind': kind,
            'Rv': Rv,
            'ebv': ebv,
            'reuse_interpolator': reuse_interpolator,
            'method': method,
            'nwalkers': nwalkers,
            'nsteps': nsteps,
            'nburns': nburns,
            'progress': progress,
            'refine': refine,
            'refine_bounds': refine_bounds,
            'kwargs_for_interpolator': kwargs_for_interpolator,
            'kwargs_for_minimize': kwargs_for_minimize,
            'kwargs_for_least_square': kwargs_for_least_square,
            'kwargs_for_emcee': kwargs_for_emcee
        }

        interpolator_teff = None

        # If using the scipy.optimize.minimize()
        if method == 'minimize':

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if not interpolated:

                    interpolator_teff = self.interpolator[j]['Teff']

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv is None:

                        self.results[j] = optimize.minimize(
                            self._chi2_minimization_distance_summed,
                            initial_guess,
                            args=(mags, mag_errors,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_minimize)

                    else:

                        if 'logg' in independent:

                            self.results[j] = optimize.minimize(
                                self._chi2_minimization_distance_red_summed,
                                initial_guess,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, None, Rv, ebv),
                                **kwargs_for_minimize)

                        else:

                            self.results[j] = optimize.minimize(
                                self._chi2_minimization_distance_red_summed,
                                initial_guess,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, logg, Rv, ebv),
                                **kwargs_for_minimize)

                # If distance is provided, fit here.
                else:

                    if Rv is None:

                        self.results[j] = optimize.minimize(
                            self._chi2_minimization_summed,
                            initial_guess,
                            args=(mags, mag_errors, distance, distance_err,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_minimize)

                    else:

                        if 'logg' in independent:

                            self.results[j] = optimize.minimize(
                                self._chi2_minimization_red_summed,
                                initial_guess,
                                args=(mags, mag_errors, distance, distance_err,
                                      [
                                          self.interpolator[j][i]
                                          for i in filters
                                      ], interpolator_teff, None, Rv, ebv),
                                **kwargs_for_minimize)

                        else:

                            self.results[j] = optimize.minimize(
                                self._chi2_minimization_red_summed,
                                initial_guess,
                                args=(mags, mag_errors, distance, distance_err,
                                      [
                                          self.interpolator[j][i]
                                          for i in filters
                                      ], interpolator_teff, logg, Rv, ebv),
                                **kwargs_for_minimize)

                # Store the chi2
                self.best_fit_params[j]['chi2'] = self.results[j].fun

                # Save the best fit results
                if len(independent) == 1:

                    self.best_fit_params[j][independent[0]] = self.results[j].x
                    self.best_fit_params[j]['logg'] = logg

                else:

                    for k in range(len(independent)):

                        self.best_fit_params[j][
                            independent[k]] = self.results[j].x[k]

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    # the [:2] is to separate the distance from the filters
                    self.best_fit_params[j][i] = float(self.interpolator[j][i](
                        self.results[j].x[:2]))

                    if distance is None:

                        self.best_fit_params[j]['distance'] =\
                            self.results[j].x[-1]

                    else:

                        self.best_fit_params[j]['distance'] = distance

                    self.best_fit_params[j]['dist_mod'] = 5. * (
                        np.log10(self.best_fit_params[j]['distance']) - 1)

        # If using scipy.optimize.least_square
        elif method == 'least_square':

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if not interpolated:

                    interpolator_teff = self.interpolator[j]['Teff']

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv is None:

                        self.results[j] = optimize.least_squares(
                            self._chi2_minimization_distance,
                            initial_guess,
                            args=(mags, mag_errors,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_least_square)

                    else:

                        if 'logg' in independent:

                            self.results[j] = optimize.least_squares(
                                self._chi2_minimization_distance_red,
                                initial_guess,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, None, Rv, ebv),
                                **kwargs_for_least_square)

                        else:

                            self.results[j] = optimize.least_squares(
                                self._chi2_minimization_distance_red,
                                initial_guess,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, logg, Rv, ebv),
                                **kwargs_for_least_square)

                # If distance is provided, fit here.
                else:

                    if Rv is None:

                        self.results[j] = optimize.least_squares(
                            self._chi2_minimization,
                            initial_guess,
                            args=(mags, mag_errors, distance, distance_err,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_least_square)

                    else:

                        if 'logg' in independent:

                            self.results[j] = optimize.least_squares(
                                self._chi2_minimization_red,
                                initial_guess,
                                args=(mags, mag_errors, distance, distance_err,
                                      [
                                          self.interpolator[j][i]
                                          for i in filters
                                      ], interpolator_teff, None, Rv, ebv),
                                **kwargs_for_least_square)

                        else:

                            self.results[j] = optimize.least_squares(
                                self._chi2_minimization_red,
                                initial_guess,
                                args=(mags, mag_errors, distance, distance_err,
                                      [
                                          self.interpolator[j][i]
                                          for i in filters
                                      ], interpolator_teff, logg, Rv, ebv),
                                **kwargs_for_least_square)

                # Store the chi2
                self.best_fit_params[j]['chi2'] = self.results[j].fun

                # Save the best fit results
                if len(independent) == 1:

                    self.best_fit_params[j][independent[0]] = self.results[j].x
                    self.best_fit_params[j]['logg'] = logg

                else:

                    for k in range(len(independent)):

                        self.best_fit_params[j][
                            independent[k]] = self.results[j].x[k]

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    # the [:2] is to separate the distance from the filters
                    self.best_fit_params[j][i] = float(self.interpolator[j][i](
                        self.results[j].x[:2]))

                    if distance is None:

                        self.best_fit_params[j]['distance'] =\
                            self.results[j].x[-1]

                    else:

                        self.best_fit_params[j]['distance'] = distance

                    self.best_fit_params[j]['dist_mod'] = 5. * (
                        np.log10(self.best_fit_params[j]['distance']) - 1)

        # If using emcee
        elif method == 'emcee':

            _initial_guess = np.array(initial_guess)
            ndim = len(_initial_guess)
            nwalkers = int(nwalkers)
            pos = np.random.random(
                (nwalkers, ndim)) * np.sqrt(_initial_guess) + _initial_guess

            # Iterative through the list of atmospheres
            for j in atmosphere:

                if not interpolated:

                    interpolator_teff = self.interpolator[j]['Teff']

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    if Rv is None:

                        self.sampler[j] = emcee.EnsembleSampler(
                            nwalkers,
                            ndim,
                            self._log_likelihood_distance,
                            args=(mags, mag_errors,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_emcee)

                    else:

                        if 'logg' in independent:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance_red,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, None, Rv, ebv),
                                **kwargs_for_emcee)

                        else:

                            self.sampler[j] = emcee.EnsembleSampler(
                                nwalkers,
                                ndim,
                                self._log_likelihood_distance_red,
                                args=(mags, mag_errors, [
                                    self.interpolator[j][i] for i in filters
                                ], interpolator_teff, logg, Rv, ebv),
                                **kwargs_for_emcee)

                # If distance is provided, fit here.
                else:

                    if Rv is None:

                        self.sampler[j] = emcee.EnsembleSampler(
                            nwalkers,
                            ndim,
                            self._log_likelihood,
                            args=(mags, mag_errors, distance, distance_err,
                                  [self.interpolator[j][i] for i in filters]),
                            **kwargs_for_emcee)

                    else:

                        self.sampler[j] = emcee.EnsembleSampler(
                            nwalkers,
                            ndim,
                            self._log_likelihood_red,
                            args=(mags, mag_errors, distance, distance_err,
                                  [self.interpolator[j][i] for i in filters
                                   ], interpolator_teff, logg, Rv, ebv),
                            **kwargs_for_emcee)

                self.sampler[j].run_mcmc(pos, nsteps, progress=progress)
                self.samples[j] = self.sampler[j].get_chain(discard=nburns,
                                                            flat=True)

                # Save the best fit results
                if len(independent) == 1:

                    self.best_fit_params[j][independent[0]] = np.percentile(
                        self.samples[j][:, 0], [50])
                    self.best_fit_params[j]['logg'] = logg

                else:

                    for k in range(len(independent)):

                        self.best_fit_params[j][
                            independent[k]] = np.percentile(
                                self.samples[j][:, k], [50])

                if refine:

                    if distance is None:

                        # setting distance to infinity so that it will be
                        # turned back to None after the line appending to the
                        # intial_guess when distance has to be found
                        self.fit(filters=filters,
                                 mags=mags,
                                 mag_errors=mag_errors,
                                 allow_none=allow_none,
                                 atmosphere=atmosphere,
                                 logg=logg,
                                 independent=independent,
                                 reuse_interpolator=True,
                                 method='minimize',
                                 distance=np.inf,
                                 distance_err=None,
                                 initial_guess=np.percentile(self.samples[j],
                                                             [50],
                                                             axis=0),
                                 Rv=Rv,
                                 ebv=ebv,
                                 kwargs_for_minimize={
                                     'bounds':
                                     np.percentile(self.samples[j],
                                                   refine_bounds,
                                                   axis=0).T
                                 })

                    else:

                        self.fit(filters=filters,
                                 mags=mags,
                                 mag_errors=mag_errors,
                                 allow_none=allow_none,
                                 atmosphere=atmosphere,
                                 logg=logg,
                                 independent=independent,
                                 reuse_interpolator=True,
                                 method='minimize',
                                 distance=distance,
                                 distance_err=distance_err,
                                 initial_guess=np.percentile(self.samples[j],
                                                             [50],
                                                             axis=0),
                                 Rv=Rv,
                                 ebv=ebv,
                                 kwargs_for_minimize={
                                     'bounds':
                                     np.percentile(self.samples[j],
                                                   refine_bounds,
                                                   axis=0).T
                                 })

                # Get the fitted parameters, the content of results vary
                # depending on the choise of minimizer.
                for i in filters:

                    if len(independent) == 1:

                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](
                                self.best_fit_params[j][independent[0]]))

                    else:

                        self.best_fit_params[j][i] = float(
                            self.interpolator[j][i](
                                self.best_fit_params[j][independent[0]],
                                self.best_fit_params[j][independent[1]]))

                    if distance is None:

                        self.best_fit_params[j]['distance'] =\
                            np.percentile(self.samples[j][:, -1], [50])

                    else:

                        self.best_fit_params[j]['distance'] = distance

                    self.best_fit_params[j]['dist_mod'] = 5. * (
                        np.log10(self.best_fit_params[j]['distance']) - 1)

        else:

            ValueError('Unknown method. Please choose from minimize, '
                       'least_square and emcee.')

        # Save the pivot wavelength and magnitude for each filter
        self.pivot_wavelengths = []
        for i in self.fitting_params['filters']:

            self.pivot_wavelengths.append(self.atm.column_wavelengths[i])

        for j in atmosphere:

            self.best_fit_mag[j] = []

            for i in self.fitting_params['filters']:

                self.best_fit_mag[j].append(self.best_fit_params[j][i])

            for name in ['Teff', 'mass', 'Mbol', 'age']:

                if len(independent) == 1:

                    self.best_fit_params[j][name] = float(
                        self.interpolator[j][name](
                            self.best_fit_params[j][independent[0]]))

                else:

                    self.best_fit_params[j][name] = float(
                        self.interpolator[j][name](
                            self.best_fit_params[j][independent[0]],
                            self.best_fit_params[j][independent[1]]))

    def show_corner_plot(self,
                         figsize=(8, 8),
                         display=True,
                         savefig=False,
                         folder=None,
                         filename=None,
                         ext=['png'],
                         return_fig=True,
                         kwarg={
                             'quantiles': [0.158655, 0.5, 0.841345],
                             'show_titles': True
                         }):
        '''
        Generate the corner plot(s) of this fit.

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
            'show_titles': True})
            Keyword argument for the corner.corner().

        Return
        ------
        fig: list of matplotlib.figure.Figure object
            Return if return_fig is set the True.

        '''

        if 'labels' in kwarg:

            labels = kwarg['labels']

        else:

            labels = self.fitting_params['independent']

            if self.fitting_params['distance'] is None:

                labels = labels + ['distance']

        fig = []
        for i, j in enumerate(self.fitting_params['atmosphere']):

            fig.append(
                corner.corner(self.samples[j],
                              fig=plt.figure(figsize=figsize),
                              labels=labels,
                              titles=labels,
                              **kwarg))
            plt.tight_layout()

            if savefig:

                if isinstance(ext, str):

                    ext = [ext]

                if folder is None:

                    _folder = os.getcwd()

                else:

                    _folder = os.path.abspath(folder)

                    if not os.path.exists(_folder):

                        os.mkdir(_folder)

                # Loop through the ext list to save figure into each image type
                for e in ext:

                    if filename is None:

                        _filename = 'corner_plot_{}_atmosphere_{}.{}'.format(
                            j, time.time(), e)

                    elif isinstance(filename, (list, np.ndarray)):

                        _filename = filename[i] + '.' + e

                    elif isinstance(filename, str):

                        _filename = filename + '.' + e

                    else:

                        raise TypeError('Please provide the filename as a '
                                        'string or a list/array of string.')

                    plt.savefig(os.path.join(_folder, _filename))

        if display:

            plt.show()

        if return_fig:

            return fig

    def show_best_fit(self,
                      figsize=(8, 6),
                      atmosphere=['H', 'He'],
                      color=['red', 'blue'],
                      title=None,
                      display=True,
                      savefig=False,
                      folder=None,
                      filename=None,
                      ext=['png'],
                      return_fig=True):
        '''
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

        '''

        if isinstance(color, str):

            color = [color]

        if isinstance(atmosphere, str):

            atmosphere = [atmosphere]

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        # Plot the photometry provided
        ax.errorbar(self.pivot_wavelengths,
                    self.fitting_params['mags'],
                    yerr=self.fitting_params['mag_errors'],
                    linestyle='None',
                    capsize=3,
                    fmt='s',
                    color='black',
                    label='Observed')

        # Plot the fitted photometry
        for j, k in enumerate(atmosphere):

            ax.scatter(self.pivot_wavelengths,
                       self.best_fit_mag[k] +
                       self.best_fit_params[k]['dist_mod'],
                       label='Best-fit {}'.format(k),
                       color=color[j],
                       zorder=15)

        # Other decorative stuff
        ax.legend()
        ax.invert_yaxis()
        ax.grid()

        ax.set_xlabel('Wavelength / A')
        ax.set_ylabel('Magnitude / mag')

        # Configure the title
        if title is None:

            if len(self.fitting_params['atmosphere']) == 1:

                ax.set_title('Best-fit {} atmosphere with {}'.format(
                    self.fitting_params['atmosphere'][0],
                    self.fitting_params['method']))

            else:

                ax.set_title('Best-fit {} atmosphere with {}'.format(
                    'H & He', self.fitting_params['method']))

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

                    os.mkdir(_folder)

            # Loop through the ext list to save figure into each image type
            for e in ext:

                if filename is None:

                    _filename = 'best_fit_wd_solution_{}.{}'.format(
                        time.time(), e)

                else:

                    _filename = filename + '.' + e

                plt.savefig(os.path.join(_folder, _filename))

        if display:

            plt.show()

        if return_fig:

            return fig
