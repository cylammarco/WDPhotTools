import numpy as np
from scipy import optimize

from atmosphere_model_reader import atm_reader


class WDfitter:
    def __init__(self):

        self.atm = atm_reader()
        self.interpolator = []

    def _interp_atm(self, dependent, atmosphere, independent, logg, fill_value, tol,
                    maxiter, rescale, kind):

        _interpolator = self.atm.interp_atm(dependent=dependent,
                                            atmosphere=atmosphere,
                                            independent=independent,
                                            logg=logg,
                                            fill_value=fill_value,
                                            tol=tol,
                                            maxiter=maxiter,
                                            rescale=rescale,
                                            kind=kind)

        return _interpolator

    def _chi2_minimization(self, x, values, errors, distance, distance_err):

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for i, interp in enumerate(self.interpolator):

            mag.append(interp(x))

        mag = np.asarray(mag) + dist_mod
        errors = np.sqrt(errors**2. + distance_err**2.)

        chi2 = (mag - values / errors)**2.

        return np.sum(chi2)

    def list_atmosphere_parameters(self):

        return self.atm.list_atmosphere_parameters()

    def fit(self,
            atmosphere='H',
            filters=['G3', 'G3_BP', 'G3_RP'],
            mags=[None, None, None],
            mag_errors=[1., 1., 1.],
            distance=10.,
            distance_err=0.,
            independent=['Mbol', 'logg'],
            initial_guess=[10., 8.0],
            logg=8.0,
            reuse_interpolator=False,
            method='lsq',
            kwargs_for_interpolator={
                'fill_value': -np.inf,
                'tol': 1e-12,
                'maxiter': 100000,
                'rescale': True,
                'kind': 'cubic'
            },
            kwargs_for_minimization={'method': 'Powell'},
            kwargs_for_emcee={}):

        if isinstance(filters, str):

            filters = [filters]

        if isinstance(independent, str):

            independent = [independent]

        if isinstance(initial_guess, (float, int)):

            initial_guess = [initial_guess]

        if reuse_interpolator & (self.interpolator != []) & (len(
                self.interpolator) == len(filters)):

            pass

        else:

            self.interpolator = []

            for i in filters:

                self.interpolator.append(
                    self._interp_atm(dependent=i,
                                     atmosphere=atmosphere,
                                     independent=independent,
                                     logg=logg,
                                     **kwargs_for_interpolator))

        if method == 'lsq':

            self.result = optimize.minimize(self._chi2_minimization,
                              initial_guess,
                              args=(np.asarray(mags), np.asarray(mag_errors),
                                    distance, distance_err),
                              **kwargs_for_minimization)

        elif method == 'emcee':

            pass

        else:

            ValueError('Unknown method. Please choose from lsq and emcee.')
