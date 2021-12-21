import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize

from atmosphere_model_reader import atm_reader

plt.rc('font', size=18)
plt.rc('legend', fontsize=12)


class WDfitter:
    def __init__(self):

        self.atm = atm_reader()
        self.interpolator = []
        self.fitting_params = None
        self.results = None

    def _interp_atm(self, dependent, atmosphere, independent, logg, **kwargs):

        _interpolator = self.atm.interp_atm(dependent=dependent,
                                            atmosphere=atmosphere,
                                            independent=independent,
                                            logg=logg,
                                            **kwargs)

        return _interpolator

    def _chi2_minimization(self, x, values, errors, distance, distance_err):

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in self.interpolator:

            mag.append(interp(x))

        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - values)**2. / errors_squared

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
            kwargs_for_interpolator={},
            kwargs_for_minimization={
                'method': 'Powell',
                'options': {
                    'xtol': 0.001
                }
            },
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
                                            args=(np.asarray(mags),
                                                  np.asarray(mag_errors),
                                                  distance, distance_err),
                                            **kwargs_for_minimization)

        elif method == 'emcee':

            pass

        else:

            ValueError('Unknown method. Please choose from lsq and emcee.')

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
            'reuse_interpolator': reuse_interpolator,
            'method': method,
            'kwargs_for_interpolator': kwargs_for_interpolator,
            'kwargs_for_minimization': kwargs_for_minimization,
            'kwargs_for_emcee': kwargs_for_emcee
        }

    def show_best_fit(self, figsize=(8, 6), title=None, display=True):

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        self.pivot_wavelengths = []

        for i in self.fitting_params['filters']:
            self.pivot_wavelengths.append(self.atm.column_wavelengths[i])

        ax.errorbar(self.pivot_wavelengths,
                    self.fitting_params['mags'],
                    yerr=self.fitting_params['mag_errors'],
                    linestyle='None',
                    capsize=3,
                    fmt='s',
                    label='Observed')
        ax.scatter(self.pivot_wavelengths,
                   self.fitting_params['mags'],
                   label='Best-fit',
                   color='black',
                   zorder=15)

        ax.legend()
        ax.invert_yaxis()
        ax.grid()

        ax.set_xlabel('Wavelength / A')
        ax.set_ylabel('Magnitude / mag')

        if title is None:

            ax.set_title('Best-fit {} atmosphere with {}'.format(
                self.fitting_params['atmosphere'],
                self.fitting_params['method']))

        fig.tight_layout()

        if display:
            plt.show()
