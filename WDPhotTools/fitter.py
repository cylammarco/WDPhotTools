from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
import time

from .atmosphere_model_reader import atm_reader

plt.rc('font', size=18)
plt.rc('legend', fontsize=12)


class WDfitter:
    def __init__(self):

        self.atm = atm_reader()
        self.interpolator = {'H': {}, 'He': {}}
        self.fitting_params = None
        self.results = {'H': {}, 'He': {}}
        self.bestfit_params = {'H': {}, 'He': {}}

    def _interp_atm(self, dependent, atmosphere, independent, logg, **kwargs):

        _interpolator = self.atm.interp_atm(dependent=dependent,
                                            atmosphere=atmosphere,
                                            independent=independent,
                                            logg=logg,
                                            **kwargs)

        return _interpolator

    def _chi2_minimization(self, x, values, errors, distance, distance_err,
                           interpolator):

        dist_mod = 5. * (np.log10(distance) - 1.)

        mag = []

        for interp in interpolator:

            mag.append(interp(x))

        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = np.sqrt(errors**2. + (distance_err / distance /
                                               2.302585092994046)**2.)

        chi2 = (mag - values)**2. / errors_squared

        return np.sum(chi2)

    def _chi2_minimization_distance(self, x, values, errors, interpolator):

        if (x[-1] <= 0.):

            return np.inf

        dist_mod = 5. * (np.log10(x[-1]) - 1.)

        mag = []

        for interp in interpolator:

            mag.append(interp(x[:2]))

        mag = np.asarray(mag).reshape(-1) + dist_mod
        errors_squared = errors**2.

        chi2 = (mag - values)**2. / errors_squared

        return np.sum(chi2)

    def list_atmosphere_parameters(self):

        return self.atm.list_atmosphere_parameters()

    def fit(self,
            atmosphere=['H', 'He'],
            filters=['G3', 'G3_BP', 'G3_RP'],
            mags=[None, None, None],
            mag_errors=[1., 1., 1.],
            distance=None,
            distance_err=None,
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

        if reuse_interpolator & (self.interpolator != []) & (len(
                self.interpolator) == len(filters)):

            pass

        else:

            for j in atmosphere:

                for i in filters:

                    self.interpolator[j][i] = self._interp_atm(
                        dependent=i,
                        atmosphere=j,
                        independent=independent,
                        logg=logg,
                        **kwargs_for_interpolator)

        if method == 'lsq':

            for j in atmosphere:

                if distance is None:

                    self.results[j] = optimize.minimize(
                        self._chi2_minimization_distance,
                        initial_guess + [10.],
                        args=(np.asarray(mags), np.asarray(mag_errors),
                              [self.interpolator[j][i] for i in filters]),
                        **kwargs_for_minimization)

                else:

                    self.results[j] = optimize.minimize(
                        self._chi2_minimization,
                        initial_guess,
                        args=(np.asarray(mags), np.asarray(mag_errors),
                              distance, distance_err,
                              [self.interpolator[j][i] for i in filters]),
                        **kwargs_for_minimization)

                self.bestfit_params[j]['chi2'] = self.results[j].fun

                if len(independent) == 1:

                    self.bestfit_params[j][independent[0]] = self.results[j].x
                    self.bestfit_params[j]['logg'] = logg

                else:

                    for k in range(len(independent)):

                        self.bestfit_params[j][
                            independent[k]] = self.results[j].x[k]

                for i in filters:

                    if distance is None:

                        self.bestfit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x[:2]))
                        self.bestfit_params[j]['distance'] = self.results[j].x[
                            2]
                        self.bestfit_params[j]['dist_mod'] = 5. * (
                            np.log10(self.results[j].x[2]) - 1)

                    else:

                        self.bestfit_params[j][i] = float(
                            self.interpolator[j][i](self.results[j].x))
                        self.bestfit_params[j]['distance'] = distance
                        self.bestfit_params[j]['dist_mod'] = 5. * (
                            np.log10(distance) - 1)

        elif method == 'emcee':

            raise NotImplementedError('Not implemented yet.')

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

    def show_best_fit(self,
                      figsize=(8, 6),
                      atmosphere=['H', 'He'],
                      color=['red', 'blue'],
                      title=None,
                      display=True,
                      savefig=False,
                      figname=None,
                      ext='png'):

        if isinstance(color, str):

            color = [color]

        if isinstance(atmosphere, str):

            atmosphere = [atmosphere]

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        self.pivot_wavelengths = []
        self.best_fit_mag = {'H': [], 'He': []}

        for i in self.fitting_params['filters']:

            self.pivot_wavelengths.append(self.atm.column_wavelengths[i])

            for j in atmosphere:

                self.best_fit_mag[j].append(self.bestfit_params[j][i])

        ax.errorbar(self.pivot_wavelengths,
                    self.fitting_params['mags'],
                    yerr=self.fitting_params['mag_errors'],
                    linestyle='None',
                    capsize=3,
                    fmt='s',
                    color='black',
                    label='Observed')

        for j, k in enumerate(atmosphere):

            ax.scatter(self.pivot_wavelengths,
                       self.best_fit_mag[k] +
                       self.bestfit_params[k]['dist_mod'],
                       label='Best-fit {}'.format(k),
                       color=color[j],
                       zorder=15)

        ax.legend()
        ax.invert_yaxis()
        ax.grid()

        ax.set_xlabel('Wavelength / A')
        ax.set_ylabel('Magnitude / mag')

        if title is None:

            if len(self.fitting_params['atmosphere']) == 1:

                ax.set_title('Best-fit {} atmosphere with {}'.format(
                    self.fitting_params['atmosphere'][0],
                    self.fitting_params['method']))

            else:

                ax.set_title('Best-fit {} atmosphere with {}'.format(
                    'H & He', self.fitting_params['method']))

        fig.tight_layout()

        if savefig:

            if isinstance(ext, str):

                ext = [ext]

            for e in ext:

                if figname is None:

                    figname = 'bestfit_wd_solution_{}.{}'.format(time.time(), e)

                else:

                    figname = figname + '.' + e

                plt.savefig(figname)

        if display:

            plt.show()

        return fig