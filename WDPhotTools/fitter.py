from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import optimize
import time

from .atmosphere_model_reader import atm_reader

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

    def _chi2_minimization(self, x, values, errors, distance, distance_err,
                           interpolator):
        '''
        Internal method for computing the ch2-squared value.

        '''

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
        '''
        Internal method for computing the ch2-squared value in cases when
        the distance is not provided.

        '''

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
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 8.0],
            logg=8.0,
            reuse_interpolator=False,
            method='lsq',
            refine_sigma=0.0,
            refine_clip=3.0,
            kwargs_for_interpolator={},
            kwargs_for_minimization={
                'method': 'Powell',
                'options': {
                    'xtol': 0.001
                }
            },
            kwargs_for_emcee={}):
        '''
        The method to execute a photometric fit. Pure hydrogen and helium
        atmospheres fitting are supported. See `atmosphere_model_reader` for
        more information. Set allow_none to True so that `mags` can be
        provided in None to default non-detection, it is not used in the fit
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
        filters: list of str (Default: ['G3', 'G3_BP', 'G3_RP'])
            Choose the filters to be fitted with.
        mags: list of float (Default: [None, None, None])
            The magnitudes in the chosen filters, in their respective
            magnitude system. None can be provided as non-detection, it does
            not contribute to the fitting.
        mag_errors: list of float (Default: [1., 1., 1.])
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
        method: str (Default: 'lsq')
            Choose from 'lsq' and 'emcee' for using the
            `scipy.optimize.minimize` or the `emcee` respectively.
        refine_sigma: str (Default: 0.0)
            A non-zero value will use a `scipy.optimize.minimize` method after
            using `emcee` bounded by the refine_sigma * sigma from the median.
        refine_clip: str (Default: 3)
            The size of sigma-clipping before getting the median for refining
            the solution.
        kwargs_for_interpolator: dict (Default: {})
            Keyword argument for the interpolator. See
            `scipy.interpolate.CloughTocher2DInterpolator`.
        kwargs_for_minimization: dict (Default:
            {'method': 'Powell', 'options': {'xtol': 0.001}})
            Keyword argument for the minimizer, see `scipy.optimize.minimize`.
        kwargs_for_emcee: dict (Default: {})
            Keyword argument for the emcee walker.

        '''

        # Reset fitted results in case of mixing up fitted results when
        # reusing the fitter
        self.results = {'H': {}, 'He': {}}
        self.best_fit_params = {'H': {}, 'He': {}}
        self.best_fit_mag = {'H': [], 'He': []}

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

        if distance is None:

            initial_guess = initial_guess + [10.]

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
            'reuse_interpolator': reuse_interpolator,
            'method': method,
            'kwargs_for_interpolator': kwargs_for_interpolator,
            'kwargs_for_minimization': kwargs_for_minimization,
            'kwargs_for_emcee': kwargs_for_emcee
        }

        # Reuse the interpolator if instructed or possible
        # The +4 is to account for ['Teff', 'mass', 'Mbol', 'age']
        if reuse_interpolator & (self.interpolator != []) & (len(
                self.interpolator) == (len(filters) + 4)):

            pass

        else:

            for j in atmosphere:

                for i in filters + ['Teff', 'mass', 'Mbol', 'age']:

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

            mask = np.isfinite(mags)
            mags = mags[mask]
            mag_errors = mag_errors[mask]
            filters = filters[mask]

        # If using the scipy.optimize.minimize()
        if method == 'lsq':

            # Iterative through the list of atmospheres
            for j in atmosphere:

                # If distance is not provided, fit for the photometric
                # distance simultaneously using an assumed logg as provided
                if distance is None:

                    self.results[j] = optimize.minimize(
                        self._chi2_minimization_distance,
                        initial_guess,
                        args=(np.asarray(mags), np.asarray(mag_errors),
                              [self.interpolator[j][i] for i in filters]),
                        **kwargs_for_minimization)

                # If distance is provided, fit here.
                else:

                    self.results[j] = optimize.minimize(
                        self._chi2_minimization,
                        initial_guess,
                        args=(np.asarray(mags), np.asarray(mag_errors),
                              distance, distance_err,
                              [self.interpolator[j][i] for i in filters]),
                        **kwargs_for_minimization)

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

            raise NotImplementedError('Not implemented yet.')

        else:

            ValueError('Unknown method. Please choose from lsq and emcee.')

        # Save the pivot wavelength and magnitude for each filter
        self.pivot_wavelengths = []

        for i in self.fitting_params['filters']:

            self.pivot_wavelengths.append(self.atm.column_wavelengths[i])

            for j in atmosphere:

                self.best_fit_mag[j].append(self.best_fit_params[j][i])

        for j in atmosphere:

            for name in ['Teff', 'mass', 'Mbol', 'age']:

                self.best_fit_params[j][name] = float(
                    self.interpolator[j][name](self.results[j].x[:2]))

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
            The filename of the figure. The default filename will be used
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

        plt.tight_layout()

        if savefig:

            if isinstance(ext, str):

                ext = [ext]

            if folder is None:

                _folder = os.getcwd()

            else:

                _folder = os.path.relpath(folder)

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
