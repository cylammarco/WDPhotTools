import numpy as np
import pkg_resources
from scipy.interpolate import RBFInterpolator


class atm_reader:
    def __init__(self):

        # DA atmosphere
        filepath_da = pkg_resources.resource_filename(
            'WDLFBuilder', 'wd_photometry/Table_DA.txt')

        # DB atmosphere
        filepath_db = pkg_resources.resource_filename(
            'WDLFBuilder', 'wd_photometry/Table_DB.txt')

        # Prepare the array column dtype
        self.column_key = np.array(
            ('Teff', 'logg', 'mass', 'Mbol', 'BC', 'U', 'B', 'V', 'R', 'I',
             'J', 'H', 'Ks', 'Y_ukidss', 'J_ukidss', 'H_ukidss', 'K_ukidss',
             'W1', 'W2', 'W3', 'W4', 'S36', 'S45', 'S58', 'S80', 'u_sdss',
             'g_sdss', 'r_sdss', 'i_sdss', 'z_sdss', 'g_ps1', 'r_ps1', 'i_ps1',
             'z_ps1', 'y_ps1', 'G2', 'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP',
             'FUV', 'NUV', 'age'))
        self.column_key_formatted = np.array(
            (r'T$_{\mathrm{eff}}$', 'log(g)', 'Mass', r'M$_{\mathrm{bol}}$',
             'BC', r'$U$', r'$B$', r'$V$', r'$R$', r'$I$', r'$J$', r'$H$',
             r'$K_{\mathrm{s}}$', r'$Y_{\mathrm{UKIDSS}}$',
             r'$J_{\mathrm{UKIDSS}}$', r'$H_{\mathrm{UKIDSS}}$',
             r'$K_{\mathrm{UKIDSS}}$', r'$W_1$', r'$W_2$', r'$W_{3}$',
             r'$W_{4}$', r'$S_{36}$', r'$S_{45}$', r'$S_{58}$', r'$S_{80}$',
             r'u$_{\mathrm{SDSS}}$', r'$g_{\mathrm{SDSS}}$',
             r'$r_{\mathrm{SDSS}}$', r'$i_{\mathrm{SDSS}}$',
             r'$z_{\mathrm{SDSS}}$', r'$g_{\mathrm{PS1}}$',
             r'$r_{\mathrm{PS1}}$', r'$i_{\mathrm{PS1}}$',
             r'$z_{\mathrm{PS1}}$', r'$y_{\mathrm{PS1}}$',
             r'$G_{\mathrm{DR2}}$', r'$G_{\mathrm{BP, DR2}}$',
             r'$G_{\mathrm{RP, DR2}}$', r'$G{_{\mathrm{DR3}}$',
             r'$G_{\mathrm{BP, DR3}}$', r'$G_{\mathrm{RP, DR3}}$', 'FUV',
             'NUV', 'log(Age)'))
        self.column_key_unit = np.array(
            ('K', r'(cm/s$^2$)', r'M$_\odot$', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'log(yr)'))
        self.column_key_wavelength = np.array((
            0., 0., 0., 0., 0., 3585., 4371., 5478., 6504., 8020.,
            12350., 16460., 21600., 10310., 12500., 16360., 22060.,
            33682., 46179., 120717., 221944., 35378., 44780., 56962., 77978., 3557.,
            4702., 6175., 7491., 8946., 4849., 6201., 7535., 8674., 9628.,
            6229., 5037., 7752., 6218., 5110., 7769., 1535., 2301., 0.
        ))

        self.column_names = {}
        self.column_units = {}
        self.column_wavelengths = {}
        for i, j, k, l in zip(self.column_key, self.column_key_formatted,
                           self.column_key_unit, self.column_key_wavelength):
            self.column_names[i] = j
            self.column_units[i] = k
            self.column_wavelengths[i] = l

        self.column_type = np.array(([np.float64] * len(self.column_key)))
        self.dtype = [(i, j)
                      for i, j in zip(self.column_key, self.column_type)]

        # Load the synthetic photometry file in a recarray
        self.model_da = np.loadtxt(filepath_da, skiprows=2, dtype=self.dtype)
        self.model_db = np.loadtxt(filepath_db, skiprows=2, dtype=self.dtype)

        self.model_da['age'] = np.log10(self.model_da['age'])
        self.model_db['age'] = np.log10(self.model_db['age'])

    def list_atmosphere_parameters(self):

        for i, j in zip(self.column_names.items(), self.column_units.items()):

            print('Parameter: {}, Column Name: {}, Unit: {}'.format(
                i[1], i[0], j[1]))

    def interp_atm(self,
                   dependent='G3',
                   atmosphere='H',
                   independent=['logg', 'Mbol'],
                   logg=8.0,
                   neighbors=None,
                   smoothing=0.0,
                   kernel='thin_plate_spline',
                   epsilon=None,
                   degree=None):
        """
        This function interpolates the grid of synthetic photometry and a few
        other physical properties as a function of 2 independent variables,
        the default choices are 'logg' and 'Mbol'.

        Parameters
        ----------
        dependent: str (Default: 'G3')
            The value to be interpolated over. Choose from:
            'Teff', 'logg', 'mass', 'Mbol', 'BC', 'U', 'B', 'V', 'R', 'I', 'J',
            'H', 'Ks', 'Y_ukidss', 'J_ukidss', 'H_ukidss', 'K_ukidss', 'W1',
            'W2', 'W3', 'W4', 'S36', 'S45', 'S58', 'S80', 'u_sdss', 'g_sdss',
            'r_sdss', 'i_sdss', 'z_sdss', 'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1',
            'y_ps1', 'G2', 'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV',
            'NUV', 'age'.
        atmosphere: str (Default: 'H')
            The atmosphere type, 'H' or 'He'.
        independent: list (Default: ['logg', 'Mbol'])
            The parameters to be interpolated over for dependent.
        logg: float (Default: 8.0)
            Only used if independent is of length 1.
        tol: float (Default: 1e-10)
            The tolerance in the CloughTocher2DInterpolator.
        maxiter: int (Default: 100000)
            The maxiter in the CloughTocher2DInterpolator.
        rescale: boolean (Default: True)
            The rescale in the CloughTocher2DInterpolator.
        kind: str or int (Default: 'cubic')
            Only use in 1D interpolation with interp1d.

        Returns
        -------
            A callable function of CloughTocher2DInterpolator.

        """

        # DA atmosphere
        if atmosphere in ['H', 'h', 'hydrogen', 'Hydrogen', 'da', 'DA']:

            model = self.model_da

        # DB atmosphere
        elif atmosphere in ['He', 'he', 'helium', 'Helium', 'db', 'DB']:

            model = self.model_db

        else:

            raise ValueError('Please choose from "H" or "He" as the atmophere '
                             'type, you have provided {}.'.format(atmosphere))

        independent = np.asarray(independent).reshape(-1)

        if len(independent) == 1:

            if independent[0] in ['Teff', 'mass', 'Mbol', 'age']:

                independent = np.concatenate((['logg'], independent))

            else:

                raise ValueError(
                    'When ony interpolating in 1-dimension, the independent '
                    'variable has to be one of: Teff, mass, Mbol, or age.')

            _atmosphere_interpolator = RBFInterpolator(
                np.stack((model[independent[0]], model[independent[1]]),
                         -1), model[dependent], neighbors, smoothing, kernel,
                epsilon, degree)

            # Interpolate with the scipy interp1d
            atmosphere_interpolator = lambda x: _atmosphere_interpolator(
                np.array([logg, x], dtype=object).reshape(1, 2))

        elif len(independent) == 2:

            # Interpolate with the scipy CloughTocher2DInterpolator
            _atmosphere_interpolator = RBFInterpolator(
                np.stack((model[independent[0]], model[independent[1]]),
                         -1), model[dependent], neighbors, smoothing, kernel,
                epsilon, degree)

            atmosphere_interpolator = lambda x: _atmosphere_interpolator(
                np.asarray(x).reshape(1, 2))

        else:

            raise TypeError('Please provide ONE varaible name as a string or '
                            'list, or TWO varaible names in a list.')

        return atmosphere_interpolator
