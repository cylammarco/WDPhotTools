import numpy as np
import pkg_resources
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import interp1d


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
             'NUV', 'Age'))
        self.column_key_unit = np.array(
            ('K', '(cgs)', r'M$_\odot$', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag',
             'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'mag', 'yr'))

        self.column_names = {}
        self.column_units = {}
        for i, j, k in zip(self.column_key, self.column_key_formatted,
                           self.column_key_unit):
            self.column_names[i] = j
            self.column_units[i] = k

        self.column_type = np.array(([np.float64] * len(self.column_key)))
        self.dtype = [(i, j)
                      for i, j in zip(self.column_key, self.column_type)]

        # Load the synthetic photometry file in a recarray
        self.model_da = np.loadtxt(filepath_da, skiprows=2, dtype=self.dtype)
        self.model_db = np.loadtxt(filepath_db, skiprows=2, dtype=self.dtype)

    def interp_atm(self,
                   depedent='G3',
                   atmosphere='H',
                   independent=['logg', 'Mbol'],
                   fill_value=-np.inf,
                   tol=1e-10,
                   maxiter=100000,
                   rescale=True,
                   kind='cubic'):
        """
        This function interpolates the grid of synthetic photometry and a few
        other physical properties as a function of 2 independent variables,
        the default choices are 'logg' and 'Mbol'.

        Parameters
        ----------
        depedent: str (Default: 'G3')
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
            The parameters to be interpolated over for depedent.
        fill_value: numeric (Default: -np.inf)
            The fill_value has to be numeric for CloughTocher2DInterpolator.
            It can be "extrapolate" with the interp1d
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

        variables = np.asarray(independent).reshape(-1)

        if len(variables) == 1:

            # Interpolate with the scipy interp1d
            atmosphere_interpolator = interp1d(model[variables[0]],
                                               model[depedent],
                                               kind=kind,
                                               fill_value=fill_value)

        elif len(variables) == 2:

            # Interpolate with the scipy CloughTocher2DInterpolator
            atmosphere_interpolator = CloughTocher2DInterpolator(
                (model[variables[0]], model[variables[1]]),
                model[depedent],
                fill_value=fill_value,
                tol=tol,
                maxiter=maxiter,
                rescale=rescale)

        else:

            raise TypeError('Please provide ONE varaible name as a string or '
                            'list, or TWO varaible names in a list.')

        return atmosphere_interpolator
