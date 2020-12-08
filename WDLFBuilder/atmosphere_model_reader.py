import glob
import numpy as np
import pkg_resources
from scipy.interpolate import CloughTocher2DInterpolator


def interp_atm(z='G3',
               atmosphere='H',
               variables=['logg', 'Mbol'],
               fill_value=-np.inf,
               tol=1e-10,
               maxiter=100000,
               rescale=True):
    """
    This function interpolates the grid of synthetic photometry as a function
    of 2 variables, the default choices are 'logg' and 'Mbol'.

    Parameters
    ----------
    z: str (Default: 'G3')
        The value to be interpolated over. Choose from:
        'Teff', 'logg', 'mass', 'Mbol', 'BC', 'U', 'B', 'V', 'R', 'I', 'J',
        'H', 'Ks', 'Y_ukidss', 'J_ukidss', 'H_ukidss', 'K_ukidss', 'W1', 'W2',
        'W3', 'W4', 'S36', 'S45', 'S58', 'S80', 'u_sdss', 'g_sdss', 'r_sdss',
        'i_sdss', 'z_sdss', 'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1', 'y_ps1', 'G2',
        'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'age'.
    atmosphere: str (Default: 'H')
        The atmosphere type, 'H' or 'He'.
    variables: list (Default: ['logg', 'Mbol'])
        The parameters to be interpolated over for z.
    fill_value: numeric (Default: -np.inf)
        The fill_value in the CloughTocher2DInterpolator.
    tol: float (Default: 1e-10)
        The tolerance in the CloughTocher2DInterpolator.
    maxiter: int (Default: 100000)
        The maxiter in the CloughTocher2DInterpolator.
    rescale: boolean (Default: True)
        The rescale in the CloughTocher2DInterpolator.

    Returns
    -------
        A callable function of CloughTocher2DInterpolator.

    """

    # DA atmosphere
    if atmosphere == 'H':
        filepath = pkg_resources.resource_filename('WDLFBuilder',
                                            'wd_photometry/Table_DA.txt')
    # DB atmosphere
    elif atmosphere == 'He':
        filepath = pkg_resources.resource_filename('WDLFBuilder',
                                            'wd_photometry/Table_DB.txt')
    else:
        raise ValueError('Please choose from "H" or "He" as the atmophere '
                         'type, you have provided {}.'.format(atmosphere))

    # Prepare the array column dtype
    column_key = np.array(
        ('Teff', 'logg', 'mass', 'Mbol', 'BC', 'U', 'B', 'V', 'R', 'I', 'J',
         'H', 'Ks', 'Y_ukidss', 'J_ukidss', 'H_ukidss', 'K_ukidss', 'W1', 'W2',
         'W3', 'W4', 'S36', 'S45', 'S58', 'S80', 'u_sdss', 'g_sdss', 'r_sdss',
         'i_sdss', 'z_sdss', 'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1', 'y_ps1', 'G2',
         'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'age'))

    column_type = np.array(([np.float64] * len(column_key)))
    dtype = [(i, j) for i, j in zip(column_key, column_type)]

    # Load the synthetic photometry file in a recarray
    model = np.loadtxt(filepath, skiprows=2, dtype=dtype)

    # Interpolate with the scipy CloughTocher2DInterpolator
    atmosphere_interpolator = CloughTocher2DInterpolator(
        (model[variables[0]], model[variables[1]]),
        model[z],
        fill_value=fill_value,
        tol=tol,
        maxiter=maxiter,
        rescale=rescale)

    return atmosphere_interpolator
