from extinction import fitzpatrick99
import itertools
import numpy as np
import os

from .util import GlobalSpline2D


# Interpolating with the custom-build (extra-)interpolator
def reddening_vector(kind='cubic'):

    # Load the reddening vectors from file
    data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'extinction/schlafly12.csv'),
                      delimiter=',')

    _xy = np.array([
        i
        for i in itertools.product(data[:, 0], np.array([2.1, 3.1, 4.1, 5.1]))
    ])
    _z = data[:, 1:].flatten()
    x, y, z = np.vstack((_xy.T, _z))

    return GlobalSpline2D(x, y, z, kind=kind)


def _fitzpatrick99(wavelength, Rv):
    '''
    Parameters
    ----------
    wavelegnth: array
        Wavelength in Angstrom
    Rv: float
        The extinction in V per unit of E(B − V), i.e. A_V/E(B − V)

    '''

    if isinstance(wavelength, (float, int)):

        wavelength = np.array([wavelength])

    return fitzpatrick99(wavelength, 1.0, Rv)
