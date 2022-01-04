import itertools
import numpy as np
import pkg_resources

from .GlobalSpline2D import GlobalSpline2D

# Load the reddening vectors from file
data = np.loadtxt(pkg_resources.resource_filename('WDPhotTools',
                                                  'extinction/schlafly12.csv'),
                  delimiter=',')

_xy = np.array(
    [i for i in itertools.product(data[:, 0], np.array([2.1, 3.1, 4.1, 5.1]))])
_z = data[:, 1:].flatten()
x, y, z = np.vstack((_xy.T, _z))


# Interpolating with the custom-build (extra-)interpolator
def reddening_vector(kind='cubic'):

    return GlobalSpline2D(x, y, z, kind=kind)