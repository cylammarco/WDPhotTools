import itertools
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

from .util import GlobalSpline2D

folder_path = os.path.dirname(os.path.abspath(__file__))

# Interpolating with the custom-build (extra-)interpolator
def reddening_vector_interpolated(kind='cubic'):

    # Load the reddening vectors from file
    data = np.loadtxt(os.path.join(folder_path,
                                   'extinction/schlafly12.csv'),
                      delimiter=',')

    _xy = np.array([
        i
        for i in itertools.product(data[:, 0], np.array([2.1, 3.1, 4.1, 5.1]))
    ])
    _z = data[:, 1:].flatten()
    x, y, z = np.vstack((_xy.T, _z))

    return GlobalSpline2D(x, y, z, kind=kind)


def reddening_vector_filter(filter):

    # Load the reddening vectors from file
    data = np.loadtxt(os.path.join(folder_path,
                                   'extinction/{}.csv'.format(filter)),
                      delimiter=',')

    Teff = np.array([
        5000., 5250., 5500., 5750., 6000., 6250., 6500., 6750., 7000., 7250.,
        7500., 7750., 8000., 8250., 8500., 8750., 9000., 9250., 9500., 9750.,
        10000., 10250., 10500., 10750., 11000., 11250., 11500., 11750., 12000.,
        12250., 12500., 12750., 13000., 13250., 13500., 13750., 14000., 14250.,
        14500., 14750., 15000., 15250., 15500., 15750., 16000., 16250., 16500.,
        16750., 17000., 17250., 17500., 17750., 18000., 18250., 18500., 18750.,
        19000., 19250., 19500., 19750., 20000., 21000., 22000., 23000., 24000.,
        25000., 26000., 27000., 28000., 29000., 30000., 32000., 34000., 35000.,
        36000., 38000., 40000., 45000., 50000., 60000., 70000., 80000.
    ])
    logg = np.array(
        [6.5, 6.75, 7., 7.25, 7.5, 7.75, 8., 8.25, 8.5, 8.75, 9., 9.25, 9.5])
    Rv = np.array([2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1])

    data = data.reshape(len(logg), len(Teff), len(Rv))

    return RegularGridInterpolator((logg, Teff, Rv),
                                   data,
                                   method='linear',
                                   bounds_error=False)
