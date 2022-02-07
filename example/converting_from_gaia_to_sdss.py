import os

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import atmosphere_model_reader as amr

HERE = os.path.dirname(os.path.realpath(__file__))

data = fits.open('GaiaEDR3_WD_SDSSspec.FITS')[1].data
size = len(data)

atm = amr.atm_reader()

loggH = data['logg_H']
dist_mod = np.log10(1000. / data['parallax']) * 5. - 5.

u_itp = atm.interp_atm(dependent='u_sdss', independent=['G3', 'logg'])
g_itp = atm.interp_atm(dependent='g_sdss', independent=['G3', 'logg'])
r_itp = atm.interp_atm(dependent='r_sdss', independent=['G3', 'logg'])
i_itp = atm.interp_atm(dependent='i_sdss', independent=['G3', 'logg'])
z_itp = atm.interp_atm(dependent='z_sdss', independent=['G3', 'logg'])

u_sdss = u_itp(data['phot_g_mean_mag'] - dist_mod, loggH)
g_sdss = g_itp(data['phot_g_mean_mag'] - dist_mod, loggH)
r_sdss = r_itp(data['phot_g_mean_mag'] - dist_mod, loggH)
i_sdss = i_itp(data['phot_g_mean_mag'] - dist_mod, loggH)
z_sdss = z_itp(data['phot_g_mean_mag'] - dist_mod, loggH)
