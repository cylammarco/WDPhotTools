import numpy as np
from matplotlib import pyplot as plt

from theoretical_lf import WDLF

wdlf = WDLF()

wdlf.set_ifmr_model('C08')
wdlf.set_sfr_model(mode='burst', duration=1e9)
wdlf.compute_cooling_age_interpolator()


logL = np.arange(28, 34., 0.05)
age = 1E9 * np.arange(8, 15, 1)
num = np.zeros((len(age), len(logL)))

wdlf.compute_density(logL=logL, T0=age, save_csv=True)

wdlf.plot_wdlf(display=False, savefig=True)


wdlf.set_sfr_model(mode='decay')
wdlf.compute_cooling_age_interpolator()


logL = np.arange(28, 34., 0.05)
age = 1E9 * np.arange(8, 15, 1)
num = np.zeros((len(age), len(logL)))

wdlf.compute_density(logL=logL, T0=age, save_csv=True)

wdlf.plot_wdlf(display=True, savefig=True)

