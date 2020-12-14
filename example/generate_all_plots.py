import numpy as np
from WDLFBuilder import theoretical_lf

wdlf = theoretical_lf.WDLF()

wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0, 20., 0.5)
age = [10E9]
num = np.zeros((len(age), len(Mag)))

wdlf.compute_density(Mag=Mag, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_sfh(display=False)
wdlf.plot_imf(display=False)
wdlf.plot_ifmr(display=False)
wdlf.plot_wdlf(display=True)
