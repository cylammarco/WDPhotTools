import numpy as np
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0, 20., 0.1)
age = [3.0E9]
num = np.zeros((len(age), len(Mag)))

wdlf.set_sfr_model(mode='burst', age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf.compute_density(Mag=Mag)

wdlf.plot_cooling_model(display=False)
fig_sfh = wdlf.plot_sfh(display=False)
fig_imf = wdlf.plot_imf(display=False)
fig_ifmr = wdlf.plot_ifmr(display=False)
fig_wdlf = wdlf.plot_wdlf(display=True)
