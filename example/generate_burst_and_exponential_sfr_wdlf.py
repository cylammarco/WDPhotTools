import numpy as np
from matplotlib import pyplot as plt
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

wdlf.set_ifmr_model('C08')

wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0, 20., 0.2)

age_list = 1E9 * np.arange(8, 15, 1)

fig1 = plt.figure(1, figsize=(12, 8))
ax1 = plt.gca()

fig2 = plt.figure(2, figsize=(12, 8))
ax2 = plt.gca()

for i, age in enumerate(age_list):

    # Burst SFR
    wdlf.set_sfr_model(mode='burst', age=age, duration=1e9)
    _, burst_density = wdlf.compute_density(Mag=Mag,
                                            passband='G3',
                                            save_csv=True)
    ax1.plot(Mag,
             np.log10(burst_density),
             label="{0:.2f} Gyr".format(age / 1e9))

    # Exponential decay SFR
    wdlf.set_sfr_model(mode='decay', age=age)
    wdlf.compute_cooling_age_interpolator()

    _, decay_density = wdlf.compute_density(Mag=Mag,
                                            passband='G3',
                                            save_csv=True)
    ax2.plot(Mag,
             np.log10(decay_density),
             label="{0:.2f} Gyr".format(age / 1e9))

ax1.legend()
ax1.grid()
ax1.set_xlabel(r'G$_{DR3}$ / mag')
ax1.set_ylabel('log(arbitrary number density)')
ax1.set_xlim(7.5, 20)
ax1.set_ylim(-5, 0)
ax1.set_title('Star Formation History: 1 Gyr Burst')
fig1.savefig(
    'burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png')

ax2.legend()
ax2.grid()
ax2.set_xlabel(r'G$_{DR3}$ / mag')
ax2.set_ylabel('log(arbitrary number density)')
ax2.set_xlim(7.5, 20)
ax2.set_ylim(-5, 0)
ax2.set_title('Star Formation History: Exponential Decay')
fig2.savefig(
    'decay_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.png')
