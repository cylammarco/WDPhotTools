import os

import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import theoretical_lf

HERE = os.path.dirname(os.path.realpath(__file__))

wdlf = theoretical_lf.WDLF()
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(4, 20.0, 0.2)
age_list = 1e9 * np.arange(8, 15, 1)

fig1 = plt.figure(2, figsize=(12, 8))
ax1 = plt.gca()

for i, age in enumerate(age_list):
    # Constant SFR
    wdlf.set_sfr_model(age=age)
    _, constant_density = wdlf.compute_density(
        Mag=Mag, save_csv=True, folder=os.path.join(HERE, "example_output")
    )
    ax1.plot(
        Mag, np.log10(constant_density), label="{0:.2f} Gyr".format(age / 1e9)
    )

ax1.legend()
ax1.grid()
ax1.set_xlabel(r"M$_{bol}$ / mag")
ax1.set_ylabel("log(arbitrary number density)")
ax1.set_xlim(5, 20)
ax1.set_ylim(-5, 0)
ax1.set_title("Star Formation History: Constant")
fig1.savefig(
    os.path.join(
        HERE,
        "example_output",
        "constant_C16_C08_montreal_co_da_20_"
        "montreal_co_da_20_montreal_co_da_20.png",
    )
)
