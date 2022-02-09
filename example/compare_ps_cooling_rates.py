import os

import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import atmosphere_model_reader as amr
from WDPhotTools import theoretical_lf


try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except:
    HERE = os.path.dirname(os.path.realpath(__name__))


# Salaris' model with phase separation
wdlf = theoretical_lf.WDLF()
wdlf.set_low_mass_cooling_model("montreal_co_da_20")
wdlf.set_intermediate_mass_cooling_model("basti_co_da_10")
wdlf.set_intermediate_mass_cooling_model("basti_co_da_10")

# Salaris' model without phase separation
wdlf_nps = theoretical_lf.WDLF()
wdlf_nps.set_low_mass_cooling_model("montreal_co_da_20")
wdlf_nps.set_high_mass_cooling_model("basti_co_da_10_nps")
wdlf_nps.set_high_mass_cooling_model("basti_co_da_10_nps")

wdlf.compute_cooling_age_interpolator()
wdlf_nps.compute_cooling_age_interpolator()

atm = amr.atm_reader()

# Default passband is G3
G = atm.interp_atm()
BP = atm.interp_atm(dependent="G3_BP")
RP = atm.interp_atm(dependent="G3_RP")
m = atm.interp_atm(dependent="mass")

logg = np.arange(7.25, 8.75, 0.02)
Mbol = np.arange(6.0, 16.0, 0.02)
logL = np.log10(10.0 ** ((4.8 - Mbol) / 2.5) * 3.826e33)

fig = plt.figure(1, figsize=(12, 8))
fig.clf()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

_x = []
_y = []
_z1 = []
_z2 = []
for i in logg:
    logg_i = np.ones_like(Mbol) * i
    _m = m(logg_i, Mbol)
    _m[~np.isfinite(_m)] = 0.0
    m_valid = _m
    # format the data into matplotlib.tricontour readable format
    _x.append(BP(logg_i, Mbol) - RP(logg_i, Mbol))
    _y.append(G(logg_i, Mbol))
    _z1.append(wdlf.cooling_rate_interpolator(logL, m_valid))
    _z2.append(wdlf_nps.cooling_rate_interpolator(logL, m_valid))

x = np.concatenate(_x)
y = np.concatenate(_y)
z1 = np.log10(np.concatenate(_z1))
z2 = np.log10(np.concatenate(_z2))

# masking the non-finite values
mask1 = np.isfinite(z1)
mask2 = np.isfinite(z2)
cmap = "RdBu_r"
levels = 50

# make the contour lines
ax1.tricontour(
    x[mask1], y[mask1], z1[mask1], levels=levels, linewidths=0.5, colors="k"
)
ax2.tricontour(
    x[mask2], y[mask2], z1[mask2], levels=levels, linewidths=0.5, colors="k"
)
ax3.tricontour(
    x[mask1 & mask2],
    y[mask1 & mask2],
    (z1 / z2)[mask1 & mask2],
    levels=levels,
    linewidths=0.5,
    colors="k",
)
# make the contour shades
contour1 = ax1.tricontourf(
    x[mask1], y[mask1], z1[mask1], levels=levels, cmap=cmap
)
contour2 = ax2.tricontourf(
    x[mask2], y[mask2], z2[mask2], levels=levels, cmap=cmap
)
contour3 = ax3.tricontourf(
    x[mask1 & mask2],
    y[mask1 & mask2],
    (z1 / z2)[mask1 & mask2],
    levels=levels,
    cmap=cmap,
)

ax1.set_ylim(16.0, 10.0)
ax1.grid()
ax1.set_xlabel("(BP - RP) / mag")
ax1.set_ylabel("G / mag")
ax1.set_title("with PS")

ax2.set_ylim(16.0, 10.0)
ax2.grid()
ax2.set_xlabel("(BP - RP) / mag")
ax2.set_yticklabels("")
ax2.set_title("without PS (nPS)")

ax3.set_ylim(16.0, 10.0)
ax3.grid()
ax3.set_xlabel("(BP - RP) / mag")
ax3.set_yticklabels("")
ax3.set_title(r"$\Delta$(PS-nPS)")

plt.suptitle("log(dL/dt) contour plot")
plt.subplots_adjust(wspace=0.0, left=0.075, right=0.975)
plt.savefig(
    os.path.join(HERE, "example_output", "compare_ps_cooling_rates.png")
)
