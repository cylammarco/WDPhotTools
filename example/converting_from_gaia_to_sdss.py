import os

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import atmosphere_model_reader as amr

try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except:
    HERE = os.path.dirname(os.path.realpath(__name__))

data = fits.open(os.path.join(HERE, "GaiaEDR3_WD_SDSSspec.FITS"))[1].data
size = len(data)

# Get the Gaia photometry
G3 = data["phot_g_mean_mag"]
G3_BP = data["phot_bp_mean_mag"]
G3_RP = data["phot_rp_mean_mag"]
loggH = data["logg_H"]
TeffH = data["Teff_H"]
dist_mod = np.log10(1000.0 / data["parallax"]) * 5.0 - 5.0

atm = amr.atm_reader()

# Getting the interpolation function depending on G3 and logg
u_itp = atm.interp_atm(dependent="u_sdss", independent=["G3", "logg"])
g_itp = atm.interp_atm(dependent="g_sdss", independent=["G3", "logg"])
r_itp = atm.interp_atm(dependent="r_sdss", independent=["G3", "logg"])

# Getting the interpolation function depending on G3_BP and logg
u_itp_bp = atm.interp_atm(dependent="u_sdss", independent=["G3_BP", "logg"])
g_itp_bp = atm.interp_atm(dependent="g_sdss", independent=["G3_BP", "logg"])
r_itp_bp = atm.interp_atm(dependent="r_sdss", independent=["G3_BP", "logg"])

# Getting the interpolation function depending on G3_RP and logg
u_itp_rp = atm.interp_atm(dependent="u_sdss", independent=["G3_RP", "logg"])
g_itp_rp = atm.interp_atm(dependent="g_sdss", independent=["G3_RP", "logg"])
r_itp_rp = atm.interp_atm(dependent="r_sdss", independent=["G3_RP", "logg"])

# Getting the interpolation function depending on G3 and Teff
u_itp_teff = atm.interp_atm(dependent="u_sdss", independent=["G3", "Teff"])
g_itp_teff = atm.interp_atm(dependent="g_sdss", independent=["G3", "Teff"])
r_itp_teff = atm.interp_atm(dependent="r_sdss", independent=["G3", "Teff"])

# Getting the interpolation function depending on G3_BP and Teff
u_itp_bp_teff = atm.interp_atm(
    dependent="u_sdss", independent=["G3_BP", "Teff"]
)
g_itp_bp_teff = atm.interp_atm(
    dependent="g_sdss", independent=["G3_BP", "Teff"]
)
r_itp_bp_teff = atm.interp_atm(
    dependent="r_sdss", independent=["G3_BP", "Teff"]
)

# Getting the interpolation function depending on G3_RP and Teff
u_itp_rp_teff = atm.interp_atm(
    dependent="u_sdss", independent=["G3_RP", "Teff"]
)
g_itp_rp_teff = atm.interp_atm(
    dependent="g_sdss", independent=["G3_RP", "Teff"]
)
r_itp_rp_teff = atm.interp_atm(
    dependent="r_sdss", independent=["G3_RP", "Teff"]
)


# Getting the interpolation function depending on G3_RP and Teff
u_itp_logg_teff = atm.interp_atm(
    dependent="u_sdss", independent=["logg", "Teff"]
)
g_itp_logg_teff = atm.interp_atm(
    dependent="g_sdss", independent=["logg", "Teff"]
)
r_itp_logg_teff = atm.interp_atm(
    dependent="r_sdss", independent=["logg", "Teff"]
)


# converting into SDSS photometry
u_sdss = u_itp(G3 - dist_mod, loggH) + dist_mod
g_sdss = g_itp(G3 - dist_mod, loggH) + dist_mod
r_sdss = r_itp(G3 - dist_mod, loggH) + dist_mod

u_sdss_bp = u_itp(G3_BP - dist_mod, loggH) + dist_mod
g_sdss_bp = g_itp(G3_BP - dist_mod, loggH) + dist_mod
r_sdss_bp = r_itp(G3_BP - dist_mod, loggH) + dist_mod

u_sdss_rp = u_itp(G3_RP - dist_mod, loggH) + dist_mod
g_sdss_rp = g_itp(G3_RP - dist_mod, loggH) + dist_mod
r_sdss_rp = r_itp(G3_RP - dist_mod, loggH) + dist_mod

u_sdss_teff = u_itp_teff(G3 - dist_mod, TeffH) + dist_mod
g_sdss_teff = g_itp_teff(G3 - dist_mod, TeffH) + dist_mod
r_sdss_teff = r_itp_teff(G3 - dist_mod, TeffH) + dist_mod

u_sdss_bp_teff = u_itp_teff(G3_BP - dist_mod, TeffH) + dist_mod
g_sdss_bp_teff = g_itp_teff(G3_BP - dist_mod, TeffH) + dist_mod
r_sdss_bp_teff = r_itp_teff(G3_BP - dist_mod, TeffH) + dist_mod

u_sdss_rp_teff = u_itp_teff(G3_RP - dist_mod, TeffH) + dist_mod
g_sdss_rp_teff = g_itp_teff(G3_RP - dist_mod, TeffH) + dist_mod
r_sdss_rp_teff = r_itp_teff(G3_RP - dist_mod, TeffH) + dist_mod

u_model = u_itp_logg_teff(loggH, TeffH) + dist_mod
g_model = g_itp_logg_teff(loggH, TeffH) + dist_mod
r_model = r_itp_logg_teff(loggH, TeffH) + dist_mod

# Plot the (u-g) vs (g-r) colour-colour diagram
# Top row is based on interpolation of (logg, Gaia filter)
# Bottom row is based on interpolation of (Teff, Gaia filter)
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    2, 3, sharex=True, figsize=(15, 15)
)
ax1.scatter(u_sdss - g_sdss, g_sdss - r_sdss, s=1, c=loggH)
ax2.scatter(u_sdss_bp - g_sdss_bp, g_sdss_bp - r_sdss_bp, s=1, c=loggH)
ax3.scatter(u_sdss_rp - g_sdss_rp, g_sdss_rp - r_sdss_rp, s=1, c=loggH)
ax4.scatter(
    u_sdss_teff - g_sdss_teff,
    g_sdss_teff - r_sdss_teff,
    s=1,
    c=loggH,
    cmap="hot",
)
ax5.scatter(
    u_sdss_bp_teff - g_sdss_bp_teff,
    g_sdss_bp_teff - r_sdss_bp_teff,
    s=1,
    c=loggH,
    cmap="hot",
)
ax6.scatter(
    u_sdss_rp_teff - g_sdss_rp_teff,
    g_sdss_rp_teff - r_sdss_rp_teff,
    s=1,
    c=loggH,
    cmap="hot",
)

ax1.set_xlim(-0.55, 2.55)
ax2.set_xlim(-0.55, 2.55)
ax3.set_xlim(-0.55, 2.55)
ax4.set_xlim(-0.55, 2.55)
ax5.set_xlim(-0.55, 2.55)
ax6.set_xlim(-0.55, 2.55)

ax1.set_ylim(-0.6, 1.25)
ax2.set_ylim(-0.6, 1.25)
ax3.set_ylim(-0.6, 1.25)
ax4.set_ylim(-0.6, 1.25)
ax5.set_ylim(-0.6, 1.25)
ax6.set_ylim(-0.6, 1.25)

ax2.set_yticklabels([""])
ax3.set_yticklabels([""])
ax5.set_yticklabels([""])
ax6.set_yticklabels([""])

ax4.set_xlabel("g - r / mag")
ax5.set_xlabel("g - r / mag")
ax6.set_xlabel("g - r / mag")

ax1.set_ylabel("u - g / mag")
ax4.set_ylabel("u - g / mag")

ax1.set_title(r"$\{G3, \log(g)\} \rightarrow \{u, g, r\}$")
ax2.set_title(r"$\{G3_{\mathrm{BP}}, \log(g)\} \rightarrow \{u, g, r\}$")
ax3.set_title(r"$\{G3_{\mathrm{RP}}, \log(g)\} \rightarrow \{u, g, r\}$")

ax4.set_title(r"$\{G3, T_{\mathrm{eff}}\} \rightarrow \{u, g, r\}$")
ax5.set_title(
    r"$\{G3_{\mathrm{BP}}, T_{\mathrm{eff}}\} \rightarrow \{u, g, r\}$"
)
ax6.set_title(
    r"$\{G3_{\mathrm{RP}}, T_{\mathrm{eff}}\} \rightarrow \{u, g, r\}$"
)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

plt.subplots_adjust(
    top=0.95, bottom=0.075, left=0.08, right=0.975, wspace=0.0, hspace=0.1
)


# Plot the residual of the catalogue value to converted values
# Top row is the residual in u from the interpolation of G_BP, G & G_RP
# Middle row is the residual in g from the interpolation of G_BP, G & G_RP
# Bottom row is the residual in r from the interpolation of G_BP, G & G_RP
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
    3, 3, sharex=True, figsize=(15, 15)
)

ax1.scatter(TeffH, data["umag"] - u_sdss_bp_teff, s=1)
ax2.scatter(TeffH, data["umag"] - u_sdss_teff, s=1)
ax3.scatter(TeffH, data["umag"] - u_sdss_rp_teff, s=1)

ax4.scatter(TeffH, data["gmag"] - g_sdss_bp_teff, s=1)
ax5.scatter(TeffH, data["gmag"] - g_sdss_teff, s=1)
ax6.scatter(TeffH, data["gmag"] - g_sdss_rp_teff, s=1)

ax7.scatter(TeffH, data["rmag"] - r_sdss_bp_teff, s=1)
ax8.scatter(TeffH, data["rmag"] - r_sdss_teff, s=1)
ax9.scatter(TeffH, data["rmag"] - r_sdss_rp_teff, s=1)

ax1.set_title(
    r"$u_{\mathrm{catalogue}} - u(T_{\mathrm{eff}}, G_{\mathrm{BP}})$"
)
ax2.set_title(r"$u_{\mathrm{catalogue}} - u(T_{\mathrm{eff}}, G)$")
ax3.set_title(
    r"$u_{\mathrm{catalogue}} - u(T_{\mathrm{eff}}, G_{\mathrm{RP}})$"
)

ax4.set_title(
    r"$g_{\mathrm{catalogue}} - g(T_{\mathrm{eff}}, G_{\mathrm{BP}})$"
)
ax5.set_title(r"$g_{\mathrm{catalogue}} - g(T_{\mathrm{eff}}, G)$")
ax6.set_title(
    r"$g_{\mathrm{catalogue}} - g(T_{\mathrm{eff}}, G_{\mathrm{RP}})$"
)

ax7.set_title(
    r"$r_{\mathrm{catalogue}} - r(T_{\mathrm{eff}}, G_{\mathrm{BP}})$"
)
ax8.set_title(r"$r_{\mathrm{catalogue}} - r(T_{\mathrm{eff}}, G)$")
ax9.set_title(
    r"$r_{\mathrm{catalogue}} - r(T_{\mathrm{eff}}, G_{\mathrm{RP}})$"
)

ax2.set_yticklabels([""])
ax3.set_yticklabels([""])
ax5.set_yticklabels([""])
ax6.set_yticklabels([""])
ax8.set_yticklabels([""])
ax9.set_yticklabels([""])

ax7.set_xlabel("Temperature / K")
ax8.set_xlabel("Temperature / K")
ax9.set_xlabel("Temperature / K")

ax1.set_ylabel(r"$\Delta(u)$ / mag")
ax4.set_ylabel(r"$\Delta(g)$ / mag")
ax7.set_ylabel(r"$\Delta(r)$ / mag")

ax1.set_xlim(4000, 110000)
ax1.set_ylim(-0.85, 0.85)
ax2.set_ylim(-0.85, 0.85)
ax3.set_ylim(-0.85, 0.85)
ax4.set_ylim(-0.85, 0.85)
ax5.set_ylim(-0.85, 0.85)
ax6.set_ylim(-0.85, 0.85)
ax7.set_ylim(-0.85, 0.85)
ax8.set_ylim(-0.85, 0.85)
ax9.set_ylim(-0.85, 0.85)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
ax9.grid()

ax1.set_xscale("log")

plt.subplots_adjust(
    top=0.95, bottom=0.07, left=0.08, right=0.975, wspace=0.0, hspace=0.15
)


# Plot the residual of the catalogue value to derived values from (logg, Teff)
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, sharex=True, sharey=True, figsize=(15, 10)
)

ax1.scatter(TeffH, data["umag"] - u_model, s=1)
ax2.scatter(TeffH, data["gmag"] - g_model, s=1)
ax3.scatter(TeffH, data["rmag"] - r_model, s=1)

ax1.set_ylabel(r"$\Delta$ / mag")

ax1.set_xlabel("Temperature / K")
ax2.set_xlabel("Temperature / K")
ax3.set_xlabel("Temperature / K")

ax1.set_xlim(4000, 110000)
ax1.set_ylim(-0.85, 0.85)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_title(r"$u_{\mathrm{catalogue}} - u(\log(g), T_{\mathrm{eff}})$")
ax2.set_title(r"$g_{\mathrm{catalogue}} - g(\log(g), T_{\mathrm{eff}})$")
ax3.set_title(r"$r_{\mathrm{catalogue}} - r(\log(g), T_{\mathrm{eff}})$")

ax1.set_xscale("log")

plt.subplots_adjust(
    top=0.95, bottom=0.075, left=0.075, right=0.975, wspace=0.0, hspace=0.15
)
