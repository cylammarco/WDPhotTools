#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test initialising model readers"""

import numpy as np

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from WDPhotTools.cooling_model_reader import CoolingModelReader


amr = AtmosphereModelReader()

Mbol_RBF_logg_Teff = amr.interp_am(
    dependent="Mbol", independent=["logg", "Teff"], interpolator="RBF"
)
Mbol_CT_logg_Teff = amr.interp_am(
    dependent="Mbol", independent=["logg", "Teff"], interpolator="CT"
)


G3_RBF_logg_Mbol = amr.interp_am(
    dependent="G3", independent=["logg", "Mbol"], interpolator="RBF"
)
G3_CT_logg_Mbol = amr.interp_am(
    dependent="G3", independent=["logg", "Mbol"], interpolator="CT"
)


Teff_range = 10.0 ** np.arange(np.log10(1500), np.log10(150000), 0.01)
Mbol_range = np.arange(5, 15, 0.1)


def test_compare_amr_ct_and_rbf_over_logg_teff():
    """Test that the CT and RBF interpolator are returning values within 1%
    interpolated over (logg, Teff)"""
    for logg in np.arange(7.5, 9.0, 0.1):
        assert np.isclose(
            Mbol_RBF_logg_Teff(logg, Teff_range),
            Mbol_CT_logg_Teff(logg, Teff_range),
            rtol=1e-2,
            atol=1e-2,
        ).all()


def test_compare_amr_ct_and_rbf_over_logg_mbol():
    """Test that the CT and RBF interpolator are returning values within 1%
    interpolated over (logg, Mbol)"""
    for logg in np.arange(7.5, 9.0, 0.1):
        assert np.isclose(
            G3_RBF_logg_Mbol(logg, Mbol_range),
            G3_CT_logg_Mbol(logg, Mbol_range),
            rtol=1e-2,
            atol=1e-2,
        ).all()


cmr_ct = CoolingModelReader()
cmr_rbf = CoolingModelReader()

cmr_ct.compute_cooling_age_interpolator(interpolator="CT")
cmr_rbf.compute_cooling_age_interpolator(interpolator="RBF")

logl_range = np.arange(28.0, 33.0, 0.1)
mass_range = np.arange(0.45, 1.2, 0.01)


def test_compare_cmr_ct_and_rbf():
    """Test that the CT and RBF interpolator are returning values within 1%"""
    for logl in logl_range:
        assert np.isclose(
            cmr_ct.cooling_interpolator(logl, mass_range),
            cmr_rbf.cooling_interpolator(logl, mass_range),
            rtol=1e-2,
            atol=1e-2,
        ).all()
