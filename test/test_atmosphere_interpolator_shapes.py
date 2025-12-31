#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


def test_interp_am_invalid_interpolator():
    atm = AtmosphereModelReader()
    with pytest.raises(ValueError):
        atm.interp_am(dependent="G3", atmosphere="H", independent=["Mbol"], interpolator="bad")


def test_interp_am_rbf_1d_scalar_and_array():
    atm = AtmosphereModelReader()
    itp = atm.interp_am(dependent="G3", atmosphere="H", independent=["Mbol"], logg=8.0, interpolator="RBF")
    # scalar input
    val_scalar = itp(10.0)
    # vector input
    val_array = itp(np.array([10.0, 12.0, 14.0]))
    assert np.isscalar(np.asarray(val_scalar).item())
    assert np.asarray(val_array).shape == (3,)


def test_interp_am_rbf_2d_broadcast_lengths():
    atm = AtmosphereModelReader()
    itp = atm.interp_am(dependent="G3", atmosphere="H", independent=["logg", "Mbol"], interpolator="RBF")
    logg = [8.0]
    mbol = [10.0, 12.0, 14.0]
    out = itp(logg, mbol)
    assert np.asarray(out).shape == (3,)
