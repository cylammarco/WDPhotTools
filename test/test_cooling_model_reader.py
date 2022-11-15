#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test the cooling model reader"""

import pytest

from WDPhotTools.cooling_model_reader import CoolingModelReader


def test_cooling_model_dictionary():
    """Test initialising a cooling model reader"""
    cmr = CoolingModelReader()
    model_list = cmr.list_cooling_model()
    assert len(model_list) == 24


@pytest.mark.xfail(raises=ValueError)
def test_loading_unknown_cooling_model():
    """Test giving an unknown model name"""
    cmr = CoolingModelReader()
    cmr.get_cooling_model("blablabla")


@pytest.mark.xfail(raises=ValueError)
def test_loading_unknown_interpolator():
    """Test giving an unknown interpolator"""
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="linear")


def test_get_lpcode_co_da_10_z001():
    """test loading a model"""
    cmr = CoolingModelReader()
    cooling_model = cmr.get_cooling_model(model="lpcode_co_da_10_z001")
    assert len(cooling_model) == 4
    assert len(cooling_model[0]) == 10
    assert len(cooling_model[1]) == 10
    assert len(cooling_model[2]) == 13
    assert len(cooling_model[3]) == 13
    assert len(cooling_model[1][cooling_model[0] == 0.659][0]) == 746


def test_loading_default_ct_interpolator():
    """test loaded default models while using CT interpolator"""
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="CT")
    assert cmr.cooling_models["low_mass_cooling_model"] == "montreal_co_da_20"
    assert (
        cmr.cooling_models["intermediate_mass_cooling_model"]
        == "montreal_co_da_20"
    )
    assert cmr.cooling_models["high_mass_cooling_model"] == "montreal_co_da_20"


def test_loading_default_rbf_interpolator():
    """test loaded default models while using RBF interpolator"""
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="RBF")
    assert cmr.cooling_models["low_mass_cooling_model"] == "montreal_co_da_20"
    assert (
        cmr.cooling_models["intermediate_mass_cooling_model"]
        == "montreal_co_da_20"
    )
    assert cmr.cooling_models["high_mass_cooling_model"] == "montreal_co_da_20"
