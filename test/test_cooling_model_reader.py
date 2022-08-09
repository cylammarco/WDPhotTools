import numpy as np
import pytest

from WDPhotTools.cooling_model_reader import CoolingModelReader


def test_cooling_model_dictionary():
    cmr = CoolingModelReader()
    model_list = cmr.list_cooling_model()
    assert len(model_list) == 24


@pytest.mark.xfail(raises=ValueError)
def test_loading_unknown_interpolator():
    cmr = CoolingModelReader()
    cmr.get_cooling_model("blablabla")


@pytest.mark.xfail(raises=ValueError)
def test_loading_unknown_interpolator():
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="linear")


def test_get_lpcode_co_da_10_z001():
    cmr = CoolingModelReader()
    cooling_model = cmr.get_cooling_model(model="lpcode_co_da_10_z001")
    assert len(cooling_model) == 4
    assert len(cooling_model[0]) == 10
    assert len(cooling_model[1]) == 10
    assert len(cooling_model[2]) == 13
    assert len(cooling_model[3]) == 13
    assert len(cooling_model[1][cooling_model[0] == 0.659][0]) == 746


def test_loading_default_CT_interpolator():
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="CT")
    assert cmr.cooling_models["low_mass_cooling_model"] == "montreal_co_da_20"
    assert (
        cmr.cooling_models["intermediate_mass_cooling_model"]
        == "montreal_co_da_20"
    )
    assert cmr.cooling_models["high_mass_cooling_model"] == "montreal_co_da_20"


def test_loading_default_RBF_interpolator():
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(interpolator="RBF")
    assert cmr.cooling_models["low_mass_cooling_model"] == "montreal_co_da_20"
    assert (
        cmr.cooling_models["intermediate_mass_cooling_model"]
        == "montreal_co_da_20"
    )
    assert cmr.cooling_models["high_mass_cooling_model"] == "montreal_co_da_20"
