#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing the computation of theoretical luminosity functions"""

import os  # For testing file existence with assert
from unittest.mock import patch

from matplotlib import pyplot as plt
import numpy as np
import pytest

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

mag = np.arange(4.0, 18.0, 4.0)
age = [3.0e9, 12.0e9]
num = np.zeros((len(age), len(mag)))


def test_default():
    """Test default setup"""
    wdlf.compute_density(mag=mag)


@pytest.mark.xfail
def test_changing_sfr_mode_to_unknown_name():
    """Test error trapping when changing sfr model to unknown"""
    wdlf.set_sfr_model("blabla")


@pytest.mark.xfail
def test_changing_imf_model_to_unknown_name():
    """Test error trapping when changing imf model to unknown"""
    wdlf.set_imf_model("blabla")


@pytest.mark.xfail
def test_changing_ms_model_to_unknown_name():
    """Test error trapping when changing ms model to unknown"""
    wdlf.set_ms_model("blabla")


@pytest.mark.xfail
def test_changing_ifmr_model_to_unknown_name():
    """Test error trapping when changing ifmr model to unknown"""
    wdlf.set_ifmr_model("blabla")


@pytest.mark.xfail
def test_changing_empty_low_mass_model_to_unknown_name():
    """Test error trapping when changing low mass cooling model to unknown"""
    wdlf.set_low_mass_cooling_model("blabla")


@pytest.mark.xfail
def test_changing_empty_intermediate_mass_model_to_unknown_name():
    """Test error trapping when changing intermediate mass cooling model
    to unknown"""
    wdlf.set_intermediate_mass_cooling_model("blabla")


@pytest.mark.xfail
def test_changing_empty_high_mass_model_to_unknown_name():
    """Test error trapping when changing high mass cooling model to unknown"""
    wdlf.set_high_mass_cooling_model("blabla")


wdlf.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf.compute_density(mag=mag)


@patch("matplotlib.pyplot.show")
def test_plotting(mock_show):
    """Test plotting to display and exporting"""
    wdlf.plot_input_models(
        display=True,
        savefig=True,
        folder="test_output",
        filename="test_input_model",
        ext="png",
    )
    wdlf.plot_input_models(
        sfh_log=True,
        imf_log=False,
        ms_time_log=False,
        cooling_model_use_mag=False,
        title="swappped linear and log axes",
        display=True,
        savefig=True,
        folder="test_output",
        ext="png",
    )
    wdlf.plot_wdlf(
        display=True,
        savefig=True,
        folder="test_output",
        filename="test_plot_wdlf",
        ext="png",
    )


mock_fig = plt.figure(1)


@patch("matplotlib.pyplot.show")
def test_plotting_to_an_external_figure_object(mock_fig):
    """Test plotting to a manually provided Figure object"""
    wdlf.plot_wdlf(
        fig=mock_fig,
        display=True,
        savefig=True,
        folder="test_output",
        filename="test_plot_wdlf_external_figure_object",
        ext=["png", "pdf"],
    )


def test_changing_sfr_model():
    """Test changing sfr models and then recompute the density"""
    wdlf.set_sfr_model(mode="constant", age=age[0])
    wdlf.compute_density(mag=mag, save_csv=True, folder="test_output")
    wdlf.set_sfr_model(mode="decay", age=age[0])
    wdlf.compute_density(
        mag=mag,
        save_csv=True,
        normed=False,
        folder="test_output",
        filename="test_saving_wdlf_csv",
    )
    assert np.sum(wdlf.number_density) != 1.0


def test_ct_interpolator():
    """Test changing the interpolator and then recompute the density"""
    wdlf.compute_density(mag=mag, interpolator="CT")
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_rbf_interpolator():
    """Test changing the interpolator and then recompute the density"""
    wdlf.compute_density(mag=mag, interpolator="RBF")
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_imf_model():
    """Test changing the imf model and then recompute the density"""
    wdlf.set_imf_model("K01")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_imf_model("C03")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_imf_model("C03b")
    wdlf.compute_density(mag=mag)
    wdlf.set_sfr_model(mode="decay", age=age[1])
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_ms_model():
    """Test changing the ms model and then recompute the density"""
    wdlf.set_ms_model("PARSECz00001")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz00002")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz00005")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0001")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0002")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0004")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0006")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0008")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz001")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0014")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz0017")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz002")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz003")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz004")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("PARSECz006")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("GENEVAz002")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("GENEVAz006")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("GENEVAz014")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem400")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem400")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem350")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem300")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem250")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem200")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem175")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem150")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem125")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem100")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem075")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem050")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFem025")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFe000")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFe025")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("MISTFe050")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_ifmr_model():
    """Test changing the ifmr model and then recompute the density"""
    wdlf.set_ifmr_model("C08")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("C08b")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("S09")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("S09b")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("W09")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("K09")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("K09b")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("C18")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    # Testing set_ifmr_model with model = "EB18"
    wdlf.set_ifmr_model("EB18")
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_imf_k01_small_mass_log():
    """Testing set_imf_model with model = "K01" and plot the imf in
    logarithmic space"""
    wdlf.set_imf_model("K01")
    wdlf.plot_input_models(display=False, imf_log=True)


def test_plotting_wdlf_log_false_folder_none():
    """Testing plot_wdlf with log=False and savefig with folder = None"""
    wdlf.plot_wdlf(
        log=False,
        display=False,
        savefig=True,
        folder=None,
        filename="test_plot_wdlf",
        ext="png",
    )
    # assert the file exists at where you intend to
    _folder = os.getcwd()
    for _e in ["png"]:
        _filename = "test_plot_wdlf" + "." + _e
        assert os.path.isfile(os.path.join(_folder, _filename))
    os.remove(os.path.join(_folder, _filename))


def manual_fn(some_input):
    """manual function for 'manual' tests"""
    # return 1.0 regardless
    return some_input**0.0


def test_manual_imf_model():
    """Testing set_imf_model with model = 'manual'"""
    wdlf.set_imf_model(model="manual", imf_function=manual_fn)
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_manual_ms_model():
    """Testing set_ms_model with model = 'manual'"""
    wdlf.set_ms_model(model="manual", ms_function=manual_fn)
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_manual_sfr_model():
    """Testing set_sfr_model with model = 'manual'"""
    wdlf.set_sfr_model(mode="manual", sfr_model=manual_fn)
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_manual_ifmr_model():
    """Testing set_ifmr_model with model = 'manual'"""
    wdlf.set_ifmr_model(model="manual", ifmr_function=manual_fn)
    wdlf.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_compute_density_savefig_folder_none():
    """Testing set_imf_model with model = 'manual' and
    save_csv with folder = None"""
    wdlf.set_ifmr_model(model="manual", ifmr_function=manual_fn)
    wdlf.compute_density(save_csv=True, folder=None, mag=mag)
    # assert the file exists at where you intend to
    _folder = os.getcwd()
    _filename = (
        f"{wdlf.t_start / 1e9:.2f}Gyr_"
        + wdlf.wdlf_params["sfr_mode"]
        + "_"
        + wdlf.wdlf_params["ms_model"]
        + "_"
        + wdlf.wdlf_params["ifmr_model"]
        + "_"
        + wdlf.cooling_models["low_mass_cooling_model"]
        + "_"
        + wdlf.cooling_models["intermediate_mass_cooling_model"]
        + "_"
        + wdlf.cooling_models["high_mass_cooling_model"]
        + ".csv"
    )
    assert os.path.isfile(os.path.join(_folder, _filename))
    os.remove(
        "10.00Gyr_manual_manual_manual_montreal_"
        "co_da_20_montreal_co_da_20_montreal_co_da_20.csv"
    )


def test_plotting_wdlf_savefig_path_not_exist():
    """Testing plot_wdlf with filename = None"""
    _folder = "test_output/test_plotting_wdlf_savefig_path_not_exist"
    wdlf.plot_wdlf(
        log=False,
        display=False,
        savefig=True,
        folder=_folder,
        filename=None,
        ext="png",
    )
    # assert the file exists at where you intend to
    for _e in ["png"]:
        _filename = (
            f"{wdlf.t_start / 1e9:.2f}Gyr_"
            + wdlf.wdlf_params["sfr_mode"]
            + "_"
            + wdlf.wdlf_params["ms_model"]
            + "_"
            + wdlf.wdlf_params["ifmr_model"]
            + "_"
            + wdlf.cooling_models["low_mass_cooling_model"]
            + "_"
            + wdlf.cooling_models["intermediate_mass_cooling_model"]
            + "_"
            + wdlf.cooling_models["high_mass_cooling_model"]
            + "."
            + _e
        )
        assert os.path.isfile(os.path.join(_folder, _filename))


def test_cooling_model_low_mass_lpcode_he_da_09():
    """Testing set_low_mass_cooling_model with model = lpcode_he_da_09"""
    wdlf1 = theoretical_lf.WDLF()
    wdlf1.set_low_mass_cooling_model(model="lpcode_he_da_09")
    wdlf1.compute_cooling_age_interpolator()
    wdlf1.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf1.number_density), 1.0)


def test_cooling_model_intermediate_mass_basti_co_da_10():
    """
    Testing set_intermediate_mass_cooling_model with model = basti_co_da_10
    """
    wdlf2 = theoretical_lf.WDLF()
    wdlf2.set_intermediate_mass_cooling_model(model="basti_co_da_10")
    wdlf2.compute_cooling_age_interpolator()
    wdlf2.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf2.number_density), 1.0)


def test_cooling_model_high_mass_basti_co_da_10():
    """Testing set_high_mass_cooling_model with model = basti_co_da_10"""
    wdlf3 = theoretical_lf.WDLF()
    wdlf3.set_high_mass_cooling_model(model="basti_co_da_10")
    wdlf3.compute_cooling_age_interpolator()
    wdlf3.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf3.number_density), 1.0)


def test_cooling_model_intermediate_mass_lpcode_co_db_17():
    """
    Testing set_intermediate_mass_cooling_model with model = lpcode_co_db_17
    """
    wdlf4 = theoretical_lf.WDLF()
    wdlf4.set_intermediate_mass_cooling_model(model="lpcode_co_db_17")
    wdlf4.compute_cooling_age_interpolator()
    wdlf4.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf4.number_density), 1.0)


def test_cooling_model_intermediate_mass_lpcode_co_db_17_z0001():
    """
    Testing set_intermediate_mass_cooling_model with model =
    lpcode_co_db_17_z0001
    """
    wdlf5 = theoretical_lf.WDLF()
    wdlf5.set_intermediate_mass_cooling_model(model="lpcode_co_db_17_z0001")
    wdlf5.compute_cooling_age_interpolator()
    wdlf5.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf5.number_density), 1.0)


def test_cooling_model_low_mass_lpcode_co_db_17_z0001():
    """
    Testing set_low_mass_cooling_model with model = lpcode_co_db_17_z0001
    """
    wdlf6 = theoretical_lf.WDLF()
    wdlf6.set_intermediate_mass_cooling_model(model="lpcode_co_db_17_z0001")
    wdlf6.compute_cooling_age_interpolator()
    wdlf6.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf6.number_density), 1.0)


def test_cooling_model_intermediate_mass_lpcode_da_22():
    """
    Testing set_low_mass_cooling_model with model = None
    Testing set_intermediate_mass_cooling_model with model = lpcode_da_22
    Testing set_high_mass_cooling_model with model = lpcode_da_22
    """
    wdlf7 = theoretical_lf.WDLF()
    wdlf7.set_low_mass_cooling_model(model=None)
    wdlf7.set_intermediate_mass_cooling_model(model="lpcode_da_22")
    wdlf7.set_high_mass_cooling_model(model="lpcode_da_22")
    wdlf7.compute_cooling_age_interpolator()
    wdlf7.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf7.number_density), 1.0)


def test_cooling_model_intermediate_mass_lpcode_db_22():
    """
    Testing set_low_mass_cooling_model with model = None
    Testing set_intermediate_mass_cooling_model with model = lpcode_db_22
    Testing set_high_mass_cooling_model with model = lpcode_db_22
    """
    wdlf8 = theoretical_lf.WDLF()
    wdlf8.set_low_mass_cooling_model(model=None)
    wdlf8.set_intermediate_mass_cooling_model(model="lpcode_db_22")
    wdlf8.set_high_mass_cooling_model(model="lpcode_db_22")
    wdlf8.compute_cooling_age_interpolator()
    wdlf8.compute_density(mag=mag)
    assert np.isclose(np.sum(wdlf8.number_density), 1.0)
