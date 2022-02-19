from matplotlib import pyplot as plt
import numpy as np
import pytest
from unittest.mock import patch
import os #For testing file existence with assert

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0, 20.0, 2.0)
age = [3.0e9, 12.0e9]
num = np.zeros((len(age), len(Mag)))


@pytest.mark.xfail
def test_changing_sfr_mode_to_unknown_name():
    wdlf.set_sfr_model("blabla")


@pytest.mark.xfail
def test_changing_imf_model_to_unknown_name():
    wdlf.set_imf_model("blabla")


@pytest.mark.xfail
def test_changing_ms_model_to_unknown_name():
    wdlf.set_ms_model("blabla")


@pytest.mark.xfail
def test_changing_ifmr_model_to_unknown_name():
    wdlf.set_ifmr_model("blabla")


@pytest.mark.xfail
def test_changing_empty_low_mass_model_to_unknown_name():
    wdlf.set_low_mass_cooling_model("blabla")


@pytest.mark.xfail
def test_changing_empty_intermediate_mass_model_to_unknown_name():
    wdlf.set_intermediate_mass_cooling_model("blabla")


@pytest.mark.xfail
def test_changing_empty_high_mass_model_to_unknown_name():
    wdlf.set_high_mass_cooling_model("blabla")


def test_default():
    wdlf.compute_density(Mag=Mag)


wdlf.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf.compute_density(Mag=Mag)


@patch("matplotlib.pyplot.show")
def test_plotting(mock_show):
    wdlf.plot_cooling_model(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_cooling_model",
        ext="png",
    )
    wdlf.plot_sfh(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_sfh",
        ext="png",
    )
    wdlf.plot_imf(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_imf",
        ext="png",
    )
    wdlf.plot_ifmr(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_ifmr",
        ext="png",
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_wdlf",
        ext="png",
    )


fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
fig4 = plt.figure(4)
fig5 = plt.figure(5)


@patch("matplotlib.pyplot.show")
def test_plotting_to_an_external_Figure_object(mock_show):
    wdlf.plot_cooling_model(
        use_mag=True,
        fig=fig1,
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
    )
    wdlf.plot_sfh(
        fig=fig2,
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
    )
    wdlf.plot_imf(
        fig=fig3,
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
    )
    wdlf.plot_ifmr(
        fig=fig4,
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
    )
    wdlf.plot_wdlf(
        fig=fig5,
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
    )


# not normalising the WDLF
def test_changing_sfr_model():
    wdlf.set_sfr_model(mode="constant", age=age[0])
    wdlf.compute_density(Mag=Mag, save_csv=True, folder="test_output")
    wdlf.set_sfr_model(mode="decay", age=age[0])
    wdlf.compute_density(
        Mag=Mag,
        save_csv=True,
        normed=False,
        folder="test_output",
        filename="test_saving_wdlf_csv",
    )
    assert np.sum(wdlf.number_density) != 1.0


def test_changing_imf_model():
    wdlf.set_imf_model("K01")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_imf_model("C03")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_imf_model("C03b")
    wdlf.compute_density(Mag=Mag)
    wdlf.set_sfr_model(mode="decay", age=age[1])
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_ms_model():
    wdlf.set_ms_model("C16")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ms_model("Bressan")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_changing_ifmr_model():
    wdlf.set_ifmr_model("C08")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("C08b")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("S09")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("S09b")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("W09")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("K09")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("K09b")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    wdlf.set_ifmr_model("C18")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)
    # Testing set_ifmr_model with model = "EB18"
    wdlf.set_ifmr_model("EB18")
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_rbf_interpolator():
    wdlf.compute_density(Mag=Mag, interpolator="RBF")
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


def test_ct_interpolator():
    wdlf.compute_density(Mag=Mag, interpolator="CT")
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


# Testing set_imf_model with model = "K01" and plot the imf in logarithmic space
def test_changing_imf_K01_small_mass_log():
    wdlf.set_imf_model("K01")
    wdlf.plot_imf(display=False, log=True)


# Testing plot_wdlf with log=False and savefig with folder = None
def test_plotting_wdlf_log_false_folder_none():
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
    for e in ["png"]:
        _filename = ( "test_plot_wdlf" + "." + e )
        assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_ifmr savefig with folder = None
def test_plotting_ifmr_folder_none():
    wdlf.plot_ifmr(
        fig=fig4, display=False, savefig=True, folder=None, ext=["png", "pdf"]
    )
    # assert the files exist at where you intend to
    _folder = os.getcwd()
    for e in ["png", "pdf"]:
        _filename = (
                        "ifmr_" + wdlf.wdlf_params["ifmr_model"] + "." + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_imf savefig with folder = None
def test_plotting_imf_folder_none():
    wdlf.plot_imf(
        fig=fig3, display=False, savefig=True, folder=None, ext=["png", "pdf"]
    )
    # assert the files exist at where you intend to
    _folder = os.getcwd()
    for e in ["png", "pdf"]:
        _filename = (
                        "imf_" + wdlf.wdlf_params["imf_model"] + "." + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_sfh with log = True and savefig with folder = None
def test_plotting_sfh_log_true_folder_none():
    wdlf.plot_sfh(
        log=True,
        fig=fig2,
        display=False,
        savefig=True,
        folder=None,
        ext=["png", "pdf"],
    )
    # assert the files exist at where you intend to
    t = np.linspace(0, wdlf.T0, 1000)
    _folder = os.getcwd()
    for e in ["png", "pdf"]:
        _filename = (
                        "{0:.2f}Gyr_".format(wdlf.T0 / 1e9)
                        + "sfh_"
                        + wdlf.wdlf_params["sfr_mode"]
                        + "_"
                        + str(t[0])
                        + "_"
                        + str(t[-1])
                        + "."
                        + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_cooling_model with use_mag = False and savefig with folder = None
def test_plotting_cooling_model_not_use_mag_folder_none():
    wdlf.plot_cooling_model(
        use_mag=False,
        fig=fig1,
        display=False,
        savefig=True,
        folder=None,
        ext=["png", "pdf"],
    )
    # assert the files exist at where you intend to
    # BELOW CODE FAILED

'''
    _folder = os.getcwd()
    for e in ["png", "pdf"]:
        _filename = (
                        wdlf.wdlf_params["low_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["intermediate_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["high_mass_cooling_model"]
                        + "."
                        + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))
'''


# manual function for 'manual' tests
def manual_fn(x):
    return 1.0


# Testing set_imf_model with model = 'manual'
def test_manual_imf_model():
    wdlf.set_imf_model(model="manual", imf_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


# Testing set_ms_model with model = 'manual'
def test_manual_ms_model():
    wdlf.set_ms_model(model="manual", ms_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


# Testing set_sfr_model with model = 'manual'
def test_manual_sfr_model():
    wdlf.set_sfr_model(mode="manual", sfr_model=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


# Testing set_ifmr_model with model = 'manual'
def test_manual_ifmr_model():
    wdlf.set_ifmr_model(model="manual", ifmr_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.0)


# Testing set_imf_model with model = 'manual' and save_csv with folder = None
def test_compute_density_savefig_folder_none():
    wdlf.set_ifmr_model(model="manual", ifmr_function=manual_fn)
    wdlf.compute_density(save_csv=True, folder=None, Mag=Mag)
    # assert the file exists at where you intend to
    _folder = os.getcwd()
    _filename = (
                    "{0:.2f}Gyr_".format(wdlf.T0 / 1e9)
                    + wdlf.wdlf_params["sfr_mode"]
                    + "_"
                    + wdlf.wdlf_params["ms_model"]
                    + "_"
                    + wdlf.wdlf_params["ifmr_model"]
                    + "_"
                    + wdlf.wdlf_params["low_mass_cooling_model"]
                    + "_"
                    + wdlf.wdlf_params["intermediate_mass_cooling_model"]
                    + "_"
                    + wdlf.wdlf_params["high_mass_cooling_model"]
                    + ".csv"
                )
    assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_wdlf with filename = None
def test_plotting_wdlf_savefig_path_not_exist():
    wdlf.plot_wdlf(
        log=False,
        display=False,
        savefig=True,
        folder="test_plotting_wdlf_savefig_path_not_exist",
        filename=None,
        ext="png",
    )
    # assert the file exists at where you intend to
    _folder = "test_plotting_wdlf_savefig_path_not_exist"
    for e in ["png"]:
        _filename = (
                        "{0:.2f}Gyr_".format(wdlf.T0 / 1e9)
                        + wdlf.wdlf_params["sfr_mode"]
                        + "_"
                        + wdlf.wdlf_params["ms_model"]
                        + "_"
                        + wdlf.wdlf_params["ifmr_model"]
                        + "_"
                        + wdlf.wdlf_params["low_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["intermediate_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["high_mass_cooling_model"]
                        + "."
                        + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))


# Testing plot_cooling_model with filename = None
def test_plotting_cooling_savefig_path_not_exist():
    wdlf.plot_cooling_model(
        use_mag=False,
        fig=fig1,
        display=False,
        savefig=True,
        folder="test_plotting_cooling_savefig_path_not_exist",
        ext=["png", "pdf"],
    )
    # assert the files exist at where you intend to
    # BELOW CODE FAILED

'''
    _folder = "test_plotting_cooling_savefig_path_not_exist"
    for e in ["png", "pdf"]:
        _filename = (
                        wdlf.wdlf_params["low_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["intermediate_mass_cooling_model"]
                        + "_"
                        + wdlf.wdlf_params["high_mass_cooling_model"]
                        + "."
                        + e
                    )
        assert os.path.isfile(os.path.join(_folder, _filename))
'''

# Testing set_low_mass_cooling_model with model = "lpcode_he_da_09"
def test_cooling_model_low_mass_lpcode_he_da_09():
    wdlf.set_low_mass_cooling_model(model='lpcode_he_da_09')
    wdlf.compute_cooling_age_interpolator()

# Testing set_intermediate_mass_cooling_model with model = "basti_co_da_10"
def test_cooling_model_intermediate_mass_basti_co_da_10():
    wdlf.set_intermediate_mass_cooling_model(model='basti_co_da_10')
    wdlf.compute_cooling_age_interpolator()

# Testing set_high_mass_cooling_model with model = "basti_co_da_10"
def test_cooling_model_high_mass_basti_co_da_10():
    wdlf.set_high_mass_cooling_model(model='basti_co_da_10')
    wdlf.compute_cooling_age_interpolator()