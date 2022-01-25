from importlib.util import module_for_loader
from re import X
from matplotlib import pyplot as plt
import numpy as np
import pytest
from unittest.mock import patch

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0, 20., 2.0)
age = [3.0E9, 12.0e9]
num = np.zeros((len(age), len(Mag)))


@pytest.mark.xfail
def test_changing_sfr_mode_to_unknown_name():
    wdlf.set_sfr_model('blabla')


@pytest.mark.xfail
def test_changing_imf_model_to_unknown_name():
    wdlf.set_imf_model('blabla')


@pytest.mark.xfail
def test_changing_ms_model_to_unknown_name():
    wdlf.set_ms_model('blabla')


@pytest.mark.xfail
def test_changing_ifmr_model_to_unknown_name():
    wdlf.set_ifmr_model('blabla')


@pytest.mark.xfail
def test_changing_empty_low_mass_model_to_unknown_name():
    wdlf.set_low_mass_cooling_model('blabla')


@pytest.mark.xfail
def test_changing_empty_intermediate_mass_model_to_unknown_name():
    wdlf.set_intermediate_mass_cooling_model('blabla')


@pytest.mark.xfail
def test_changing_empty_high_mass_model_to_unknown_name():
    wdlf.set_high_mass_cooling_model('blabla')


def test_default():
    wdlf.compute_density(Mag=Mag)


wdlf.set_sfr_model(mode='burst', age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf.compute_density(Mag=Mag)


@patch("matplotlib.pyplot.show")
def test_plotting(mock_show):
    wdlf.plot_cooling_model(display=False,
                            savefig=True,
                            folder='test_output',
                            filename='test_plot_cooling_model',
                            ext='png')
    wdlf.plot_sfh(display=False,
                  savefig=True,
                  folder='test_output',
                  filename='test_plot_sfh',
                  ext='png')
    wdlf.plot_imf(display=False,
                  savefig=True,
                  folder='test_output',
                  filename='test_plot_imf',
                  ext='png')
    wdlf.plot_ifmr(display=False,
                   savefig=True,
                   folder='test_output',
                   filename='test_plot_ifmr',
                   ext='png')
    wdlf.plot_wdlf(display=False,
                   savefig=True,
                   folder='test_output',
                   filename='test_plot_wdlf',
                   ext='png')


fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
fig4 = plt.figure(4)
fig5 = plt.figure(5)


@patch("matplotlib.pyplot.show")
def test_plotting_to_an_external_Figure_object(mock_show):
    wdlf.plot_cooling_model(use_mag=True,
                            fig=fig1,
                            display=True,
                            savefig=True,
                            folder='test_output',
                            ext=['png', 'pdf'])
    wdlf.plot_sfh(fig=fig2,
                  display=True,
                  savefig=True,
                  folder='test_output',
                  ext=['png', 'pdf'])
    wdlf.plot_imf(fig=fig3,
                  display=True,
                  savefig=True,
                  folder='test_output',
                  ext=['png', 'pdf'])
    wdlf.plot_ifmr(fig=fig4,
                   display=True,
                   savefig=True,
                   folder='test_output',
                   ext=['png', 'pdf'])
    wdlf.plot_wdlf(fig=fig5,
                   display=True,
                   savefig=True,
                   folder='test_output',
                   ext=['png', 'pdf'])


# not normalising the WDLF
def test_changing_sfr_model():
    wdlf.set_sfr_model(mode='constant', age=age[0])
    wdlf.compute_density(Mag=Mag, save_csv=True, folder='test_output')
    wdlf.set_sfr_model(mode='decay', age=age[0])
    wdlf.compute_density(Mag=Mag,
                         save_csv=True,
                         normed=False,
                         folder='test_output',
                         filename='test_saving_wdlf_csv')
    assert np.sum(wdlf.number_density) != 1.


def test_changing_imf_model():
    wdlf.set_imf_model('K01')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_imf_model('C03')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_imf_model('C03b')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_sfr_model(mode='decay', age=age[1])
    assert np.isclose(np.sum(wdlf.number_density), 1.)


def test_changing_ms_model():
    wdlf.set_ms_model('C16')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ms_model('Bressan')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


def test_changing_ifmr_model():
    wdlf.set_ifmr_model('C08')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('C08b')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('S09')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('S09b')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('W09')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('K09')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('K09b')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    wdlf.set_ifmr_model('C18')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)
    #added by YKW 23jan2022
    wdlf.set_ifmr_model('EB18')
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


#YKW Test 1 23Jan2022
def test_changing_imf_K01_small_mass_log():
    wdlf.set_imf_model('K01')
    wdlf.plot_imf(display=False, log=True)


#YKW Test 2 23Jan2022
def test_plotting_wdlf_log_false_folder_none():
    wdlf.plot_wdlf(log=False,
                   display=False,
                   savefig=True,
                   folder=None,
                   filename='test_plot_wdlf',
                   ext='png')
    # assert the file exists at where you intend to


#YKW Test 3 23Jan2022
def test_plotting_ifmr_folder_none():
    wdlf.plot_ifmr(fig=fig4,
                   display=False,
                   savefig=True,
                   folder=None,
                   ext=['png', 'pdf'])
    # assert the files exist at where you intend to


#YKW Test 4 23Jan2022
def test_plotting_imf_folder_none():
    wdlf.plot_imf(fig=fig3,
                  display=False,
                  savefig=True,
                  folder=None,
                  ext=['png', 'pdf'])
    # assert the files exist at where you intend to


#YKW Test 5 23Jan2022
def test_plotting_sfh_log_true_folder_none():
    wdlf.plot_sfh(log=True,
                  fig=fig2,
                  display=False,
                  savefig=True,
                  folder=None,
                  ext=['png', 'pdf'])
    # assert the files exist at where you intend to


#YKW Test 6 23Jan2022
def test_plotting_cooling_model_not_use_mag_folder_none():
    wdlf.plot_cooling_model(use_mag=False,
                            fig=fig1,
                            display=False,
                            savefig=True,
                            folder=None,
                            ext=['png', 'pdf'])
    # assert the files exist at where you intend to


# manual function for 'manual' tests
def manual_fn(x):
    return 1.0


# YKW Test 1 test 'manual' imf 23Jan2022
def test_manual_imf_model():
    wdlf.set_imf_model(model='manual', imf_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


# YKW Test 2 test 'manual' ms model 23Jan2022
def test_manual_ms_model():
    wdlf.set_ms_model(model='manual', ms_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


# YKW Test 3 test 'manual' sfr mode 23Jan2022
def test_manual_sfr_model():
    wdlf.set_sfr_model(mode='manual', sfr_model=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


# YKW Test 4 test 'manual' ifmr model 23Jan2022
def test_manual_ifmr_model():
    wdlf.set_ifmr_model(model='manual', ifmr_function=manual_fn)
    wdlf.compute_density(Mag=Mag)
    assert np.isclose(np.sum(wdlf.number_density), 1.)


# YKW Test 5 test compute density savefig folder none 23Jan2022
def test_compute_density_savefig_folder_none():
    wdlf.set_ifmr_model(model='manual', ifmr_function=manual_fn)
    wdlf.compute_density(save_csv=True, folder=None, Mag=Mag)
    # assert the file exists at where you intend to


# YKW Test 6 wdlf savefig path not exist 23Jan2022
def test_plotting_wdlf_savefig_path_not_exist():
    wdlf.plot_wdlf(log=False,
                   display=False,
                   savefig=True,
                   folder='test_plot_wdlf_ykw_23jan',
                   filename=None,
                   ext='png')
    # assert the file exists at where you intend to


# YKW Test 7 cooling savefig path not exist 23Jan2022
def test_plotting_cooling_savefig_path_not_exist():
    wdlf.plot_cooling_model(use_mag=False,
                            fig=fig1,
                            display=False,
                            savefig=True,
                            folder='test_plot_cooling_ykw_23jan',
                            ext=['png', 'pdf'])
    # assert the files exist at where you intend to
