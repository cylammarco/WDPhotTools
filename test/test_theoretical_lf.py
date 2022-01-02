import numpy as np
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

Mag = np.arange(0, 20., 1.0)
age = [3.0E9]
num = np.zeros((len(age), len(Mag)))

wdlf.set_sfr_model(mode='burst', age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf.compute_density(Mag=Mag)


def test_plotting():
    wdlf.plot_cooling_model(display=False)
    wdlf.plot_sfh(display=False)
    wdlf.plot_imf(display=False)
    wdlf.plot_ifmr(display=False)
    wdlf.plot_wdlf(display=False)


def test_changing_sfr_model():
    wdlf.set_sfr_model(mode='constant', age=age[0])
    wdlf.compute_density(Mag=Mag)
    wdlf.set_sfr_model(mode='decay', age=age[0])
    wdlf.compute_density(Mag=Mag)


def test_changing_imf_model():
    wdlf.set_imf_model('K01')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_imf_model('C03')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_imf_model('C03b')
    wdlf.compute_density(Mag=Mag)


def test_changing_ms_model():
    wdlf.set_ms_model('C16')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ms_model('Bressan')
    wdlf.compute_density(Mag=Mag)


def test_changing_ifmr_model():
    wdlf.set_ifmr_model('C08')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('C08b')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('S09')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('S09b')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('W09')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('K09')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('K09b')
    wdlf.compute_density(Mag=Mag)
    wdlf.set_ifmr_model('C18')
    wdlf.compute_density(Mag=Mag)
