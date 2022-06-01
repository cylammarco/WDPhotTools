import numpy as np
from scipy.interpolate import interp1d

from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()
wdlf_manual = theoretical_lf.WDLF()

Mag = np.arange(4.0, 16.0, 2.0)
age = [3.0e9, 12.0e9]
num = np.zeros((len(age), len(Mag)))

wdlf.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
wdlf_manual.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf_manual.compute_cooling_age_interpolator()


# manual imf function
def manual_K01_imf(M):
    M = np.asarray(M).reshape(-1)
    MF = M**-2.3
    # mass lower than 0.08 is impossible, so that range is ignored.
    if (M < 0.5).any():
        M_mask = M < 0.5
        MF[M_mask] = M[M_mask] ** 1.3
    return MF


# manual ifmr function
def manual_C08_ifmr(M):
    M = np.asarray(M).reshape(-1)
    m = 0.117 * M + 0.384
    if (m < 0.4349).any():
        m[m < 0.4349] = 0.4349
    return m


# manual sfr function
def manual_constant_sfr(age):
    t1 = age
    t0 = t1 * 1.00001
    # current time = 0.
    t2 = 0.0
    t3 = t2 * 0.99999
    sfr = interp1d(
        np.array([30e9, t0, t1, t2, t3, -30e9]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        fill_value="extrapolate",
    )
    return sfr


# Testing set_imf_model with model = 'manual'
def test_manual_imf_model():
    wdlf.set_imf_model(model="K01")
    wdlf.compute_density(Mag=Mag)
    wdlf_manual.set_imf_model(model="manual", imf_function=manual_K01_imf)
    wdlf_manual.compute_density(Mag=Mag)
    assert np.isclose(wdlf.number_density, wdlf_manual.number_density).all()


# Testing set_ifmr_model with model = 'manual'
def test_manual_ifmr_model():
    wdlf.set_ifmr_model(model="C08")
    wdlf.compute_density(Mag=Mag)
    wdlf_manual.set_ifmr_model(model="manual", ifmr_function=manual_C08_ifmr)
    wdlf_manual.compute_density(Mag=Mag)
    assert np.isclose(wdlf.number_density, wdlf_manual.number_density).all()


# Testing set_sfr_model with model = 'manual'
def test_manual_sfr_model():
    wdlf.set_sfr_model(mode="constant", age=5e9)
    wdlf.compute_density(Mag=Mag)
    wdlf_manual.set_sfr_model(
        mode="manual", sfr_model=manual_constant_sfr(5e9)
    )
    wdlf_manual.compute_density(Mag=Mag)
    assert np.isclose(
        wdlf.number_density, wdlf_manual.number_density, rtol=1e-3, atol=1e-3
    ).all()
