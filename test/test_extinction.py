import numpy as np
import pytest

from WDPhotTools.extinction import get_extinction_fraction


def test_get_extinction_fraction():
    """Test exitnction function"""
    np.isclose(
        get_extinction_fraction(
            distance=250.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        1.0,
    )
    np.isclose(
        get_extinction_fraction(
            distance=75.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        0.0,
    )
    np.isclose(
        get_extinction_fraction(
            distance=100.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        0.1386644841777267,
    )
    np.isclose(
        get_extinction_fraction(
            distance=175.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        0.6386628473110217,
    )
    np.isclose(
        get_extinction_fraction(
            distance=250.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        1.0,
    )
    np.isclose(
        get_extinction_fraction(
            distance=251.0,
            ra=192.85949646,
            dec=27.12835323,
            z_min=100.0,
            z_max=250.0,
        ),
        1.0,
    )
    np.isclose(
        get_extinction_fraction(
            distance=500.0,
            ra=240.63412385,
            dec=-11.01234783,
            z_min=100.0,
            z_max=250.0,
        ),
        1.0,
    )


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_pass_zmin_zmax_none():
    """Test dec larger than 90"""
    get_extinction_fraction(
        distance=150.0, ra=10.0, dec=100.0, z_min=100.0, z_max=250.0
    )


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_fail_ra():
    """test ra and dec are None"""
    get_extinction_fraction(
        distance=150.0, ra=None, dec=None, z_min=100.0, z_max=250.0
    )
