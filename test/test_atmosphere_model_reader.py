import pytest

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


@pytest.mark.xfail(raises=ValueError)
def test_unknown_atm_type():
    amr = AtmosphereModelReader()
    amr.interp_am(atmosphere="DC")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_independent_variables():
    amr = AtmosphereModelReader()
    amr.interp_am(independent="BC")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_interpolator_variables_2D():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="interp2d")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_interpolator_variables_1D():
    amr = AtmosphereModelReader()
    amr.interp_am(independent="Mbol", interpolator="interp1d")


@pytest.mark.xfail(raises=TypeError)
def test_supplying_3_independent_variables():
    amr = AtmosphereModelReader()
    amr.interp_am(
        independent=["logg", "Mbol", "Teff"], interpolator="interp1d"
    )


def test_interpolator_1D_RBF():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol"])


def test_interpolator_1D_CT():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol"])


def test_interpolator_2D_RBF():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["logg", "Mbol"])


def test_interpolator_2D_RBF_swap_order_of_independent_variables():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol", "logg"])


def test_interpolator_2D_CT():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["logg", "Mbol"])


def test_interpolator_2D_CT_swap_order_of_independent_variables():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol", "logg"])
