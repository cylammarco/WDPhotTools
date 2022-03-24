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


def test_interpolator_1D_RBF_Mbol():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol"])


def test_interpolator_1D_CT_Mbol():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol"])


def test_interpolator_1D_CT_Teff():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Teff"])


def test_interpolator_2D_RBF_logg_Mbol():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["logg", "Mbol"])


def test_interpolator_2D_RBF_Mbol_logg():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol", "logg"])


def test_interpolator_2D_RBF_Teff_logg():
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Teff", "logg"])


def test_interpolator_2D_CT_logg_Mbol():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["logg", "Mbol"])


def test_interpolator_2D_CT_Mbol_logg():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol", "logg"])


def test_interpolator_2D_CT_Teff_logg():
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Teff", "logg"])
