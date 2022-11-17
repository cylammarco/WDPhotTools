#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test the atmosphere model reader"""

import pytest

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


@pytest.mark.xfail(raises=ValueError)
def test_unknown_atm_type():
    """test unknown atmosphere type"""
    amr = AtmosphereModelReader()
    amr.interp_am(atmosphere="DC")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_independent_variables():
    """test unknown independent variable"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent="BC")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_interpolator_variables_2d():
    """test unknown 2D interpolator"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="interp2d")


@pytest.mark.xfail(raises=ValueError)
def test_unsupported_interpolator_variables_1d():
    """test unknown 1D interpolator"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent="Mbol", interpolator="interp1d")


@pytest.mark.xfail(raises=TypeError)
def test_supplying_3_independent_variables():
    """test giving 3 variables"""
    amr = AtmosphereModelReader()
    amr.interp_am(
        independent=["logg", "Mbol", "Teff"], interpolator="interp1d"
    )


def test_interpolator_1d_rbf_mbol():
    """test RBF interpolator in 1D"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol"])


def test_interpolator_1d_ct_mbol():
    """test CT interpolator for Mbol"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol"])


def test_interpolator_1d_ct_teff():
    """test CT interpolator for Teff"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Teff"])


def test_interpolator_2d_rbf_logg_mbol():
    """test RBF interpolator for logg and Mbol"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["logg", "Mbol"])


def test_interpolator_2d_rbf_mbol_logg():
    """test RBF interpolator for Mbol and logg"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Mbol", "logg"])


def test_interpolator_2d_rbf_teff_logg():
    """test RBF interpolator for Teff and logg"""
    amr = AtmosphereModelReader()
    amr.interp_am(independent=["Teff", "logg"])


def test_interpolator_2d_ct_logg_mbol():
    """test CT interpolator for logg and Mbol"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["logg", "Mbol"])


def test_interpolator_2d_ct_mbol_logg():
    """test CT interpolator for Mbol and logg"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Mbol", "logg"])


def test_interpolator_2d_ct_teff_logg():
    """test CT interpolator for Teff and logg"""
    amr = AtmosphereModelReader()
    amr.interp_am(interpolator="CT", independent=["Teff", "logg"])
