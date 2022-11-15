#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing the global spline in 2D"""

import numpy as np
import pytest

from WDPhotTools.util import GlobalSpline2D


x = np.arange(10)
y = np.arange(10) ** 2.0
z = np.arange(10) ** 3.0


def test_gs2d():
    """Run a 2D global spline and check against the diagonal values"""
    gs2d = GlobalSpline2D(x, y, z)
    assert np.isclose(np.diag(gs2d(x, y)), z).all()


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_linear_x_lt_2():
    """give wrong size of x"""
    GlobalSpline2D([1.0], y, z)


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_cubic_x_lt_4():
    """give wrong size of x"""
    GlobalSpline2D([1.0, 2.0, 3.0], y, z, kind="cubic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_quintic_x_lt_6():
    """give wrong size of x"""
    GlobalSpline2D([1.0, 2.0, 3.0, 4.0, 5.0], y, z, kind="quintic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_linear_y_lt_2():
    """give wrong size of y"""
    GlobalSpline2D(x, [1.0], z)


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_cubic_y_lt_4():
    """give wrong size of y"""
    GlobalSpline2D(x, [1.0, 2.0, 3.0], z, kind="cubic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_quintic_y_lt_6():
    """give wrong size of y"""
    GlobalSpline2D(x, [1.0, 2.0, 3.0, 4.0, 5.0], z, kind="quintic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_unsupported_kind():
    """give wrong str for kind"""
    GlobalSpline2D(x, y, z, kind="blabla")
