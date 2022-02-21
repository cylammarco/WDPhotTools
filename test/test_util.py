import numpy as np
import pytest

from WDPhotTools.util import GlobalSpline2D


x = np.arange(10)
y = np.arange(10) ** 2.0
z = np.arange(10) ** 3.0


def test_gs2d():
    gs2d = GlobalSpline2D(x, y, z)


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_linear_x_lt_2():
    gs2d = GlobalSpline2D([1.0], y, z)


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_cubic_x_lt_4():
    gs2d = GlobalSpline2D([1.0, 2.0, 3.0], y, z, kind="cubic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_quintic_x_lt_6():
    gs2d = GlobalSpline2D([1.0, 2.0, 3.0, 4.0, 5.0], y, z, kind="quintic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_linear_y_lt_2():
    gs2d = GlobalSpline2D(x, [1.0], z)


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_cubic_y_lt_4():
    gs2d = GlobalSpline2D(x, [1.0, 2.0, 3.0], z, kind="cubic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_quintic_y_lt_6():
    gs2d = GlobalSpline2D(x, [1.0, 2.0, 3.0, 4.0, 5.0], z, kind="quintic")


@pytest.mark.xfail(raises=ValueError)
def test_gs2d_unsupported_kind():
    gs2d = GlobalSpline2D(x, y, z, kind="blabla")
