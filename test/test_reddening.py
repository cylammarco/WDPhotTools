import numpy as np
import pytest
from WDPhotTools.reddening import reddening_vector_interpolated
from WDPhotTools.reddening import reddening_vector_filter

wave_grizyJHK = np.array((4876.7, 6200.1, 7520.8, 8665.3, 9706.3, 12482.9, 16588.4, 21897.7))

rv_grizyJHK_21 = np.array((3.634, 2.241, 1.568, 1.258, 1.074, 0.764, 0.502, 0.331))
rv_grizyJHK_31 = np.array((3.172, 2.271, 1.682, 1.322, 1.087, 0.709, 0.449, 0.302))
rv_grizyJHK_41 = np.array((2.958, 2.284, 1.734, 1.352, 1.094, 0.684, 0.425, 0.288))
rv_grizyJHK_51 = np.array((2.835, 2.292, 1.765, 1.369, 1.097, 0.669, 0.411, 0.280))

rv = reddening_vector_interpolated(kernel="cubic")
rv_linear = reddening_vector_interpolated(kernel="linear")
rv_quintic = reddening_vector_interpolated(kernel="quintic")


# Test the Av values when rv = 2.1, 3.1, 4.1 and 5.1
def test_rv21():
    assert np.allclose(rv(wave_grizyJHK, 2.1), rv_grizyJHK_21, rtol=1e-2, atol=1e-2)


def test_rv31():
    assert np.allclose(rv(wave_grizyJHK, 3.1), rv_grizyJHK_31, rtol=1e-2, atol=1e-2)


def test_rv41():
    assert np.allclose(rv(wave_grizyJHK, 4.1), rv_grizyJHK_41, rtol=1e-2, atol=1e-2)


def test_rv51():
    assert np.allclose(rv(wave_grizyJHK, 5.1), rv_grizyJHK_51, rtol=1e-2, atol=1e-2)


# repeat of linear interpolation
# Test the Av values when rv = 2.1, 3.1, 4.1 and 5.1
def test_rv21_linear():
    # Note this one achieves lower accuracy than the rest
    assert np.allclose(rv_linear(wave_grizyJHK, 2.1), rv_grizyJHK_21, rtol=1e-2, atol=1e-2)


def test_rv31_linear():
    assert np.allclose(rv_linear(wave_grizyJHK, 3.1), rv_grizyJHK_31, rtol=1e-2, atol=1e-2)


def test_rv41_linear():
    assert np.allclose(rv_linear(wave_grizyJHK, 4.1), rv_grizyJHK_41, rtol=1e-2, atol=1e-2)


def test_rv51_linear():
    assert np.allclose(rv_linear(wave_grizyJHK, 5.1), rv_grizyJHK_51, rtol=1e-2, atol=1e-2)


# repeat of quintic interpolation which don't reach 1% accuracy.
# Test the Av values when rv = 2.1, 3.1, 4.1 and 5.1
@pytest.mark.xfail
def test_rv21_quintic():
    assert np.allclose(rv_quintic(wave_grizyJHK, 2.1), rv_grizyJHK_21)


@pytest.mark.xfail
def test_rv31_quintic():
    assert np.allclose(rv_quintic(wave_grizyJHK, 3.1), rv_grizyJHK_31)


@pytest.mark.xfail
def test_rv41_quintic():
    assert np.allclose(rv_quintic(wave_grizyJHK, 4.1), rv_grizyJHK_41)


@pytest.mark.xfail
def test_rv51_quintic():
    assert np.allclose(rv_quintic(wave_grizyJHK, 5.1), rv_grizyJHK_51)


red_g = reddening_vector_filter("g_ps1")
red_r = reddening_vector_filter("r_ps1")
red_i = reddening_vector_filter("i_ps1")
red_z = reddening_vector_filter("z_ps1")
red_y = reddening_vector_filter("y_ps1")
red_J = reddening_vector_filter("J_mko")
red_H = reddening_vector_filter("H_mko")
red_K = reddening_vector_filter("K_mko")


def test_rv21_filter():
    red = np.array(
        [
            red_g([8.0, 7000.0, 2.1]),
            red_r([8.0, 7000.0, 2.1]),
            red_i([8.0, 7000.0, 2.1]),
            red_z([8.0, 7000.0, 2.1]),
            red_y([8.0, 7000.0, 2.1]),
            red_J([8.0, 7000.0, 2.1]),
            red_H([8.0, 7000.0, 2.1]),
            red_K([8.0, 7000.0, 2.1]),
        ]
    ).flatten()
    assert np.allclose(red, rv_grizyJHK_21, rtol=1e-2, atol=1e-2)


def test_rv31_filter():
    red = np.array(
        [
            red_g([8.0, 7000.0, 3.1]),
            red_r([8.0, 7000.0, 3.1]),
            red_i([8.0, 7000.0, 3.1]),
            red_z([8.0, 7000.0, 3.1]),
            red_y([8.0, 7000.0, 3.1]),
            red_J([8.0, 7000.0, 3.1]),
            red_H([8.0, 7000.0, 3.1]),
            red_K([8.0, 7000.0, 3.1]),
        ]
    ).flatten()
    assert np.allclose(red, rv_grizyJHK_31, rtol=1e-2, atol=1e-2)


def test_rv41_filter():
    red = np.array(
        [
            red_g([8.0, 7000.0, 4.1]),
            red_r([8.0, 7000.0, 4.1]),
            red_i([8.0, 7000.0, 4.1]),
            red_z([8.0, 7000.0, 4.1]),
            red_y([8.0, 7000.0, 4.1]),
            red_J([8.0, 7000.0, 4.1]),
            red_H([8.0, 7000.0, 4.1]),
            red_K([8.0, 7000.0, 4.1]),
        ]
    ).flatten()
    assert np.allclose(red, rv_grizyJHK_41, rtol=1e-2, atol=1e-2)


def test_rv51_filter():
    red = np.array(
        [
            red_g([8.0, 7000.0, 5.1]),
            red_r([8.0, 7000.0, 5.1]),
            red_i([8.0, 7000.0, 5.1]),
            red_z([8.0, 7000.0, 5.1]),
            red_y([8.0, 7000.0, 5.1]),
            red_J([8.0, 7000.0, 5.1]),
            red_H([8.0, 7000.0, 5.1]),
            red_K([8.0, 7000.0, 5.1]),
        ]
    ).flatten()
    assert np.allclose(red, rv_grizyJHK_51, rtol=1e-2, atol=1e-2)
