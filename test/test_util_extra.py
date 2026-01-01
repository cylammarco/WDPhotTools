#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from WDPhotTools.util import get_uncertainty_least_squares, get_uncertainty_emcee, load_ms_lifetime_datatable


class DummyRes:
    def __init__(self, jac):
        self.jac = jac


def test_get_uncertainty_least_squares_degenerate():
    # Rank-deficient Jacobian should still return finite stdevs for well-conditioned cols
    J = np.array([[1.0, 0.0], [0.0, 0.0]])  # second column zero -> singular
    res = DummyRes(J)
    stdev = get_uncertainty_least_squares(res)
    assert np.isfinite(stdev[0])
    assert stdev.shape[0] == 2


def test_get_uncertainty_emcee_shapes():
    # 1D samples
    s1 = np.random.normal(size=1000)
    lo_hi = get_uncertainty_emcee(s1)
    assert lo_hi.shape == (2,)
    # 2D samples (flattening behavior)
    s2 = np.random.normal(size=(50, 20))
    lo_hi2 = get_uncertainty_emcee(s2)
    assert lo_hi2.shape == (2,)


def test_load_ms_lifetime_datatable_smoke():
    # Pick a known CSV key from mapping in WDLF, ensure file loads
    dt = load_ms_lifetime_datatable("PARSECz0017.csv")
    assert dt.ndim == 2 and dt.shape[1] >= 2
