#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from WDPhotTools.diff2_functions_least_square import (
    diff2,
    diff2_distance_red_filter,
    diff2_distance_red_filter_fixed_logg,
)


def _const_interp(val):
    return lambda *_: float(val)


def test_diff2_basic_shapes():
    interps = [_const_interp(10.0) for _ in range(3)]
    obs = np.array([10.0, 10.5, 11.0])
    err = np.array([0.1, 0.1, 0.1])
    d2, e2 = diff2(np.array([0.0]), obs, err, 10.0, 0.1, interps, True)
    assert d2.shape == obs.shape and e2.shape == obs.shape


def test_diff2_distance_red_filter_invalid_distance_returns_inf():
    interps = [_const_interp(10.0) for _ in range(2)]
    teff_itp = _const_interp(10000.0)
    redv = [_const_interp(0.0) for _ in range(2)]
    d2, e2 = diff2_distance_red_filter(
        np.array([8.0, 10.0, -1.0]),
        np.array([11.0, 12.0]),
        np.array([0.1, 0.1]),
        interps,
        teff_itp,
        0,  # logg_pos
        3.1,
        "total",
        redv,
        0.0,
        None,
        None,
        None,
        None,
        True,
    )
    assert np.isinf(d2).all() and np.isinf(e2).all()


def test_diff2_distance_red_filter_fixed_logg_invalid_distance_returns_inf():
    interps = [_const_interp(10.0) for _ in range(2)]
    teff_itp = _const_interp(10000.0)
    redv = [_const_interp(0.0) for _ in range(2)]
    d2, e2 = diff2_distance_red_filter_fixed_logg(
        np.array([8.0, 10.0, -1.0]),
        np.array([11.0, 12.0]),
        np.array([0.1, 0.1]),
        interps,
        teff_itp,
        8.0,
        3.1,
        "total",
        redv,
        0.0,
        None,
        None,
        None,
        None,
        True,
    )
    assert np.isinf(d2).all() and np.isinf(e2).all()
