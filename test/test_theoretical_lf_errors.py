#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from WDPhotTools.theoretical_lf import WDLF


def test_set_ms_model_invalid():
    wdlf = WDLF()
    with pytest.raises(ValueError):
        wdlf.set_ms_model("NOT_A_MODEL")


def test_compute_density_manual_sfr_smoke():
    wdlf = WDLF()
    wdlf.set_sfr_model(mode="manual", sfr_model=lambda t: 1.0 if t > 0 else 0.0, age=1e9)
    mag = np.linspace(5, 15, 5)
    edges, density = wdlf.compute_density(mag)
    assert edges.size == density.size
