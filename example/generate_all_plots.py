#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot everything"""

import numpy as np

from WDPhotTools import theoretical_lf
from example_utils import get_example_output_dir

OUTPUT_DIR = get_example_output_dir()

wdlf = theoretical_lf.WDLF()

mag = np.arange(0, 20.0, 2.5)
age = [3.0e9]
num = np.zeros((len(age), len(mag)))

wdlf.set_sfr_model(mode="burst", age=age[0], duration=1e8)
wdlf.compute_cooling_age_interpolator()
fig_input_models = wdlf.plot_input_models(
    cooling_model_use_mag=False,
    imf_log=True,
    display=True,
    folder=OUTPUT_DIR,
    ext=["png", "pdf"],
    savefig=True,
)

wdlf.compute_density(mag=mag)

fig_wdlf = wdlf.plot_wdlf(display=True)
