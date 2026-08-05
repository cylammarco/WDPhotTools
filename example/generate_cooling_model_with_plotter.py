#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the default cooling model"""

from WDPhotTools import plotter
from example_utils import get_example_output_dir

OUTPUT_DIR = get_example_output_dir()

plotter.plot_cooling_model(
    mass=[0.2, 0.4, 0.6, 0.8, 1.0],
    savefig=True,
    folder=OUTPUT_DIR,
    filename="DA_cooling_model_from_plotter",
)
