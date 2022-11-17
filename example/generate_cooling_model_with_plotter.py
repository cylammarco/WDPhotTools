#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the default cooling model"""

from WDPhotTools import plotter

plotter.plot_cooling_model(
    mass=[0.2, 0.4, 0.6, 0.8, 1.0],
    savefig=True,
    folder="example_output",
    filename="DA_cooling_model_from_plotter",
)
