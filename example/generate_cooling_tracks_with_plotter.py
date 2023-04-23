#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create the default cooling tracks"""

import os

from WDPhotTools import plotter


try:
    HERE = os.path.dirname(os.path.realpath(__file__))

except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))

plotter.plot_atmosphere_model(
    invert_yaxis=True,
    savefig=True,
    folder="example_output",
    filename="DA_cooling_tracks_from_plotter",
)
