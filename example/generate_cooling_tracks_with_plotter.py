#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create the default cooling tracks."""

from WDPhotTools import plotter
from example_utils import get_example_output_dir

OUTPUT_DIR = get_example_output_dir()

plotter.plot_atmosphere_model(
    invert_yaxis=True,
    savefig=True,
    folder=OUTPUT_DIR,
    filename="DA_cooling_tracks_from_plotter",
)
