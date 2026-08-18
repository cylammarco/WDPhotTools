#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the cooling tracks imported"""

import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from example_utils import get_example_output_dir


try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    HERE = os.path.dirname(os.path.realpath(__name__))

OUTPUT_DIR = get_example_output_dir()

atm = AtmosphereModelReader()

# Default passband is G3. The plotted boundary regions use the bounded
# extrapolation support provided by the atmosphere interpolators.
G = atm.interp_am(allow_extrapolation=True)
BP = atm.interp_am(dependent="G3_BP", allow_extrapolation=True)
RP = atm.interp_am(dependent="G3_RP", allow_extrapolation=True)

model_logg = np.unique(atm.model_da["logg"])
model_mbol = [atm.model_da["Mbol"][atm.model_da["logg"] == gravity] for gravity in model_logg]
mbol_interpolation_bounds = (
    max(np.min(values) for values in model_mbol),
    min(np.max(values) for values in model_mbol),
)
logg = np.array((6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0))
Mbol = np.linspace(0.0, 20.0, 201)
in_grid_mbol = (Mbol >= mbol_interpolation_bounds[0]) & (Mbol <= mbol_interpolation_bounds[1])

plt.figure(1, figsize=(8, 8))
for i in logg:
    logg_i = np.ones_like(Mbol) * i
    colour, magnitude = BP(logg_i, Mbol) - RP(logg_i, Mbol), G(logg_i, Mbol)
    is_extrapolated_logg = i < model_logg.min() or i > model_logg.max()

    if is_extrapolated_logg:
        plt.plot(
            colour,
            magnitude,
            linestyle="-.",
            label=rf"$\log(g) = {i}$ (extrapolated)",
        )
        continue

    line = plt.plot(
        colour[in_grid_mbol],
        magnitude[in_grid_mbol],
        label=rf"$\log(g) = {i}$",
    )[0]
    low_mbol_extrapolation = Mbol < mbol_interpolation_bounds[0]
    high_mbol_extrapolation = Mbol > mbol_interpolation_bounds[1]
    plt.plot(
        colour[low_mbol_extrapolation],
        magnitude[low_mbol_extrapolation],
        color=line.get_color(),
        linestyle="-.",
        label="_nolegend_",
    )
    plt.plot(
        colour[high_mbol_extrapolation],
        magnitude[high_mbol_extrapolation],
        color=line.get_color(),
        linestyle="-.",
        label="_nolegend_",
    )

plt.ylim(20.0, 6.0)
plt.grid()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    [Line2D([], [], color="black", linestyle="-."), *handles],
    ["Mbol extrapolation", *labels],
)
plt.xlabel("(BP - RP) / mag")
plt.ylabel("G / mag")
plt.title("DA Cooling tracks")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "DA_cooling_tracks.png"))
