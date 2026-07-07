#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Smoke tests for figure-generation snippets in README."""

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from WDPhotTools import plotter
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader

matplotlib.use("Agg")


def test_readme_plotter_atmosphere_figure(tmp_path):
    output = Path(tmp_path)
    plotter.plot_atmosphere_model(
        invert_yaxis=True,
        display=False,
        savefig=True,
        folder=str(output),
        filename="DA_cooling_tracks_from_plotter",
        ext="png",
    )
    assert (output / "DA_cooling_tracks_from_plotter.png").is_file()


def test_readme_custom_atmosphere_figure(tmp_path):
    atm = AtmosphereModelReader()

    g_band = atm.interp_am()
    bp_band = atm.interp_am(dependent="G3_BP")
    rp_band = atm.interp_am(dependent="G3_RP")

    logg = np.arange(7.0, 9.5, 0.5)
    mbol = np.arange(0.0, 20.0, 0.1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for value in logg:
        ax.plot(
            bp_band(value, mbol) - rp_band(value, mbol),
            g_band(value, mbol),
        )
    ax.set_ylim(20.0, 6.0)
    ax.grid()
    ax.set_xlabel("(BP - RP) / mag")
    ax.set_ylabel("G / mag")
    ax.set_title("DA Cooling tracks")
    fig.tight_layout()

    output = Path(tmp_path) / "DA_cooling_tracks.png"
    fig.savefig(output)
    plt.close(fig)
    assert output.is_file()


def test_readme_plotter_cooling_figure(tmp_path):
    output = Path(tmp_path)
    plotter.plot_cooling_model(
        mass=[0.2, 0.4, 0.6, 0.8, 1.0],
        display=False,
        savefig=True,
        folder=str(output),
        filename="DA_cooling_model_from_plotter",
        ext="png",
    )
    assert (output / "DA_cooling_model_from_plotter.png").is_file()
