#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initialise the import"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed


from . import atmosphere_model_reader
from . import cooling_model_reader
from . import fitter
from . import plotter
from . import reddening
from . import theoretical_lf
from . import util

__all__ = [
    "atmosphere_model_reader",
    "cooling_model_reader",
    "fitter",
    "plotter",
    "reddening",
    "theoretical_lf",
    "util",
]
__credits__ = ["K W Yuen", "M Green", "W Li"]
__status__ = "Production"
