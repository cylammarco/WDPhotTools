#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

__credits__ = ["K W Yuen", "W Li", "M Green"]
__status__ = "Production"

from . import theoretical_lf
from . import cooling_model_reader
from . import atmosphere_model_reader
from . import plotter
from . import fitter
from . import reddening
from . import util

__all__ = [
    "theoretical_lf",
    "cooling_model_reader",
    "atmosphere_model_reader",
    "plotter",
    "fitter",
    "reddening",
    "util",
]
