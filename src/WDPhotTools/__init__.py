#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Initialise the import"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass  # package is not installed


from . import atmosphere_model_reader
from . import cooling_model_reader
from . import diff2_functions_least_square
from . import diff2_functions_minimize
from . import fitter
from . import plotter
from . import reddening
from . import theoretical_lf
from . import util

__all__ = [
    "atmosphere_model_reader",
    "cooling_model_reader",
    "diff2_functions_least_square",
    "diff2_functions_minimize",
    "fitter",
    "plotter",
    "reddening",
    "theoretical_lf",
    "util",
]
__credits__ = ["K W Yuen", "M Green", "W Li"]
__status__ = "Production"

