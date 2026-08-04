#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for example scripts."""

import os


def get_example_output_dir():
    """
    Return the output directory used by example scripts.

    The default location is `<example>/example_output`. Set the environment
    variable `WDPHOTTOOLS_EXAMPLE_OUTPUT_DIR` to override this location.
    """

    here = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.environ.get(
        "WDPHOTTOOLS_EXAMPLE_OUTPUT_DIR",
        os.path.join(here, "example_output"),
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
