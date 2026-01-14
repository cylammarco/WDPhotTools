#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pytest
from importlib.resources import files


def _repo_path(*parts):
    return str(files("WDPhotTools").joinpath(*parts))


def _load_numeric_table(path, delimiter=None):
    num_re = re.compile(r"[+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?")
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in f:
            if re.match(r"^[=\-\s]*$", i):
                continue
            nums = [float(m.group(0)) for m in num_re.finditer(i)]
            if len(nums) >= 2:
                rows.append(nums)
    if not rows:
        raise AssertionError(f"No numeric lines detected in {path}")
    lengths = {}
    for r in rows:
        lengths[len(r)] = lengths.get(len(r), 0) + 1
    target_len = max(lengths, key=lengths.get)
    filtered = [r[:target_len] for r in rows if len(r) >= target_len]
    return np.array(filtered, dtype=float)


@pytest.mark.parametrize(
    "fname",
    [
        # ms_lifetime CSVs
        os.path.join(_repo_path("ms_lifetime"), f)
        for f in os.listdir(_repo_path("ms_lifetime"))
        if f.endswith(".csv")
    ],
)
def test_ms_lifetime_csv_loads(fname):
    dt = np.loadtxt(fname, delimiter=",")
    assert dt.ndim == 2 and dt.shape[0] > 0 and dt.shape[1] > 0


@pytest.mark.parametrize(
    "fname",
    [
        # extinction CSVs
        os.path.join(_repo_path("extinction"), f)
        for f in os.listdir(_repo_path("extinction"))
        if f.endswith(".csv")
    ],
)
def test_extinction_csv_loads(fname):
    dt = np.loadtxt(fname, delimiter=",")
    assert dt.ndim == 2 and dt.shape[0] > 0 and dt.shape[1] > 0


@pytest.mark.parametrize(
    "fname",
    [
        # wd_cooling bedard20 txts
        os.path.join(_repo_path("wd_cooling", "bedard20"), f)
        for f in os.listdir(_repo_path("wd_cooling", "bedard20"))
        if f.endswith(".txt")
    ],
)
def test_wd_cooling_bedard20_txt_loads(fname):
    dt = _load_numeric_table(fname)
    assert dt.ndim == 2 and dt.shape[0] > 0 and dt.shape[1] > 0


@pytest.mark.parametrize(
    "fname",
    [
        # wd_photometry tables
        os.path.join(_repo_path("wd_photometry"), f)
        for f in os.listdir(_repo_path("wd_photometry"))
        if f.endswith(".txt")
    ],
)
def test_wd_photometry_txt_loads(fname):
    dt = _load_numeric_table(fname)
    assert dt.ndim == 2 and dt.shape[0] > 0 and dt.shape[1] > 0
