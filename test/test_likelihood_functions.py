"""Regression tests for finite emcee log-probability handling."""

import numpy as np
import pytest

from WDPhotTools.fitter import _initialise_emcee_walkers
from WDPhotTools.likelihood_functions import log_likelihood


def _constant_interpolator(_x):
    """Return a finite model magnitude for likelihood tests."""

    return 10.0


@pytest.mark.parametrize(
    ("parameters", "errors", "interpolator", "prior"),
    (
        ([np.inf, 8.0], [0.02], _constant_interpolator, lambda *_: 0.0),
        ([13000.0, 8.0], [0.0], _constant_interpolator, lambda *_: 0.0),
        ([13000.0, 8.0], [0.02], lambda _x: np.inf, lambda *_: 0.0),
        ([13000.0, 8.0], [0.02], _constant_interpolator, lambda *_: np.nan),
    ),
)
def test_log_likelihood_rejects_non_finite_proposals(parameters, errors, interpolator, prior):
    """Invalid numerical terms must be rejected, never returned to emcee."""

    result = log_likelihood(
        parameters,
        obs=np.array([10.0]),
        errors=np.asarray(errors),
        distance=10.0,
        distance_err=None,
        interpolator_filter=[interpolator],
        prior=prior,
    )

    assert result == -np.inf


class _FiniteSampler:
    """Small sampler stand-in returning finite posterior values."""

    @staticmethod
    def compute_log_prob(positions):
        return -np.sum(np.asarray(positions) ** 2.0, axis=1), None


class _InvalidSampler:
    """Small sampler stand-in returning rejected posterior values."""

    @staticmethod
    def compute_log_prob(positions):
        return np.full(len(positions), -np.inf), None


def test_emcee_walkers_start_with_finite_log_probabilities():
    """Walker initialization must never pass ``-inf`` states to emcee."""

    positions = _initialise_emcee_walkers(_FiniteSampler(), [4000.0, 7.5], 12)

    assert positions.shape == (12, 2)
    assert np.isfinite(positions).all()


def test_emcee_walker_initialization_rejects_invalid_start():
    """An invalid posterior at every proposed start must fail clearly."""

    with pytest.raises(ValueError, match="finite log probabilities"):
        _initialise_emcee_walkers(_InvalidSampler(), [4000.0, 7.5], 12)
