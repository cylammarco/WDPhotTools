from functools import wraps

import numpy as np

from .diff2_functions_least_square import (
    diff2,
    diff2_distance,
    diff2_distance_red_filter,
    diff2_distance_red_filter_fixed_logg,
    diff2_distance_red_interpolated,
    diff2_distance_red_interpolated_fixed_logg,
    diff2_red_filter,
    diff2_red_filter_fixed_logg,
    diff2_red_interpolated,
)


_INVALID_RESIDUAL = np.float64(1e30)


def _safe_log_probability(log_probability):
    """Return ``-inf`` when a sampler proposal cannot be evaluated safely."""

    @wraps(log_probability)
    def wrapped(_x, *args, **kwargs):
        try:
            if not np.isfinite(np.asarray(_x, dtype=float)).all():
                return -np.inf
            with np.errstate(all="ignore"):
                value = log_probability(_x, *args, **kwargs)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return -np.inf

        value = np.asarray(value, dtype=float)
        if value.size != 1 or not np.isfinite(value.item()):
            return -np.inf
        return float(value.item())

    return wrapped


def _log_posterior(_x, d2, e2, prior):
    """Compute a finite posterior or reject an invalid sampler proposal."""

    d2 = np.asarray(d2, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if (
        not np.isfinite(d2).all()
        or not np.isfinite(e2).all()
        or np.any(d2 < 0.0)
        or np.any(d2 >= _INVALID_RESIDUAL)
        or np.any(e2 <= 0.0)
    ):
        return -np.inf

    prior_value = np.asarray(prior(*_x), dtype=float)
    if prior_value.size != 1 or not np.isfinite(prior_value.item()):
        return -np.inf

    with np.errstate(all="ignore"):
        posterior = -0.5 * np.sum(d2 + np.log(2.0 * np.pi * e2)) + prior_value.item()
    return posterior if np.isfinite(posterior) else -np.inf


def log_dummy_prior(*args):
    """
    Default log(prior) returns zero.

    """

    return 0.0


@_safe_log_probability
def log_likelihood(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2(
        _x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_distance(
    _x,
    obs,
    errors,
    interpolator_filter,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided (for emcee).

    """

    d2, e2 = diff2_distance(
        _x,
        obs,
        errors,
        interpolator_filter,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_distance_red_filter(
    _x,
    obs,
    errors,
    interpolator_filter,
    interpolator_teff,
    logg_pos,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_filter(
        _x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg_pos,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_distance_red_filter_fixed_logg(
    _x,
    obs,
    errors,
    interpolator_filter,
    interpolator_teff,
    logg,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_filter_fixed_logg(
        _x,
        obs,
        errors,
        interpolator_filter,
        interpolator_teff,
        logg,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_distance_red_interpolated(
    _x,
    obs,
    errors,
    interpolator_filter,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_interpolated(
        _x,
        obs,
        errors,
        interpolator_filter,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_distance_red_interpolated_fixed_logg(
    _x,
    obs,
    errors,
    interpolator_filter,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_interpolated_fixed_logg(
        _x,
        obs,
        errors,
        interpolator_filter,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_red_filter(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    interpolator_teff,
    logg_pos,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_red_filter(
        _x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg_pos,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_red_filter_fixed_logg(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    interpolator_teff,
    logg,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_red_filter_fixed_logg(
        _x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        interpolator_teff,
        logg,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)


@_safe_log_probability
def log_likelihood_red_interpolated(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    z_min,
    z_max,
    prior,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_red_interpolated(
        _x,
        obs,
        errors,
        distance,
        distance_err,
        interpolator_filter,
        rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        z_min,
        z_max,
        True,
        photometry_space=photometry_space,
    )
    return _log_posterior(_x, d2, e2, prior)
