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


def log_dummy_prior(*args):
    """
    Default log(prior) returns zero.

    """

    return 0.0


def log_likelihood(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    prior,
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2(_x, obs, errors, distance, distance_err, interpolator_filter, True)
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


def log_likelihood_distance(_x, obs, errors, interpolator_filter, prior):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided (for emcee).

    """

    d2, e2 = diff2_distance(_x, obs, errors, interpolator_filter, True)
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf


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
    )
    p = prior(*_x)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2)) + p

    else:
        return -np.inf
