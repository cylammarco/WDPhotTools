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


def log_likelihood(
    _x, obs, errors, distance, distance_err, interpolator_filter
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2(
        _x, obs, errors, distance, distance_err, interpolator_filter, True
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_distance(_x, obs, errors, interpolator_filter):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided (for emcee).

    """

    d2, e2 = diff2_distance(_x, obs, errors, interpolator_filter, True)

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_distance_red_filter(
    _x,
    obs,
    errors,
    interpolator_filter,
    interpolator_teff,
    logg_pos,
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
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
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_distance_red_filter_fixed_logg(
    _x,
    obs,
    errors,
    interpolator_filter,
    interpolator_teff,
    logg,
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
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
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_distance_red_interpolated(
    _x,
    obs,
    errors,
    interpolator_filter,
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_interpolated(
        _x,
        obs,
        errors,
        interpolator_filter,
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_distance_red_interpolated_fixed_logg(
    _x,
    obs,
    errors,
    interpolator_filter,
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
):
    """
    Internal method for computing the ch2-squared value (for emcee).

    """

    d2, e2 = diff2_distance_red_interpolated_fixed_logg(
        _x,
        obs,
        errors,
        interpolator_filter,
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

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
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
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
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

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
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
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
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf


def log_likelihood_red_interpolated(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    Rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
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
        Rv,
        extinction_mode,
        reddening_vector,
        ebv,
        ra,
        dec,
        True,
    )

    if np.isfinite(d2).all():
        return -0.5 * np.sum(d2 + np.log(2 * np.pi * e2))

    else:
        return -np.inf
