import numpy as np

from .diff2_functions_least_square import (
    diff2,
    diff2_distance,
    diff2_red_filter,
    diff2_red_filter_fixed_logg,
    diff2_red_interpolated,
    diff2_distance_red_filter,
    diff2_distance_red_filter_fixed_logg,
    diff2_distance_red_interpolated,
    diff2_distance_red_interpolated_fixed_logg,
)


def diff2_summed(_x, obs, errors, distance, distance_err, interpolator_filter, return_err):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.minimize).

    """

    d2, e2 = diff2(_x, obs, errors, distance, distance_err, interpolator_filter, True)

    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_red_filter_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )

    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_red_filter_fixed_logg_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.minimize).

    """
    d2, e2 = diff2_red_filter_fixed_logg(
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
        zmin,
        zmax,
        True,
    )

    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_red_interpolated_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )
    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_distance_red_filter_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )
    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_distance_red_filter_fixed_logg_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )
    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_distance_summed(_x, obs, errors, interpolator_filter, return_error):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

    """

    d2, e2 = diff2_distance(_x, obs, errors, interpolator_filter, True)

    if return_error:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_distance_red_interpolated_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )

    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)


def diff2_distance_red_interpolated_fixed_logg_summed(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.minimize).

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
        zmin,
        zmax,
        True,
    )

    if return_err:
        return np.sum(d2), 1.0 / np.sum(1.0 / e2)

    else:
        return np.sum(d2)
