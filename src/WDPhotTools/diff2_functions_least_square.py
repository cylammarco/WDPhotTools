import numpy as np

from .extinction import get_extinction_fraction


def diff2(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    return_err,
):
    """
    Internal method for computing the ch2-squared value
    (for scipy.optimize.least_squares).

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x))

    mag = np.asarray(mag).reshape(-1)

    e2 = (
        errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
    ) * 1.1788231063225867

    d2 = (
        (10.0 ** ((obs - mag - 5.0 * np.log10(distance) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_distance(_x, obs, errors, interpolator_filter, return_err):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided (for scipy.optimize.least_squares).

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:-1]))

    mag = np.asarray(mag).reshape(-1)
    e2 = errors**2.0

    d2 = (
        (10.0 ** ((obs - mag - 5.0 * np.log10(_x[-1]) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_distance_red_interpolated(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided.

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:2]))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            _x[-1], ra, dec, zmin, zmax
        )

    Av = (
        np.array([i(Rv) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = errors**2.0

    d2 = (
        (10.0 ** ((obs - Av - mag - 5.0 * np.log10(_x[-1]) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_distance_red_interpolated_fixed_logg(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided.

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:-1]))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            _x[-1], ra, dec, zmin, zmax
        )

    Av = (
        np.array([i(Rv) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = errors**2.0

    d2 = (
        (10.0 ** ((obs - Av - mag - 5.0 * np.log10(_x[-1]) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_distance_red_filter(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided.

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:2]))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            _x[-1], ra, dec, zmin, zmax
        )

    teff = float(interpolator_teff(_x[:2]))
    logg = _x[logg_pos]
    Av = (
        np.array([i([logg, teff, Rv]) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = errors**2.0

    d2 = (
        (10.0 ** ((obs - Av - mag - 5.0 * np.log10(_x[-1]) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_distance_red_filter_fixed_logg(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value in cases when
    the distance is not provided.

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:-1]))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            _x[-1], ra, dec, zmin, zmax
        )

    teff = float(interpolator_teff(_x[:-1]))
    Av = (
        np.array([i([logg, teff, Rv]) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = errors**2.0

    d2 = (
        (10.0 ** ((obs - Av - mag - 5.0 * np.log10(_x[-1]) + 5.0) / 2.5) - 1.0)
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_red_interpolated(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value.

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            distance, ra, dec, zmin, zmax
        )

    Av = (
        np.array([i(Rv) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = (
        errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
    ) * 1.1788231063225867

    d2 = (
        (
            10.0 ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
            - 1.0
        )
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_red_filter(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.least_square).

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x))

    teff = float(interpolator_teff(_x))

    if not np.isfinite(teff):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            distance, ra, dec, zmin, zmax
        )

    logg = _x[logg_pos]
    Av = (
        np.array([i([logg, teff, Rv]) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = (
        errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
    ) * 1.1788231063225867

    d2 = (
        (
            10.0 ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
            - 1.0
        )
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf


def diff2_red_filter_fixed_logg(
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
    zmin,
    zmax,
    return_err,
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.least_square).

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x))

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(
            distance, ra, dec, zmin, zmax
        )

    teff = float(interpolator_teff(_x))
    Av = (
        np.array([i([logg, teff, Rv]) for i in reddening_vector]).reshape(-1)
        * ebv
        * extinction_fraction
    )
    mag = np.asarray(mag).reshape(-1)
    e2 = (
        errors**2.0 + (distance_err / distance * 2.17147241) ** 2.0
    ) * 1.1788231063225867

    d2 = (
        (
            10.0 ** ((obs - Av - mag - 5.0 * np.log10(distance) + 5.0) / 2.5)
            - 1.0
        )
        ** 2.0
    ) / e2

    if np.isfinite(d2).all():
        if return_err:
            return d2, e2

        else:
            return d2

    else:
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs)
