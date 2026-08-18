import numpy as np

from .extinction import get_extinction_fraction

_MAG_DISTANCE_FACTOR = 2.17147241
_MAG_TO_FRAC_FLUX_VAR = 0.8483036976765438


def _compute_residual_terms(
    obs,
    errors,
    model_mag,
    distance,
    distance_err,
    photometry_space,
):
    """Compute chi-square terms in either magnitude or relative flux space."""
    invalid_model = ~np.isfinite(model_mag)

    if photometry_space == "magnitude":
        if distance_err is None:
            e2 = errors**2.0
        else:
            # 5 / ln(10) converts fractional distance error to magnitude error.
            # (ln(10) / 2.5)^2 converts magnitude variance to fractional flux variance.
            e2 = (errors**2.0 + (distance_err / distance * _MAG_DISTANCE_FACTOR) ** 2.0) * _MAG_TO_FRAC_FLUX_VAR
        d2 = ((10.0 ** ((obs - model_mag) / 2.5) - 1.0) ** 2.0) / e2
        return np.where(invalid_model, np.inf, d2), e2

    if photometry_space == "flux":
        model_flux = 10.0 ** (-0.4 * model_mag)
        e2 = errors**2.0
        if distance_err is not None:
            # Relative flux follows d^-2, so sigma_f = 2 * sigma_d / d * f.
            e2 = e2 + (2.0 * distance_err / distance * model_flux) ** 2.0
        d2 = ((obs - model_flux) ** 2.0) / e2
        return np.where(invalid_model, np.inf, d2), e2

    raise ValueError("Unknown photometry_space. Please choose from 'magnitude' and 'flux'.")


def diff2(
    _x,
    obs,
    errors,
    distance,
    distance_err,
    interpolator_filter,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.least_squares).

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:2]))

    model_mag = np.asarray(mag).reshape(-1) + 5.0 * np.log10(distance) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=distance,
        distance_err=distance_err,
        photometry_space=photometry_space,
    )
    # Ensure finite residuals
    d2 = np.where(np.isfinite(d2), d2, np.float64(1e30))
    if return_err:
        e2 = np.where(np.isfinite(e2), e2, np.float64(1e30))
        return d2, e2
    else:
        return d2


def diff2_distance(
    _x,
    obs,
    errors,
    interpolator_filter,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided (for
    scipy.optimize.least_squares).

    """

    if (_x[-1] <= 0.0) or (_x[-1] > 10000.0):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x[:-1]))

    model_mag = np.asarray(mag).reshape(-1) + 5.0 * np.log10(_x[-1]) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=_x[-1],
        distance_err=None,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided.

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
        extinction_fraction = get_extinction_fraction(_x[-1], ra, dec, zmin, zmax)

    av = np.array([i(rv) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(_x[-1]) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=_x[-1],
        distance_err=None,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided.

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
        extinction_fraction = get_extinction_fraction(_x[-1], ra, dec, zmin, zmax)

    av = np.array([i(rv) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(_x[-1]) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=_x[-1],
        distance_err=None,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
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
        extinction_fraction = get_extinction_fraction(_x[-1], ra, dec, zmin, zmax)

    teff = float(np.asarray(interpolator_teff(_x[:2])).reshape(-1)[0])
    logg = _x[logg_pos]
    av = np.array([i([logg, teff, rv]) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(_x[-1]) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=_x[-1],
        distance_err=None,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value in cases when the distance is not provided.

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
        extinction_fraction = get_extinction_fraction(_x[-1], ra, dec, zmin, zmax)

    teff = float(np.asarray(interpolator_teff(_x[:-1])).reshape(-1)[0])
    av = np.array([i([logg, teff, rv]) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(_x[-1]) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=_x[-1],
        distance_err=None,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
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
        extinction_fraction = get_extinction_fraction(distance, ra, dec, zmin, zmax)

    av = np.array([i(rv) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(distance) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=distance,
        distance_err=distance_err,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
):
    """
    Internal method for computing the ch2-squared value (for scipy.optimize.least_square).

    """

    mag = []

    for interp in interpolator_filter:
        mag.append(interp(_x))

    teff = float(np.asarray(interpolator_teff(_x)).reshape(-1)[0])

    if not np.isfinite(teff):
        if return_err:
            return np.ones_like(obs) * np.inf, np.ones_like(obs) * np.inf

        else:
            return np.ones_like(obs) * np.inf

    if extinction_mode == "total":
        extinction_fraction = 1.0

    else:
        extinction_fraction = get_extinction_fraction(distance, ra, dec, zmin, zmax)

    logg = _x[logg_pos]
    av = np.array([i([logg, teff, rv]) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(distance) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=distance,
        distance_err=distance_err,
        photometry_space=photometry_space,
    )

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
    rv,
    extinction_mode,
    reddening_vector,
    ebv,
    ra,
    dec,
    zmin,
    zmax,
    return_err,
    photometry_space="magnitude",
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
        extinction_fraction = get_extinction_fraction(distance, ra, dec, zmin, zmax)

    teff = float(np.asarray(interpolator_teff(_x)).reshape(-1)[0])
    av = np.array([i([logg, teff, rv]) for i in reddening_vector]).reshape(-1) * ebv * extinction_fraction
    model_mag = np.asarray(mag).reshape(-1) + av + 5.0 * np.log10(distance) - 5.0
    d2, e2 = _compute_residual_terms(
        obs=obs,
        errors=errors,
        model_mag=model_mag,
        distance=distance,
        distance_err=distance_err,
        photometry_space=photometry_space,
    )

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
