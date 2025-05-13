import astropy.coordinates as coord
import astropy.units as u


def get_extinction_fraction(distance, ra, dec, z_min, z_max):
    """
    The linear mode follows the scheme on page 5 of Harris et al. (2006)
    in https://arxiv.org/pdf/astro-ph/0510820.pdf.

    The conversion from distance, ra and dec to z is powered by AstroPy coordinate transformation.

    Parameters
    ----------
    distance : float
        Distance to the target (in unit of pc)
    ra : float
        Right Ascension in unit of degree.
    dec : float
        Declination in unit of degree.

    Returns
    -------
    The fraction (of extinction) should be used, in the range of [0.0, 1.0].

    """

    _c = coord.SkyCoord(
        ra=ra * u.degree,
        dec=dec * u.degree,
        distance=distance * u.pc,
        frame="icrs",
    )
    c_gal_cen = _c.transform_to(coord.Galactocentric)

    if (z_min is None) or (z_max is None):
        raise ValueError(
            "z_min and z_max cannot be None, please initialise with set_extinction_mode()"
        )

    else:
        # Get the distance from the Galactic mid-plane
        _z = getattr(c_gal_cen, "z").value

        # if z is lower than the lower limit, assume no extinction
        if _z < z_min:
            return 0.0

        # if z is higher than the upper limit, assume total extinction
        elif _z > z_max:
            return 1.0

        # Otherwise, apply a linear approximation of the extinction
        else:
            return (_z - z_min) / (z_max - z_min)
