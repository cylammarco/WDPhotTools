#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""For computing theorectical WDLFs"""

import warnings
import os

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from .atmosphere_model_reader import AtmosphereModelReader
from .cooling_model_reader import CoolingModelReader
from .util import load_ms_lifetime_datatable


class WDLF(AtmosphereModelReader, CoolingModelReader):
    """
    Computing the theoretical WDLFs based on the input IFMR, WD cooling and
    MS lifetime models.

    We are using little m for WD mass and big M for MS mass throughout this
    package.

    All the models are reporting in different set of units. They are all
    converted by the formatter to this set of units: (1) mass is in solar mass,
    (2) luminosity is in erg/s, (3) time/age is in year.

    For conversion, we use (1) M_sun = 1.98847E30 and (2) L_sun = 3.826E33.

    """

    def __init__(
        self,
        imf_model="C03",
        ifmr_model="C08",
        low_mass_cooling_model="montreal_co_da_20",
        intermediate_mass_cooling_model="montreal_co_da_20",
        high_mass_cooling_model="montreal_co_da_20",
        ms_model="PARSECz0017",
    ):
        super(WDLF, self).__init__()

        self.cooling_interpolator = None
        self.wdlf_params = {
            "imf_model": None,
            "ifmr_model": None,
            "sfr_mode": None,
            "ms_model": None,
        }

        self.imf_model_list = ["K01", "C03", "C03b", "manual"]

        self.ifmr_model_list = [
            "C08",
            "C08b",
            "S09",
            "S09b",
            "W09",
            "K09",
            "K09b",
            "C18",
            "EB18",
            "manual",
        ]

        self.sfr_mode_list = ["constant", "burst", "decay", "manual"]

        # Map ms_model values to CSV filenames
        self.ms_model_to_file = {
            # PARSEC - https://people.sissa.it/~sbressan/parsec.html
            "PARSECz00001": "PARSECz00001.csv",
            "PARSECz00002": "PARSECz00002.csv",
            "PARSECz00005": "PARSECz00005.csv",
            "PARSECz0001": "PARSECz0001.csv",
            "PARSECz0002": "PARSECz0002.csv",
            "PARSECz0004": "PARSECz0004.csv",
            "PARSECz0006": "PARSECz0006.csv",
            "PARSECz0008": "PARSECz0008.csv",
            "PARSECz001": "PARSECz001.csv",
            "PARSECz0014": "PARSECz0014.csv",
            "PARSECz0017": "PARSECz0017.csv",
            "PARSECz002": "PARSECz002.csv",
            "PARSECz003": "PARSECz003.csv",
            "PARSECz004": "PARSECz004.csv",
            "PARSECz006": "PARSECz006.csv",
            # GENEVA - https://obswww.unige.ch/Research/evol/tables_grids2011/
            "GENEVAz014": "geneva2011z014.csv",
            "GENEVAz006": "geneva2011z006.csv",
            "GENEVAz002": "geneva2011z002.csv",
            # MIST - http://waps.cfa.harvard.edu/MIST/
            "MISTFe050": "MISTv1p2Fe050.csv",
            "MISTFe025": "MISTv1p2Fe025.csv",
            "MISTFe000": "MISTv1p2Fe000.csv",
            "MISTFem025": "MISTv1p2Fem025.csv",
            "MISTFem050": "MISTv1p2Fem050.csv",
            "MISTFem075": "MISTv1p2Fem075.csv",
            "MISTFem100": "MISTv1p2Fem100.csv",
            "MISTFem125": "MISTv1p2Fem125.csv",
            "MISTFem150": "MISTv1p2Fem150.csv",
            "MISTFem175": "MISTv1p2Fem175.csv",
            "MISTFem200": "MISTv1p2Fem200.csv",
            "MISTFem250": "MISTv1p2Fem250.csv",
            "MISTFem300": "MISTv1p2Fem300.csv",
            "MISTFem350": "MISTv1p2Fem350.csv",
            "MISTFem400": "MISTv1p2Fem400.csv",
        }

        # The IFMR, WD cooling and MS lifetime models are required to
        # initialise the object.
        self.set_imf_model(imf_model)
        self.set_ifmr_model(ifmr_model)
        self.set_low_mass_cooling_model(low_mass_cooling_model)
        self.set_intermediate_mass_cooling_model(intermediate_mass_cooling_model)
        self.set_high_mass_cooling_model(high_mass_cooling_model)
        self.set_ms_model(ms_model)
        self.set_sfr_model()
        self._update_filename()

        self.mag = None
        self.mag_to_mbol_itp = None
        self.number_density = None

    def _update_filename(self):
        self._filename_middle = (
            f"_{self.wdlf_params['sfr_mode']}"
            f"_{self.wdlf_params['ms_model']}"
            f"_{self.wdlf_params['ifmr_model']}"
            f"_{self.cooling_models['low_mass_cooling_model']}"
            f"_{self.cooling_models['intermediate_mass_cooling_model']}"
            f"_{self.cooling_models['high_mass_cooling_model']}."
        )

    def _find_mass_ms_min(self, mass_ms, mag):
        """
        A function to be minimised to find the minimum mass limit that a MS star could have turned into a WD in the
        given age of the population (which is given by the SFR).

        Parameters
        ----------
        mass_ms: float
            MS mass.
        logL: float
            log WD luminosity.

        Return
        ------
        The difference between the total time and the sum of the cooling time and main sequence lifetime.

        """

        # Get the WD mass
        mass = self._ifmr(mass_ms)

        # Get the bolometric magnitude
        mbol = self.mag_to_mbol_itp(mass, mag)
        if mbol == -np.inf:
            return np.inf

        logL = (4.75 - mbol) / 2.5 + 33.582744965691276

        # Get the cooling age from the WD mass and the luminosity
        t_cool = self.cooling_interpolator(logL, mass)
        if t_cool <= 0.0:
            return np.inf

        # Get the MS life time
        t_ms = self._ms_model(mass_ms)
        if t_ms <= 0.0:
            return np.inf

        # Time since star formation
        time = self.t_start - t_cool - t_ms

        if time < 0.0:
            return np.inf

        else:
            return mass_ms**2.0

    def _integrand(self, mass_ms, mag):
        """
        The integrand of the number density computation based on the pre-selected (1) MS lifetime model, (2) initial
        mass function, (3) initial-final mass relation, and (4) WD cooling model.

        Parameters
        ----------
        M: float
            Main sequence stellar mass
        mag: float
            Absolute magnitude in a given passband
        T0: float
            Look-back time
        passband: str (Default: mbol)
            passband to be integrated in

        Return
        ------
        The product for integrating to the number density.

        """
        # Get the WD mass
        mass = self._ifmr(mass_ms)

        # Get the mass function
        mass_function = self._imf(mass_ms)

        mbol = self.mag_to_mbol_itp(mass, mag)
        logL = (4.75 - mbol) / 2.5 + 33.582744965691276

        # Get the WD cooling time
        t_cool = self.cooling_interpolator(logL, mass)

        # Get the MS lifetime
        t_ms = self._ms_model(mass_ms)

        # Get the time since star formation
        # and then the SFR
        sfr = self.sfr(t_cool + t_ms)

        # Get the cooling rate
        dLdt = -self.cooling_rate_interpolator(logL, mass)

        # To reduce the dimension from array (of size 1) to float
        total_contribution = np.sum(mass_function * sfr * dLdt)

        if np.isfinite(total_contribution):
            if total_contribution < 0.0:
                return 0.0

            return total_contribution

        else:
            return 0.0

    def set_sfr_model(
        self,
        mode="constant",
        age=10e9,
        duration=1e9,
        mean_lifetime=3e9,
        sfr_model=None,
    ):
        """
        Set the SFR scenario, we only provide a few basic forms, free format can be supplied as a callable function
        through sfr_model.

        The SFR function accepts the time in unit of year, which is the lookback time (i.e. today is 0, age of the
        universe is ~13.8E9).

        For burst and constant SFH, tophat functions are used:

            - t1 is the beginning of the star burst
            - t2 is the end
            - t0 and t3 are tiny deviations from t1 and t2 required for
              interpolation

        >>>    SFR
        >>>    ^                x-------x
        >>>    |                |       |
        >>>    |                |       |
        >>>    |    x-----------x       x-----------------x
        >>>        -30E9   0   t3/t2    t1/t0   13.8E9   30E9
        >>>                Lookback Time

        Parameters
        ----------
        mode: str (Default: 'constant')
            Choice of SFR mode:

                1. constant
                2. burst
                3. decay
                4. manual

        age: float (Default: 10E9)
            Lookback time in unit of years.
        duration: float (Default: 1E9)
            Duration of the starburst, only used if mode is 'burst'.
        mean_lifetime: float (Default: 3E9)
            Only used if mode is 'decay'. The default value has a SFR mean lifetime of 3 Gyr (SFR dropped by a factor
            of e after 3 Gyr).
        sfr_model: callable function (Default: None)
            The free-form star formation rate, in unit of years. If not callable, it reverts to using a constant star
            formation rate. It is necessary to fill in the age argument.

        """

        if mode not in self.sfr_mode_list:
            raise ValueError("Please provide a valid SFR mode.")

        else:
            if mode == "manual":
                if callable(sfr_model):
                    self.sfr = sfr_model

                else:
                    warnings.warn(
                        "The sfr_model provided is not callable, None is applied, i.e. constant star fomration."
                    )
                    mode = "constant"

            elif mode == "constant":
                t_1 = age
                t_0 = t_1 * 1.00001
                # current time = 0.
                t_2 = 0.0
                t_3 = t_2 * 0.99999

                self.sfr = interp1d(
                    np.array((30e9, t_0, t_1, t_2, t_3, -30e9)),
                    np.array((0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
                    fill_value="extrapolate",
                )

            elif mode == "burst":
                t_1 = age
                t_0 = t_1 * 1.00001
                t_2 = t_1 - duration
                t_3 = t_2 * 0.99999

                self.sfr = interp1d(
                    np.array((30e9, t_0, t_1, t_2, t_3, -30e9)),
                    np.array((0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
                    fill_value="extrapolate",
                )

            else:
                _t = 10.0 ** np.linspace(0, np.log10(age), 10000)
                _sfr = np.exp((_t - age) / mean_lifetime)

                self.sfr = interp1d(_t, _sfr, bounds_error=False, fill_value=0.0)

        self.t_start = age
        self.wdlf_params["sfr_mode"] = mode
        self._update_filename()

    def set_imf_model(self, model, imf_function=None):
        """
        Set the initial mass function.

        Parameters
        ----------
        model: str (Default: 'C03')
            Choice of IFMR model:

                1. K01 - Kroupa 2001
                2. C03 - Charbrier 2003
                3. C03b - Charbrier 2003 (including binary)
                4. manual

        imf_function: callable function (Default: None)
            A callable imf function, only used if model is 'manual'.

        """

        if model in self.imf_model_list:
            self.wdlf_params["imf_model"] = model

        else:
            raise ValueError("Please provide a valid Imass_function model.")

        if model == "K01":

            # mass lower than 0.08 is impossible, so that range is ignored.
            # (0.5**-2.3) / (0.5**-1.3) = 2.0
            def imf_func(m):
                m = np.ravel(m).astype(float)
                mf = m**-2.3
                mask = m < 0.5
                if mask.any():
                    mf[mask] = m[mask] ** -1.3 * 2.0  # normalization factor
                return mf

            self._imf = imf_func

        elif model == "C03":

            def imf_func(m):
                m = np.ravel(m).astype(float)
                mf = m**-2.3
                mask = m < 1.0
                if mask.any():
                    # 0.158 / (ln(10) * mass_ms) = 0.06861852814 / mass_ms
                    # log(0.079) = -1.1023729087095586
                    # 2 * 0.69**2. = 0.9522
                    # Normalisation factor (at mass_ms=1) is 0.01915058
                    norm = 0.01915058
                    factor = 0.06861852814
                    logconst = 1.1023729087095586
                    sigma_sq = 0.9522
                    mf[mask] = (factor / m[mask]) * np.exp(-((np.log10(m[mask]) + logconst) ** 2) / sigma_sq) / norm
                return mf

            self._imf = imf_func

        elif model == "C03b":

            def imf_func(m):
                m = np.ravel(m).astype(float)
                mf = m**-2.3
                mask = m <= 1.0
                if mask.any():
                    # 0.086 * 1. / (ln(10) * M) = 0.03734932544 / M
                    # log(0.22) = -0.65757731917
                    # 2 * 0.57**2. = 0.6498
                    # Normalisation factor (at M=1) is 0.01919917
                    norm = 0.01919917
                    factor = 0.03734932544
                    logconst = 0.65757731917
                    sigma_sq = 0.6498
                    mf[mask] = (factor / m[mask]) * np.exp(-((np.log10(m[mask]) + logconst) ** 2) / sigma_sq) / norm
                return mf

            self._imf = imf_func

        else:
            if not callable(imf_function):
                raise RuntimeError("imf_function provided is not callable.")
            self._imf = imf_function

        self._update_filename()

    def set_ms_model(self, model, ms_function=None):
        """
        Set the total stellar evolution lifetime model.

        Parameters
        ----------
        model: str (Default: 'PARSECz0017')
            Choice of MS model are from the PARSEC, Geneva and MIST stellar evolution models. The complete list of
            available models is as follow:

                1. PARSECz00001 - Z = 0.0001, Y = 0.249
                2. PARSECz00002 - Z = 0.0002, Y = 0.249
                3. PARSECz00005 - Z = 0.0005, Y = 0.249
                4. PARSECz0001 - Z = 0.001, Y = 0.25
                5. PARSECz0002 - Z = 0.002, Y = 0.252
                6. PARSECz0004 - Z = 0.004, Y = 0.256
                7. PARSECz0006 - Z = 0.006, Y = 0.259
                8. PARSECz0008 - Z = 0.008, Y = 0.263
                9. PARSECz001 - Z = 0.01, Y = 0.267
                10. PARSECz0014 - Z = 0.014, Y = 0.273
                11. PARSECz0017 - Z = 0.017, Y = 0.279
                12. PARSECz002 - Z = 0.02, Y = 0.284
                13. PARSECz003 - Z = 0.03, Y = 0.302
                14. PARSECz004 - Z = 0.04, Y = 0.321
                15. PARSECz006 - Z = 0.06, Y = 0.356
                16. GENEVAz002 - Z = 0.002
                17. GENEVAz006 - Z = 0.006
                18. GENEVAz014 - Z = 0.014
                19. MISTFem400 - [Fe/H] = -4.0
                20. MISTFem350 - [Fe/H] = -3.5
                21. MISTFem300 - [Fe/H] = -3.0
                22. MISTFem250 - [Fe/H] = -2.5
                23. MISTFem200 - [Fe/H] = -2.0
                24. MISTFem175 - [Fe/H] = -1.75
                25. MISTFem150 - [Fe/H] = -1.5
                26. MISTFem125 - [Fe/H] = -1.25
                27. MISTFem100 - [Fe/H] = -1.0
                28. MISTFem075 - [Fe/H] = -0.75
                29. MISTFem050 - [Fe/H] = -0.5
                30. MISTFem025 - [Fe/H] = -0.25
                31. MISTFe000 - [Fe/H] = 0.0
                32. MISTFe025 - [Fe/H] = 0.25
                33. MISTFe050 - [Fe/H] = 0.5

        ms_function: callable function (Default: None)
            A callable MS lifetime function, only used if model is 'manual'.

        """

        if model in self.ms_model_to_file:

            self.wdlf_params["ms_model"] = model
            datatable = load_ms_lifetime_datatable(self.ms_model_to_file[model])
            massi = np.array(datatable[:, 0]).astype(np.float64)
            time = np.array(datatable[:, 1]).astype(np.float64)
            self._ms_model = interp1d(massi, time, kind="cubic", fill_value="extrapolate")

        elif model == "manual":

            if callable(ms_function):

                self.wdlf_params["ms_model"] = model
                self._ms_model = ms_function

            else:

                raise RuntimeError(
                    "The ms_function provided is not callable, None is applied, i.e. no MS lifetime model."
                )

        else:

            raise ValueError(f"Unknown ms_model: {model}")

        self._update_filename()

    def set_ifmr_model(self, model, ifmr_function=None):
        """
        Set the initial-final mass relation (IFMR).

        Parameters
        ----------
        model: str (Default: 'EB18')
            Choice of IFMR model:

                1. C08 - Catalan et al. 2008
                2. C08b - Catalan et al. 2008 (two-part)
                3. S09 - Salaris et al. 2009
                4. S09b - Salaris et al. 2009 (two-part)
                5. W09 - Williams, Bolte & Koester 2009
                6. K09 - Kalirai et al. 2009
                7. K09b - Kalirai et al. 2009 (two-part)
                8. C18 - Cummings et al. 2018
                9. EB18 - El-Badry et al. 2018
                10. manual

        ifmr_function: callable function (Default: None)
            A callable ifmr function, only used if model is 'manual'.

        """

        # Check if the model is in the list of available models
        if model in self.ifmr_model_list:
            self.wdlf_params["ifmr_model"] = model

        else:
            raise ValueError("Please provide a valid IFMR mode.")

        # Set the IFMR function based on the model
        if model == "C08":
            self._ifmr = lambda m: np.maximum(0.117 * np.ravel(m) + 0.384, 0.4349)

        elif model == "C08b":

            def func(m):
                m = np.ravel(m).astype(float)
                mass = 0.096 * m + 0.429
                mask = m >= 2.7
                mass[mask] = 0.137 * m[mask] + 0.318
                return np.maximum(mass, 0.4746)

            self._ifmr = func

        elif model == "S09":
            self._ifmr = lambda m: np.maximum(0.084 * np.ravel(m) + 0.466, 0.5088)

        elif model == "S09b":

            def func(m):
                m = np.ravel(m).astype(float)
                mass = np.empty_like(m)
                mask = m < 4.0
                mass[mask] = 0.134 * m[mask] + 0.331
                mass[~mask] = 0.047 * m[~mask] + 0.679
                return np.maximum(mass, 0.3823)

            self._ifmr = func

        elif model == "W09":
            self._ifmr = lambda m: np.maximum(0.129 * np.ravel(m) + 0.339, 0.3893)

        elif model == "K09":
            self._ifmr = lambda m: np.maximum(0.109 * np.ravel(m) + 0.428, 0.4804)

        elif model == "K09b":
            self._ifmr = lambda m: np.maximum(0.101 * np.ravel(m) + 0.463, 0.4804)

        elif model == "C18":
            f = interp1d(
                (0.83, 2.85, 3.60, 7.20),
                (0.5554, 0.71695, 0.8572, 1.2414),
                fill_value="extrapolate",
                bounds_error=False,
            )
            self._ifmr = lambda m: f(np.ravel(m))

        elif model == "EB18":
            f = interp1d(
                (0.95, 2.75, 3.54, 5.21, 8.0),
                (0.5, 0.67, 0.81, 0.91, 1.37),
                fill_value="extrapolate",
                bounds_error=False,
            )
            self._ifmr = lambda m: f(np.ravel(m))

        else:
            # Fall back to user-supplied callable
            if not callable(ifmr_function):
                raise RuntimeError("Please provide a valid callable function")

            self._ifmr = ifmr_function

        self._update_filename()

    def compute_density(
        self,
        mag,
        passband="Mbol",
        atmosphere="H",
        interpolator="CT",
        mass_ms_max=8.0,
        limit=10000,
        n_points=50,
        epsabs=1e-6,
        epsrel=1e-6,
        normed=True,
        save_csv=False,
        folder=None,
        filename=None,
    ):
        """
        Compute the density based on the pre-selected models: (1) MS lifetime
        model, (2) initial mass function, (3) initial-final mass relation, and
        (4) WD cooling model. It integrates over the function _integrand().

        Parameters
        ----------
        mag: float or array of float
            Absolute magnitude in the given passband
        passband: str (Default: "Mbol")
            The passband to be integrated in.
        atmosphere: str (Default: H)
            The atmosphere type.
        interpolator: str (Default: CT)
            Choose between 'CT' and 'RBF.'
        mass_ms_max: float (Deafult: 8.0)
            The upper limit of the main sequence stellar mass. This may not be used if it exceeds the upper bound of
            the IFMR model.
        limit: int (Default: 10000)
            The maximum number of steps of integration
        n_points: int (Default: 100)
            The number of points for initial integration sampling, too small a value will lead to failed integration
            because it can underestimate the density if the star formation periods are short. While too large a value
            will lead to low performance due to oversampling, though the accuracy is guaranteed. The default value is
            sufficient to compute WDLF for star burst as short as 1E8 years. For burst as short as 1E7, we recommand
            an n_points of 10000 or larger.
        epsabs: float (Default: 1e-6)
            The absolute tolerance of the integration step. For star burst, we recommend a step smaller than 1e-8.
        epsrel: float (Default: 1e-6)
            The relative tolerance of the integration step. For star burst, we recommend a step smaller than 1e-8.
        normed: boolean (Default: True)
            Set to True to return a WDLF sum to 1. Otherwise, it is arbitrary to the integrator.
        save_csv: boolean (Default: False)
            Set to True to save the WDLF as CSV files. One CSV per T0.
        folder: str (Default: None)
            The relative or absolute path to destination, the current working directory will be used if None.
        filename: str (Default: None)
            The filename of the csv. The default filename will be used if None.

        Returns
        -------
        mag: array of float
            The magnitude at which the number density is computed.
        number_density: array of float
            The (arbitrary) number density at that magnitude.

        """

        if self.cooling_interpolator is None:
            self.compute_cooling_age_interpolator(interpolator=interpolator)

        mag = np.asarray(mag).reshape(-1)

        number_density = np.zeros_like(mag)

        self.mag_to_mbol_itp = self.interp_am(
            dependent="Mbol",
            atmosphere=atmosphere,
            independent=["mass", passband],
            interpolator=interpolator,
        )
        # perform a search for the limits of integration
        mass_grid = np.arange(0.4, mass_ms_max + 0.001, 0.001)  # fairly fine
        spacing = mass_grid[1] - mass_grid[0]
        v_min = None
        v_max = None
        for i, mag_i in enumerate(mag):

            # Find the lower mass limit for integration
            for m in mass_grid:
                mass_wd = self._ifmr(m)
                mbol = self.mag_to_mbol_itp(mass_wd, mag_i)
                if ~np.isfinite(mbol) | (mbol < -2.0) | (mbol > 20.0):
                    continue
                logL = (4.75 - mbol) / 2.5 + 33.582744965691276
                # Get the WD cooling time
                t_cool = self.cooling_interpolator(logL, mass_wd)
                if t_cool <= 0.0:
                    continue
                # Get the MS lifetime
                t_ms = self._ms_model(m)
                # get the SFR
                sfr = self.sfr(t_cool + t_ms)
                if sfr <= 0.0:
                    continue
                v_min = m
                break

            # Find the upper mass limit for integration
            for m in mass_grid[::-1]:
                mass_wd = self._ifmr(m)
                mbol = self.mag_to_mbol_itp(mass_wd, mag_i)
                if ~np.isfinite(mbol) | (mbol < -2.0) | (mbol > 20.0):
                    continue
                logL = (4.75 - mbol) / 2.5 + 33.582744965691276
                # Get the WD cooling time
                t_cool = self.cooling_interpolator(logL, mass_wd)
                if t_cool <= 0.0:
                    continue
                # Get the MS lifetime
                t_ms = self._ms_model(m)
                # get the SFR
                sfr = self.sfr(t_cool + t_ms)
                if sfr <= 0.0:
                    continue
                v_max = m
                break

            # If we have found both limits, compute the number density
            if v_min != v_max:
                valid_points = np.linspace(v_min - spacing, v_max + spacing, n_points)
                number_density[i] = integrate.quad(
                    self._integrand,
                    valid_points[0],
                    valid_points[-1],
                    args=[mag_i],
                    limit=limit,
                    points=valid_points,
                    epsabs=epsabs,
                    epsrel=epsrel,
                )[0]

        number_density[np.isnan(number_density) | (number_density <= 0.0)] = +0.0

        # Normalise the WDLF only if the function returned is not all zero
        if normed & (number_density > 0.0).any():
            number_density /= np.nansum(number_density)

        self.mag = mag
        self.number_density = number_density

        if save_csv:
            if folder is None:
                _folder = os.getcwd()

            else:
                _folder = os.path.abspath(folder)

            if filename is None:
                _filename = f"{self.t_start / 1e9:.3f}Gyr" f"{self._filename_middle}csv"

            else:
                _filename = filename

            np.savetxt(
                os.path.join(_folder, _filename),
                np.column_stack((mag, number_density)),
                delimiter=",",
            )

        return mag, number_density

    def plot_input_models(
        self,
        figsize=(15, 15),
        title=None,
        display=True,
        savefig=False,
        folder=None,
        filename=None,
        ext=["png"],
        sfh_log=False,
        imf_log=True,
        ms_time_log=True,
        cooling_model_use_mag=True,
        **kwargs,
    ):
        """
        Plot the input cooling model.

        Parameters
        ----------
        use_mag: bool (Default: True)
            Set to use magnitude instead of luminosity
        figsize: array of size 2 (Default: (12, 8))
            Set the dimension of the figure.
        title: str (Default: None)
            Set the title of the figure.
        display: bool (Default: True)
            Set to display the figure.
        savefig: bool (Default: False)
            Set to save the figure.
        folder: str (Default: None)
            The relative or absolute path to destination, the current working directory will be used if None.
        filename: str (Default: None)
            The filename of the figure. The default filename will be used if None.
        ext: str (Default: ['png'])
            Image type to be saved, multiple extensions can be provided. The supported types are those available in
            `matplotlib.pyplot.savefig`.
        sfh_log: bool (Default: False)
            Set to plot the SFH in logarithmic space
        imf_log: bool (Default: False)
            Set to plot the Imass_function in logarithmic space
        ms_time_log: bool (Default: True)
            Set to plot the MS lifetime in logarithmic space
        cooling_model_use_mag: bool (Default: True)
            Set to plot the Cooling model in logarithmic space
        fig: matplotlib.figure.Figure (Default: None)
            Overplotting on an existing Figure.
        kwargs: dict (Default: {})
            Keyword arguments for the colorbar()

        """

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=figsize)
        # top row
        ax1 = axs[0, 0]  # Initial Mass Function
        ax2 = axs[0, 1]  # Star Formation History
        # middle row
        ax3 = axs[1, 0]  # MS lifetime
        ax4 = axs[1, 1]  # Initial-Final Mass Relation
        # bottom row
        ax5 = axs[2, 0]  # Cooling Model: Mobl(t) or L(t)
        ax6 = axs[2, 1]  # Cooling Model: d(Mobl)/d(t) or d(L)/d(t)

        #
        # Initial Mass Function
        #
        mass = np.linspace(0.25, 8.25, 1000)

        if imf_log:
            ax1.plot(mass, np.log10(self._imf(mass)))
            ax1.set_ylabel("log(Imass_function)")

        else:
            ax1.plot(mass, self._imf(mass))
            ax1.set_ylabel("Imass_function")

        ax1.set_xlabel(r"Mass / M$_\odot$")
        ax1.set_xlim(0.25, 8.25)
        ax1.grid()

        ax1.set_title("Initial Mass Function")

        #
        # Star formation History
        #
        _t = np.linspace(0, self.t_start, 1000)
        ax2.plot(_t / 1e9, self.sfr(_t))

        if sfh_log:
            ax2.set_yscale("log")
            ax2.set_ylabel("log(Relative SFR)")

        else:
            ax2.set_ylabel("Relative SFR")

        ax2.set_xlabel("Look-back Time / Gyr")
        ax2.set_title("Star Formation History")
        ax2.grid()

        #
        # Main Sequence Lifetime
        #
        ax3.plot(mass, self._ms_model(mass))

        if ms_time_log:
            ax3.set_yscale("log")
            ax3.set_ylabel("log(MS Lifetime / yr)")

        else:
            ax3.set_ylabel("MS Lifetime / yr")

        ax3.set_xlabel(r"ZAMS Mass / M$_\odot$")
        ax3.set_title("MS Lifetime")
        ax3.grid()

        #
        # Initial-Final Mass Relation
        #
        ax4.plot(mass, self._ifmr(mass))
        ax4.set_ylabel(r"Final Mass / M$_\odot$")
        ax4.set_xlabel(r"Initial Mass / M$_\odot$")
        ax4.set_xlim(0.25, 8.25)
        ax4.grid()

        ax4.set_title("Initial-Final Mass Relation")

        #
        # Cooling Model : Mobl(t) or L(t)
        #
        if cooling_model_use_mag:
            # Get absolute magnitude from the bolometric luminosity
            brightness = 4.75 - (np.log10(self.luminosity) - 33.582744965691276) * 2.5

        else:
            brightness = self.luminosity

        sc5 = ax5.scatter(self.age / 1e9, brightness, c=self.mass, s=5)

        # colorbar
        cbar5 = plt.colorbar(mappable=sc5, ax=ax5, **kwargs)
        cbar5.ax.set_ylabel("Solar Mass", rotation=270, labelpad=15)

        # y axis
        if cooling_model_use_mag:
            ax5.set_ylabel(r"M$_{\mathrm{bol}}$ / mag")

        else:
            ax5.set_ylabel(r"L$_{\mathrm{bol}}$")
            ax5.set_yscale("log")

        ax5.set_ylim(np.nanmin(brightness), np.nanmax(brightness))

        # x axis
        ax5.set_xlabel(r"Age / Gyr")
        ax5.set_xlim(0.0, 16.0)

        ax5.grid()
        ax5.set_title("Cooling Model")

        #
        # Cooling Model: d(mbol)/d(t) or d(L)/d(t)
        #
        if cooling_model_use_mag:
            # 2.5 * 1e9 * (365.25 * 24. * 60. * 60.) / np.log(10) =
            # 3.426322886e16
            rate_of_change = -3.426322886e16 / self.luminosity * self.dLdt

        else:
            rate_of_change = self.dLdt * -1.0

        rate_of_change[np.isnan(rate_of_change)] = 0.0
        rate_of_change[~np.isfinite(rate_of_change)] = 0.0

        sc6 = ax6.scatter(self.age / 1e9, rate_of_change, c=self.mass, s=5)
        cbar6 = plt.colorbar(mappable=sc6, ax=ax6, **kwargs)
        cbar6.ax.set_ylabel("Solar Mass", rotation=270, labelpad=15)

        # y axis
        if cooling_model_use_mag:
            ax6.set_ylabel(r"d(M$_{\mathrm{bol}})/dt (Gyr)$")
            ax6.set_ylim(-0.005, np.nanmax(rate_of_change) * 0.6)

        else:
            ax6.set_ylabel(r"-d(L$_{\mathrm{bol}})/dt (s)$")
            ax6.set_yscale("log")
            ax6.set_ylim(np.nanmin(rate_of_change), np.nanmax(rate_of_change))

        # x axis
        ax6.set_xlabel(r"Age / Gyr")
        ax6.set_xlim(0.0, 16.0)

        ax6.grid()
        ax6.set_title("Cooling Rate")

        plt.subplots_adjust(
            top=0.95,
            bottom=0.075,
            left=0.075,
            right=0.99,
            hspace=0.4,
            wspace=0.225,
        )

        if title is not None:
            plt.suptitle(title)

        if savefig:
            if isinstance(ext, str):
                ext = [ext]

            if folder is None:
                _folder = os.getcwd()

            else:
                _folder = os.path.abspath(folder)

                if not os.path.exists(_folder):
                    os.makedirs(_folder)

            # Loop through the ext list to save figure into each image type
            for _e in ext:
                if filename is None:
                    _filename = "input_model." + _e

                else:
                    _filename = filename + "." + _e

                plt.savefig(os.path.join(_folder, _filename))

        if display:
            plt.show()

        return fig

    def plot_wdlf(
        self,
        log=True,
        figsize=(12, 8),
        title=None,
        display=True,
        savefig=False,
        folder=None,
        filename=None,
        ext=["png"],
        fig=None,
    ):
        """
        Plot the input Initial-Final Mass Relation.

        Parameters
        ----------
        log: bool (Default: True)
            Set to plot the WDLF in logarithmic space
        figsize: array of size 2 (Default: (12, 8))
            Set the dimension of the figure.
        title: str (Default: None)
            Set the title of the figure.
        display: bool (Default: True)
            Set to display the figure.
        savefig: bool (Default: False)
            Set to save the figure.
        folder: str (Default: None)
            The relative or absolute path to destination, the current working directory will be used if None.
        filename: str (Default: None)
            The filename of the figure. The default filename will be used if None.
        ext: str (Default: ['png'])
            Image type to be saved, multiple extensions can be provided. The supported types are those available in
            `matplotlib.pyplot.savefig`.
        fig: matplotlib.figure.Figure (Default: None)
            Overplotting on an existing Figure.

        """

        if fig is None:
            fig = plt.figure(figsize=figsize)

        _density = self.number_density

        plt.plot(self.mag, _density, label=f"{self.t_start / 1e90:.3f} Gyr")
        plt.xlim(0, 20)
        plt.xlabel(r"M$_{\mathrm{bol}}$ / mag")

        _density_finite = _density[np.isfinite(_density) & (_density > 0.0)]

        # If there is nothing to plot...
        if len(_density_finite) == 0:
            return 0

        ymin = np.floor(np.nanmin(_density_finite))
        ymax = np.ceil(np.nanmax(_density_finite))
        plt.ylim(ymin, ymax)
        plt.ylabel(r"$\log{(N)}$")

        if log:
            plt.yscale("log")

        plt.grid()
        plt.legend()

        if title is None:
            title = f"WDLF: {self.t_start / 1e9:.3f} Gyr"

        plt.title(title)
        plt.tight_layout()

        if savefig:
            if isinstance(ext, str):
                ext = [ext]

            if folder is None:
                _folder = os.getcwd()

            else:
                _folder = os.path.abspath(folder)

                if not os.path.exists(_folder):
                    os.makedirs(_folder)

            # Loop through the ext list to save figure into each image type
            for _e in ext:
                if filename is None:
                    _filename = f"{self.t_start / 1e9:.3f}Gyr{self._filename_middle}{_e}"

                else:
                    _filename = filename + "." + _e

                plt.savefig(os.path.join(_folder, _filename))

        if display:
            plt.show()

        return fig
