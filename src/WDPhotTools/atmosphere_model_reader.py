#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Handling the formatting of different atmosphere models"""

import os

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RBFInterpolator


class AtmosphereModelReader(object):
    """Handling the formatting of different atmosphere models"""

    def __init__(self):
        super(AtmosphereModelReader, self).__init__()

        self.this_file = os.path.dirname(os.path.abspath(__file__))

        self.model_list = {
            "montreal_co_da_20": "Bedard et al. 2020 CO DA",
            "montreal_co_db_20": "Bedard et al. 2020 CO DB",
            "lpcode_he_da_07": "Panei et al. 2007 He DA",
            "lpcode_co_da_07": "Panei et al. 2007 CO DA",
            "lpcode_he_da_09": "Althaus et al. 2009 He DA",
            "lpcode_co_da_10_z001": "Renedo et al. 2010 CO DA Z=0.01",
            "lpcode_co_da_10_z0001": "Renedo et al. 2010 CO DA Z=0.001",
            "lpcode_co_da_15_z00003": "Althaus et al. 2015 DA Z=0.00003",
            "lpcode_co_da_15_z0001": "Althaus et al. 2015 DA Z=0.0001",
            "lpcode_co_da_15_z0005": "Althaus et al. 2015 DA Z=0.0005",
            "lpcode_co_db_17_z00005": "Althaus et al. 2017 DB Y=0.4",
            "lpcode_co_db_17_z0001": "Althaus et al. 2017 DB Y=0.4",
            "lpcode_co_db_17": "Camisassa et al. 2017 DB",
            "lpcode_one_da_07": "Althaus et al. 2007 ONe DA",
            "lpcode_one_da_19": "Camisassa et al. 2019 ONe DA",
            "lpcode_one_db_19": "Camisassa et al. 2019 ONe DB",
            "lpcode_da_22": "Althaus et al. 2013 He DA, "
            + "Camisassa et al. 2016 CO DA,  Camisassa et al. 2019 ONe DA",
            "lpcode_db_22": "Camisassa et al. 2017 CO DB, "
            + "Camisassa et al. 2019 ONe DB",
        }

        # DA atmosphere
        filepath_da = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "wd_photometry/Table_DA_13012021.txt",
        )

        # DB atmosphere
        filepath_db = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "wd_photometry/Table_DB_13012021.txt",
        )

        # Prepare the array column dtype
        self.column_key = np.array(
            [
                "Teff",
                "logg",
                "mass",
                "Mbol",
                "BC",
                "U",
                "B",
                "V",
                "R",
                "I",
                "J",
                "H",
                "Ks",
                "Y_mko",
                "J_mko",
                "H_mko",
                "K_mko",
                "W1",
                "W2",
                "W3",
                "W4",
                "S36",
                "S45",
                "S58",
                "S80",
                "u_sdss",
                "g_sdss",
                "r_sdss",
                "i_sdss",
                "z_sdss",
                "g_ps1",
                "r_ps1",
                "i_ps1",
                "z_ps1",
                "y_ps1",
                "G2",
                "G2_BP",
                "G2_RP",
                "G3",
                "G3_BP",
                "G3_RP",
                "FUV",
                "NUV",
                "age",
            ]
        )
        self.column_key_formatted = np.array(
            [
                r"T$_{\mathrm{eff}}$",
                "log(g)",
                "Mass",
                r"M$_{\mathrm{bol}}$",
                "BC",
                r"$U$",
                r"$B$",
                r"$V$",
                r"$R$",
                r"$I$",
                r"$J$",
                r"$H$",
                r"$K_{\mathrm{s}}$",
                r"$Y_{\mathrm{MKO}}$",
                r"$J_{\mathrm{MKO}}$",
                r"$H_{\mathrm{MKO}}$",
                r"$K_{\mathrm{MKO}}$",
                r"$W_{1}$",
                r"$W_{2}}$",
                r"$W_{3}$",
                r"$W_{4}$",
                r"$S_{36}$",
                r"$S_{45}$",
                r"$S_{58}$",
                r"$S_{80}$",
                r"u$_{\mathrm{SDSS}}$",
                r"$g_{\mathrm{SDSS}}$",
                r"$r_{\mathrm{SDSS}}$",
                r"$i_{\mathrm{SDSS}}$",
                r"$z_{\mathrm{SDSS}}$",
                r"$g_{\mathrm{PS1}}$",
                r"$r_{\mathrm{PS1}}$",
                r"$i_{\mathrm{PS1}}$",
                r"$z_{\mathrm{PS1}}$",
                r"$y_{\mathrm{PS1}}$",
                r"$G_{\mathrm{DR2}}$",
                r"$G_{\mathrm{BP, DR2}}$",
                r"$G_{\mathrm{RP, DR2}}$",
                r"$G{_{\mathrm{DR3}}$",
                r"$G_{\mathrm{BP, DR3}}$",
                r"$G_{\mathrm{RP, DR3}}$",
                "FUV",
                "NUV",
                "Age",
            ]
        )
        self.column_key_unit = np.array(
            [
                "K",
                r"(cm/s$^2$)",
                r"M$_\odot$",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
                "yr",
            ]
        )
        self.column_key_wavelength = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3585.0,
                4371.0,
                5478.0,
                6504.0,
                8020.0,
                12350.0,
                16460.0,
                21600.0,
                10310.0,
                12500.0,
                16360.0,
                22060.0,
                33682.0,
                46179.0,
                120717.0,
                221944.0,
                35378.0,
                44780.0,
                56962.0,
                77978.0,
                3557.0,
                4702.0,
                6175.0,
                7491.0,
                8946.0,
                4849.0,
                6201.0,
                7535.0,
                8674.0,
                9628.0,
                6229.0,
                5037.0,
                7752.0,
                6218.0,
                5110.0,
                7769.0,
                1535.0,
                2301.0,
                0.0,
            ]
        )

        self.column_names = {}
        self.column_units = {}
        self.column_wavelengths = {}
        for i, j, k, _l in zip(
            self.column_key,
            self.column_key_formatted,
            self.column_key_unit,
            self.column_key_wavelength,
        ):
            self.column_names[i] = j
            self.column_units[i] = k
            self.column_wavelengths[i] = _l

        self.column_type = np.array(([np.float64] * len(self.column_key)))
        self.dtype = [
            (i, j) for i, j in zip(self.column_key, self.column_type)
        ]

        # Load the synthetic photometry file in a recarray
        self.model_da = np.loadtxt(filepath_da, skiprows=2, dtype=self.dtype)
        self.model_db = np.loadtxt(filepath_db, skiprows=2, dtype=self.dtype)

        self.model_da["age"] = self.model_da["age"]
        self.model_db["age"] = self.model_db["age"]

    def list_atmosphere_parameters(self):
        """
        Print the formatted list of parameters available from the atmophere
        models.

        """

        for i, j in zip(self.column_names.items(), self.column_units.items()):
            print(f"Parameter: {i[1]}, Column Name: {i[0]}, Unit: {j[1]}")

    def interp_am(
        self,
        dependent="G3",
        atmosphere="H",
        independent=["logg", "Mbol"],
        logg=8.0,
        interpolator="CT",
        kwargs_for_RBF={},
        kwargs_for_CT={},
    ):
        """
        This function interpolates the grid of synthetic photometry and a few
        other physical properties as a function of 2 independent variables,
        the Default choices are 'logg' and 'Mbol'.

        Parameters
        ----------
        dependent: str (Default: 'G3')
            The value to be interpolated over. Choose from:
            'Teff', 'logg', 'mass', 'Mbol', 'BC', 'U', 'B', 'V', 'R', 'I', 'J',
            'H', 'Ks', 'Y_mko', 'J_mko', 'H_mko', 'K_mko', 'W1',
            'W2', 'W3', 'W4', 'S36', 'S45', 'S58', 'S80', 'u_sdss', 'g_sdss',
            'r_sdss', 'i_sdss', 'z_sdss', 'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1',
            'y_ps1', 'G2', 'G2_BP', 'G2_RP', 'G3', 'G3_BP', 'G3_RP', 'FUV',
            'NUV', 'age'.
        atmosphere: str (Default: 'H')
            The atmosphere type, 'H' or 'He'.
        independent: list (Default: ['logg', 'Mbol'])
            The parameters to be interpolated over for dependent.
        logg: float (Default: 8.0)
            Only used if independent is of length 1.
        interpolator: str (Default: 'RBF')
            Choose between 'RBF' and 'CT'.
        kwargs_for_RBF: dict (Default: {"neighbors": None,
            "smoothing": 0.0, "kernel": "thin_plate_spline",
            "epsilon": None, "degree": None,})
            Keyword argument for the interpolator. See
            `scipy.interpolate.RBFInterpolator`.
        kwargs_for_CT: dict (Default: {'fill_value': -np.inf,
            'tol': 1e-10, 'maxiter': 100000})
            Keyword argument for the interpolator. See
            `scipy.interpolate.CloughTocher2DInterpolator`.

        Returns
        -------
            A callable function of CloughTocher2DInterpolator.

        """
        _kwargs_for_RBF = {
            "neighbors": None,
            "smoothing": 0.0,
            "kernel": "thin_plate_spline",
            "epsilon": None,
            "degree": None,
        }
        _kwargs_for_RBF.update(**kwargs_for_RBF)

        _kwargs_for_CT = {
            "fill_value": -1e10,
            "tol": 1e-10,
            "maxiter": 100000,
            "rescale": True,
        }
        _kwargs_for_CT.update(**kwargs_for_CT)

        # DA atmosphere
        if atmosphere.lower() in ["h", "hydrogen", "da"]:
            model = self.model_da

        # DB atmosphere
        elif atmosphere.lower() in ["he", "helium", "db"]:
            model = self.model_db

        else:
            raise ValueError(
                'Please choose from "h", "hydrogen", "da", "he", "helium" or '
                '"db" as the atmophere type, you have provided '
                "{}.format(atmosphere.lower())"
            )

        independent = np.asarray(independent).reshape(-1)

        independent_list = ["Teff", "mass", "Mbol", "age", "logg"]
        independent_list_lower_cases = np.char.lower(independent_list)

        # If only performing a 1D interpolation, the logg has to be assumed.
        if len(independent) == 1:
            if independent[0].lower() in independent_list_lower_cases:
                independent = np.array(("logg", independent[0]))

            else:
                raise ValueError(
                    "When ony interpolating in 1-dimension, the independent "
                    "variable has to be one of: Teff, mass, Mbol, or age."
                )

            _independent_arg_0 = np.where(
                independent[0].lower() == independent_list_lower_cases
            )[0][0]
            _independent_arg_1 = np.where(
                independent[1].lower() == independent_list_lower_cases
            )[0][0]

            independent = np.array(
                [
                    independent_list[_independent_arg_0],
                    independent_list[_independent_arg_1],
                ]
            )

            arg_0 = model[independent[0]]
            arg_1 = model[independent[1]]

            arg_1_min = np.nanmin(arg_1)
            arg_1_max = np.nanmax(arg_1)

            if independent[1] in ["Teff", "age"]:
                arg_1 = np.log10(arg_1)

            if interpolator.lower() == "ct":
                # Interpolate with the scipy CloughTocher2DInterpolator
                _atmosphere_interpolator = CloughTocher2DInterpolator(
                    (arg_0, arg_1),
                    model[dependent],
                    **_kwargs_for_CT,
                )

                def atmosphere_interpolator(_x):
                    if independent[1] in ["Teff", "age"]:
                        _x = np.log10(_x)

                    return _atmosphere_interpolator(logg, _x)

            elif interpolator.lower() == "rbf":
                # Interpolate with the scipy RBFInterpolator
                _atmosphere_interpolator = RBFInterpolator(
                    np.stack((arg_0, arg_1), -1),
                    model[dependent],
                    **_kwargs_for_RBF,
                )

                def atmosphere_interpolator(_x):
                    if isinstance(_x, (float, int)):
                        length = 1
                        _logg = logg

                    else:
                        length = len(_x)
                        _logg = [logg] * length

                    _logg = np.asarray(_logg)
                    _x = np.asarray(_x)

                    _x[_x < arg_1_min] = arg_1_min
                    _x[_x > arg_1_max] = arg_1_max

                    if independent[1] in ["Teff", "age"]:
                        _x = np.log10(_x)

                    return _atmosphere_interpolator(
                        np.array([_logg, _x], dtype="object").T.reshape(
                            length, 2
                        )
                    )

            else:
                raise ValueError(
                    "Interpolator should be CT or RBF,"
                    f"{interpolator} is given."
                )

        # If a 2D grid is to be interpolated, normally is the logg and another
        # parameter
        elif len(independent) == 2:
            _independent_arg_0 = np.where(
                independent[0].lower() == independent_list_lower_cases
            )[0][0]
            _independent_arg_1 = np.where(
                independent[1].lower() == independent_list_lower_cases
            )[0][0]

            independent = np.array(
                [
                    independent_list[_independent_arg_0],
                    independent_list[_independent_arg_1],
                ]
            )

            arg_0 = model[independent[0]]
            arg_1 = model[independent[1]]

            arg_0_min = np.nanmin(arg_0)
            arg_0_max = np.nanmax(arg_0)
            arg_1_min = np.nanmin(arg_1)
            arg_1_max = np.nanmax(arg_1)

            if independent[0] in ["Teff", "age"]:
                arg_0 = np.log10(arg_0)

            if independent[1] in ["Teff", "age"]:
                arg_1 = np.log10(arg_1)

            if interpolator.lower() == "ct":
                # Interpolate with the scipy CloughTocher2DInterpolator
                _atmosphere_interpolator = CloughTocher2DInterpolator(
                    (arg_0, arg_1),
                    model[dependent],
                    **_kwargs_for_CT,
                )

                def atmosphere_interpolator(*x):
                    x_0, x_1 = np.asarray(x, dtype="object").reshape(-1)

                    if independent[0] in ["Teff", "age"]:
                        x_0 = np.log10(x_0)

                    if independent[1] in ["Teff", "age"]:
                        x_1 = np.log10(x_1)

                    return _atmosphere_interpolator(x_0, x_1)

            elif interpolator.lower() == "rbf":
                # Interpolate with the scipy RBFInterpolator
                _atmosphere_interpolator = RBFInterpolator(
                    np.stack((arg_0, arg_1), -1),
                    model[dependent],
                    **_kwargs_for_RBF,
                )

                def atmosphere_interpolator(*x):
                    x_0, x_1 = np.asarray(x, dtype="object").reshape(-1)

                    if isinstance(x_0, (float, int, np.int32)):
                        length0 = 1
                    else:
                        length0 = len(x_0)

                    if isinstance(x_1, (float, int, np.int32)):
                        length1 = 1
                    else:
                        length1 = len(x_1)

                    if length0 == length1:
                        pass

                    elif (length0 == 1) & (length1 > 1):
                        x_0 = [x_0] * length1
                        length0 = length1

                    elif (length0 > 1) & (length1 == 1):
                        x_1 = [x_1] * length0
                        length1 = length0

                    else:
                        raise ValueError(
                            "Either one variable is a float, int or of size "
                            "1, or two variables should have the same size."
                        )

                    _x_0 = np.asarray(x_0)
                    _x_1 = np.asarray(x_1)

                    _x_0[_x_0 < arg_0_min] = arg_0_min
                    _x_0[_x_0 > arg_0_max] = arg_0_max
                    _x_1[_x_1 < arg_1_min] = arg_1_min
                    _x_1[_x_1 > arg_1_max] = arg_1_max

                    if independent[0] in ["Teff", "age"]:
                        _x_0 = np.log10(_x_0)

                    if independent[1] in ["Teff", "age"]:
                        _x_1 = np.log10(_x_1)

                    return _atmosphere_interpolator(
                        np.array([_x_0, _x_1], dtype="object").T.reshape(
                            length0, 2
                        )
                    )

            else:
                raise ValueError("This should never happen.")

        else:
            raise TypeError(
                "Please provide ONE varaible name as a string or "
                "list, or TWO varaible names in a list."
            )

        return atmosphere_interpolator
