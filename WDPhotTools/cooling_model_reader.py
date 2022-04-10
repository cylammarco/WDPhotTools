import io
import glob
import os

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RBFInterpolator


class CoolingModelReader(object):
    def __init__(self):

        super(CoolingModelReader, self).__init__()

        self.THIS_FILE = os.path.dirname(os.path.abspath(__file__))

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
            "basti_co_da_10": "Salaris et al. 2010 CO DA",
            "basti_co_db_10": "Salaris et al. 2010 CO DB",
            "basti_co_da_10_nps": "Salaris et al. 2010 CO DA, no phase separation",
            "basti_co_db_10_nps": "Salaris et al. 2010 CO DB, no phase separation",
            "lpcode_one_da_07": "Althaus et al. 2007 ONe DA",
            "lpcode_one_da_19": "Camisassa et al. 2019 ONe DA",
            "lpcode_one_db_19": "Camisassa et al. 2019 ONe DB",
            "mesa_one_da_18": "Lauffer et al. 2018 ONe DA",
            "mesa_one_db_18": "Lauffer et al. 2018 ONe DB",
        }

        self.low_mass_cooling_model_list = [
            "montreal_co_da_20",
            "montreal_co_db_20",
            "lpcode_he_da_07",
            "lpcode_co_da_07",
            "lpcode_he_da_09",
            None,
        ]

        self.intermediate_mass_cooling_model_list = [
            "montreal_co_da_20",
            "montreal_co_db_20",
            "lpcode_co_da_10_z001",
            "lpcode_co_da_10_z0001",
            "lpcode_co_da_15_z00003",
            "lpcode_co_da_15_z0001",
            "lpcode_co_da_15_z0005",
            "lpcode_co_db_17_z0001",
            "lpcode_co_db_17_z00005",
            "lpcode_co_da_17_y04",
            "lpcode_co_db_17",
            "basti_co_da_10",
            "basti_co_db_10",
            "basti_co_da_10_nps",
            "basti_co_db_10_nps",
            None,
        ]

        self.high_mass_cooling_model_list = [
            "montreal_co_da_20",
            "montreal_co_db_20",
            "lpcode_one_da_07",
            "lpcode_one_da_19",
            "lpcode_one_db_19",
            "basti_co_da_10",
            "basti_co_db_10",
            "basti_co_da_10_nps",
            "basti_co_db_10_nps",
            "mesa_one_da_18",
            "mesa_one_db_18",
            None,
        ]

        # Default to montreal_co_da_20
        self.cooling_models = {
            "low_mass_cooling_model": "montreal_co_da_20",
            "intermediate_mass_cooling_model": "montreal_co_da_20",
            "high_mass_cooling_model": "montreal_co_da_20",
        }

    def list_cooling_model(self, print_to_screen=True):
        """
        Print the formatted list of available cooling models.

        Parameters
        ----------
        print_to_screen: bool (Default: True)
            Set to True to print the list of cooling models to screen.

        Returns
        -------
        model_list:
            The names and references of the cooling models.

        """

        if print_to_screen:

            for i in self.model_list.items():

                print("Model: {}, Reference: {}".format(i[0], i[1]))

        return self.model_list.items()

    def list_cooling_parameters(self, model, print_to_screen=True):
        """
        Print the formatted list of parameters available for the specified
        cooling models.

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.
        print_to_screen: bool (Default: True)
            Set to True to print the cooling model parameters to screen.

        Returns
        -------
        mass:
            WD mass available in the specified model.
        column_names:
            Available parameters in the specified model.
        column_units:
            Unites of the parameters in the specified model.

        """

        mass, _, column_names, column_units = self.get_cooling_model(model)

        if print_to_screen:

            print("Available WD mass: {}".format(mass))

            for i, j in zip(column_names.items(), column_units.items()):

                print(
                    "Parameter: {}, Column Name: {}, Unit: {}".format(
                        i[1], i[0], j[1]
                    )
                )

        return mass, column_names.items(), column_units.items()

    def get_cooling_model(self, model, mass_range="all"):
        """
        Choose the specified cooling model for the chosen mass range.

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.
        mass_range: str (Default: 'all')
            The mass range in which the cooling model should return.
            The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.

        """

        if model in ["montreal_co_da_20", "montreal_co_db_20"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._bedard20_formatter(model, mass_range)

        elif model in ["lpcode_he_da_07", "lpcode_co_da_07"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._panei07_formatter(model)

        elif model == "lpcode_he_da_09":

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._althaus09_formatter(mass_range)

        elif model in ["lpcode_co_db_17_z00005", "lpcode_co_db_17_z0001"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._althaus17_formatter(model, mass_range)

        elif model in ["lpcode_co_da_10_z001", "lpcode_co_da_10_z0001"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._renedo10_formatter(model)

        elif model in [
            "lpcode_co_da_15_z00003",
            "lpcode_co_da_15_z0001",
            "lpcode_co_da_15_z0005",
        ]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._althaus15_formatter(model)

        elif model == "lpcode_co_db_17":

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._camisassa17_formatter()

        elif model in [
            "basti_co_da_10",
            "basti_co_db_10",
            "basti_co_da_10_nps",
            "basti_co_db_10_nps",
        ]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._salaris10_formatter(model, mass_range)

        elif model == "lpcode_one_da_07":

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._althaus07_formatter()

        elif model in ["lpcode_one_da_19", "lpcode_one_db_19"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._camisassa19_formatter(model)

        elif model in ["mesa_one_da_18", "mesa_one_db_18"]:

            (
                mass,
                cooling_model,
                column_names,
                column_units,
            ) = self._lauffer18_formatter(model)

        else:

            raise ValueError("Invalid model name.")

        return mass, cooling_model, column_names, column_units

    def _althaus07_formatter(self):
        """
        A formatter to load the Althaus et al. 2007 WD cooling model

        """

        filelist = glob.glob(
            os.path.join(self.THIS_FILE, "wd_cooling/althaus07/*.dat")
        )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "logg",
                "B-V",
                "V-R",
                "V-K",
                "R-I",
                "J-H",
                "H-K",
                "V-I",
                "U-V",
                "BC",
                "dmag_v",
                "age",
            )
        )
        column_key_formatted = np.array(
            (
                "Luminosity",
                "log(g)",
                r"$B-V$",
                r"$V-R$",
                r"$V-K$",
                r"$R-I$",
                r"$J-H$",
                r"$H-K$",
                r"$V-I$",
                r"$U-V$",
                "$Bolometric Correction$",
                r"$V$",
                "$log(Age)$",
            )
        )
        column_key_unit = np.array(
            (
                r"L$_{\odot}$",
                "(cgs)",
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
                "(yr)",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split("_")[-1][:3] for i in filelist]).astype(
                np.float64
            )
            / 100000.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"]

        return mass, cooling_model, column_names, column_units

    def _althaus09_formatter(self, mass_range="all"):
        """
        A formatter to load the Althaus et al. 2009 WD cooling model

        Parameters
        ----------
        mass_range: str (Default: 'all')
            The mass range in which the cooling model should return.
            The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.

        """

        filelist = glob.glob(
            os.path.join(self.THIS_FILE, "wd_cooling/althaus09/z.*")
        )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "Teff",
                "logg",
                "lum",
                "age",
                "BC",
                "M_V",
                "U",
                "B",
                "V",
                "R",
                "I",
                "J",
                "H",
                "K",
                "L",
                "U-B",
                "B-V",
                "V-K",
                "V-I",
                "R-I",
                "J-H",
                "H-K",
                "K-L",
            )
        )
        column_key_formatted = np.array(
            (
                r"T$_{\mathrm{eff}}$",
                "log(g)",
                "Luminosity",
                "$log(Age)$",
                "$Bolometric Correction$",
                r"$V$",
                r"$U$",
                r"$B$",
                r"$V$",
                r"$R$",
                r"$I$",
                r"$J$",
                r"$H$",
                r"$K$",
                r"$L$",
                r"$U-B$",
                r"$B-V$",
                r"$V-K$",
                r"$V-I$",
                r"$R-I$",
                r"$J-H$",
                r"$H-K$",
                r"$K-L$",
            )
        )
        column_key_unit = np.array(
            (
                "K",
                r"(cm/s$^2$)",
                r"L$_{\odot}$",
                "(yr)",
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
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split(".")[-2] for i in filelist]).astype(np.float64)
            / 100000.0
        )

        if mass_range == "all":
            pass
        elif mass_range == "low":
            mask_low = mass < 0.5
            mass = mass[mask_low]
            filelist = np.array(filelist)[mask_low]
        else:
            raise ValueError(
                "Unknown mass range requested. Please choose "
                "from 'all' or 'low' for althaus09 models."
            )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] *= 1e9

        return mass, cooling_model, column_names, column_units

    def _althaus15_formatter(self, model):
        """
        A formatter to load the Althaus et al. 2015 WD cooling model

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.

        """

        # Z=0.00003 models
        if model == "lpcode_co_da_15_z00003":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/althaus15/Z=3d-5/*.trk"
                )
            )

        # Z=0.0001 models
        if model == "lpcode_co_da_15_z0001":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/althaus15/Z=1d-4/*.trk"
                )
            )

        # Z=0.0005 models
        if model == "lpcode_co_da_15_z0005":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/althaus15/Z=5d-4/*.trk"
                )
            )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "Teff",
                "Tc",
                "Roc",
                "Hc",
                "Hec",
                "Con_s",
                "Con_c",
                "age",
                "mass",
                "mdot",
                "model_no",
                "Lpp",
                "Lcno",
                "LHe",
                "LCC",
                "dSdt",
                "Lnu",
                "MHtot",
                "HeBuf",
                "mass_Hfc",
                "mass_Hefc",
                "logg",
                "Rsun",
                "LH",
                "ps",
            )
        )
        column_key_formatted = np.array(
            (
                "Luminosity",
                r"log(T$_{\mathrm{eff}})$",
                r"T$_{\mathrm{c}}$",
                r"$\rho_c$",
                r"X$_c$",
                r"Y$_c$",
                "Outer Convective Zone",
                "Inner Convective Zone",
                "log(Age)",
                "Mass",
                "log(Rate of Change of Mass)",
                "Model Number",
                r"log($L_{PP}$)",
                r"log($L_{CNO}$)",
                r"log($L_{He}$)",
                r"log($L_{CC}$)",
                r"$\int\frac{\D{S}}{\D{t}}$",
                r"log($L_{\nu}$)",
                r"log(M$_{H, tot}$)",
                r"log(Mass$_{\mathrm{He Buffer}}$)",
                r"log(Mass$_{\mathrm{H-free Core}}$)",
                r"log(Mass$_{\mathrm{He-free Core}}$)",
                "log(g)",
                r"Radius",
                "Latent Heat",
                "Phase Separation",
            )
        )
        column_key_unit = np.array(
            (
                r"L$_{\odot}$",
                "(K)",
                r"($10^6$ K)",
                r"(g/cm$^3$)",
                "",
                "",
                "%",
                "%",
                "($10^6$ K)",
                r"M$_\odot$",
                r"(M$_\odot$ / yr)",
                "",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                "",
                r"L$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"(cm/s$2^$)",
                r"R$_{\odot}$",
                "erg/s",
                "erg/s",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split(".")[-2][-5:] for i in filelist]).astype(
                np.float64
            )
            / 100000.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"] * 1e6
            cooling_model[i]["age"] -= min(cooling_model[i]["age"])

        return mass, cooling_model, column_names, column_units

    def _althaus17_formatter(self, model, mass_range="all"):
        """
        A formatter to load the Althaus et al. 2017 WD cooling model

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.
        mass_range: str (Default: 'all')
            The mass range in which the cooling model should return.
            The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.

        """

        # Y=0.4, Z=0.001 models
        if model == "lpcode_co_db_17_z00005":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/althaus17/*d4.trk")
            )

        # Y=0.4, Z=0.0005 models
        if model == "lpcode_co_db_17_z0001":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/althaus17/*d3.trk")
            )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "Teff",
                "Tc",
                "Roc",
                "Hc",
                "Hec",
                "Con_s",
                "Con_c",
                "age",
                "mass",
                "mdot",
                "model_no",
                "Lpp",
                "Lcno",
                "LHe",
                "LCC",
                "dSdt",
                "Lnu",
                "MHtot",
                "HeBuf",
                "mass_Hfc",
                "mass_Hefc",
                "logg",
                "Rsun",
                "LH",
                "ps",
            )
        )
        column_key_formatted = np.array(
            (
                "Luminosity",
                r"log(T$_{\mathrm{eff}})$",
                r"T$_{\mathrm{c}}$",
                r"$\rho_c$",
                r"X$_c$",
                r"Y$_c$",
                "Outer Convective Zone",
                "Inner Convective Zone",
                "log(Age)",
                "Mass",
                "log(Rate of Change of Mass)",
                "Model Number",
                r"log($L_{PP}$)",
                r"log($L_{CNO}$)",
                r"log($L_{He}$)",
                r"log($L_{CC}$)",
                r"$\int\frac{\D{S}}{\D{t}}$",
                r"log($L_{\nu}$)",
                r"log(M$_{H, tot}$)",
                r"Mass$_{\mathrm{He Buffer}}$",
                r"Mass$_{\mathrm{H-free Core}}$",
                r"Mass$_{\mathrm{He-free Core}}$",
                "log(g)",
                "Radius",
                "Latent Heat",
                "Phase Separation",
            )
        )
        column_key_unit = np.array(
            (
                r"L$_{\odot}$",
                "(K)",
                r"($10^6$ K)",
                r"(g/cm$^3$)",
                "",
                "",
                "%",
                "%",
                "($10^6$ K)",
                r"M$_\odot$",
                r"(M$_\odot$ / yr)",
                "",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                "",
                r"L$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"(cm/s$^2$)",
                r"R$_{\odot}$",
                "erg/s",
                "erg/s",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = np.array(
            [i.split(os.sep)[-1].split("_")[0] for i in filelist]
        ).astype(np.float64)
        wd_mass = np.zeros_like(mass)

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            wd_mass[i] = cooling_model[i]["mass"][0]
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"] * 1e6
            cooling_model[i]["age"] -= min(cooling_model[i]["age"])

        if mass_range == "all":
            pass
        elif mass_range == "low":
            mask_low = mass < 0.5
            wd_mass = wd_mass[mask_low]
            cooling_model = cooling_model[mask_low]
        elif mass_range == "intermediate":
            mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
            wd_mass = wd_mass[mask_intermediate]
            cooling_model = cooling_model[mask_intermediate]
        else:
            raise ValueError(
                "Unknown mass range requested. Please choose from"
                "'all', 'low' or 'intermediate' for althaus17 models."
            )

        return wd_mass, cooling_model, column_names, column_units

    def _bedard20_formatter(self, model, mass_range="all"):
        """
        A formatter to load the Bedard et al. 2020 WD cooling model from
        http://www.astro.umontreal.ca/~bergeron/CoolingModels/

        The thick and thin models are for DA and DB WD, respectively.

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.
        mass_range: str (Default: 'all')
            The mass range in which the cooling model should return.
            The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.

        """

        # DA models
        if model == "montreal_co_da_20":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/bedard20/*thick*")
            )

        # DB models
        if model == "montreal_co_db_20":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/bedard20/*thin*")
            )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "step",
                "Teff",
                "logg",
                "r",
                "age",
                "lum",
                "logTc",
                "logPc",
                "logrhoc",
                "MxM",
                "logqx",
                "lumnu",
                "logH",
                "logHe",
                "logC",
                "logO",
            )
        )
        column_key_formatted = np.array(
            (
                "Step",
                r"T$_{\mathrm{eff}}$",
                "log(g)",
                "Radius",
                "Age",
                "Luminosity",
                r"log(T$_{\mathrm{c}}$)",
                r"log(P$_{\mathrm{c}}$)",
                r"log($\rho_c$)",
                "Mass Fraction of Crystallisation",
                "Location of The Crystallization Front",
                r"$L_{\nu}$",
                r"log(Mass Fraction$_{H}$",
                r"log(Mass Fraction$_{He}$",
                r"log(Mass Fraction$_{C}$",
                r"log(Mass Fraction$_{O}$",
            )
        )
        column_key_unit = np.array(
            (
                r"",
                "K",
                r"(cm/s$^2$)",
                "cm",
                "yr",
                "erg/s",
                "(K)",
                "(K)",
                r"(g/cm$^3$)",
                "",
                "",
                "erg/s",
                "",
                "",
                "",
                "",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split("_")[2] for i in filelist]).astype(np.float64)
            / 100.0
        )

        if mass_range == "all":
            pass
        elif mass_range == "low":
            mask_low = mass < 0.5
            mass = mass[mask_low]
            filelist = np.array(filelist)[mask_low]
        elif mass_range == "intermediate":
            mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
            mass = mass[mask_intermediate]
            filelist = np.array(filelist)[mask_intermediate]
        elif mass_range == "high":
            mask_high = mass > 1.0
            mass = mass[mask_high]
            filelist = np.array(filelist)[mask_high]
        else:
            raise ValueError(
                "Unknown mass range requested. Please choose from"
                "'all', 'low', 'intermediate' or 'high' for bedard20 models."
            )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            with open(filepath) as infile:

                count = -5
                cooling_model_text = ""
                for line_i in infile:

                    count += 1

                    if count <= 0:
                        continue

                    if count % 3 != 0:
                        cooling_model_text += line_i.rstrip("\n")
                    else:
                        cooling_model_text += line_i

            cooling_model[i] = np.loadtxt(
                io.StringIO(cooling_model_text), dtype=dtype
            )

        return mass, cooling_model, column_names, column_units

    def _camisassa17_formatter(self):
        """
        A formatter to load the Camisassa et al. 2017 WD cooling model

        The progenitor lifetime is taken off based on the extrapolation from
        Table 1
        https://iopscience.iop.org/article/10.3847/0004-637X/823/2/158

        """

        # Y=0.4, Z=0.0005 models
        filelist = glob.glob(
            os.path.join(self.THIS_FILE, "wd_cooling/camisassa17/*.trk")
        )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "Teff",
                "Tc",
                "Roc",
                "Hc",
                "Hec",
                "Con_s",
                "Con_c",
                "age",
                "mass",
                "mdot",
                "model_no",
                "Lpp",
                "Lcno",
                "LHe",
                "LCC",
                "logG",
                "Lnu",
                "MHtot",
                "HeBuf",
                "mass_Hfc",
                "mass_Hefc",
                "logg",
                "Rsun",
            )
        )
        column_key_formatted = np.array(
            (
                "Luminosity",
                r"log(T$_{\mathrm{eff}})$",
                r"T$_{\mathrm{c}}$",
                r"$\rho_c$",
                r"X$_c$",
                r"Y$_c$",
                "Outer Convective Zone",
                "Inner Convective Zone",
                "log(Age)",
                "Mass",
                "log(Rate of Change of Mass)",
                "Model Number",
                r"log($L_{PP}$)",
                r"log($L_{CNO}$)",
                r"log($L_{He}$)",
                r"log($L_{CC}$)",
                r"log($L_{G}$)",
                r"log($L_{\nu}$)",
                r"log(M$_{H, tot}$)",
                r"log(HeBuf)",
                r"Mass$_{H-free Core}$",
                r"Mass$_{He-free Core}$",
                "log(g)",
                r"Radius",
                "Latent Heat",
                "Phase Separation",
            )
        )
        column_key_unit = np.array(
            (
                r"L$_{\odot}$",
                "(K)",
                r"($10^6$ K)",
                r"(g/cm$^3$)",
                "",
                "",
                "%",
                "%",
                r"($10^6$ K)",
                r"M$_\odot$",
                r"(M$_\odot$ / yr)",
                "",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"M$_{\odot}$",
                r"(cm/s$^2$)",
                r"R$_{\odot}$",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split(os.sep)[-1][:3] for i in filelist]).astype(
                np.float64
            )
            / 100.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"] * 1e6
            cooling_model[i]["age"] -= min(cooling_model[i]["age"])

        return mass, cooling_model, column_names, column_units

    def _camisassa19_formatter(self, model):
        """
        A formatter to load the Camisassa et al. 2019 ultramassive WD cooling
        model.

        Some columns populated with 'I' are replaced with the nearest values.

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.

        """

        # DA model
        if model == "lpcode_one_da_19":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/camisassa19/*hrich.dat"
                )
            )

        # DB model
        if model == "lpcode_one_db_19":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/camisassa19/*hdef.dat"
                )
            )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "Teff",
                "Tc",
                "Roc",
                "Hc",
                "Hec",
                "Con_s",
                "Con_c",
                "age",
                "mass",
                "mdot",
                "Lnu",
                "MHtot",
                "logg",
                "Rsun",
                "LH",
                "sf",
            )
        )
        column_key_formatted = np.array(
            (
                "Luminosity",
                r"log(T$_{\mathrm{eff}})$",
                r"T$_{\mathrm{c}}$",
                r"$\rho_c$",
                r"X$_c$",
                r"Y$_c$",
                "Outer Convective Zone",
                "Inner Convective Zone",
                "log(Age)",
                "Mass",
                "log(Rate of Change of Mass)",
                r"log($L_{\nu}$)",
                r"log(M$_{H, tot}$)",
                "log(g)",
                r"Radius",
                "Latent Heat",
                "Phase Separation",
            )
        )
        column_key_unit = np.array(
            (
                r"L$_{\odot}$",
                "(K)",
                r"($10^6$ K)",
                r"(g/cm$^3$)",
                "",
                "",
                "%",
                "%",
                r"M$_\odot$",
                r"(M$_\odot$ / yr)",
                r"L$_{\odot}$",
                r"M$_{\odot}$",
                r"(cm/s$^2$)",
                r"R$_{\odot}$",
                "erg/s",
                "erg/s",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split(os.sep)[-1][:3] for i in filelist]).astype(
                np.float64
            )
            / 100.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=2, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"] * 1e6
            cooling_model[i]["age"] -= min(cooling_model[i]["age"])

        return mass, cooling_model, column_names, column_units

    def _lauffer18_formatter(self, model):
        """
        A formatter to load the Lauffer et al. 2018 WD cooling model

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.

        """

        # H models
        if model == "mesa_one_da_18":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/lauffer18/H_*.dat")
            )

        # He models
        if model == "mesa_one_db_18":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/lauffer18/He_*.dat")
            )

        # Prepare the array column dtype
        column_key = np.array(
            ("Teff", "lum", "logg", "Rsun", "mass", "age", "total_age")
        )
        column_key_formatted = np.array(
            (
                r"log(T$_{\mathrm{eff}})$",
                "Luminosity",
                "log(g)",
                r"Radius",
                "Mass",
                "log(Cooling Age)",
                "log(Total Age)",
            )
        )
        column_key_unit = np.array(
            (
                "(K)",
                r"L$_{\odot}$",
                r"(cm/s$^2$)",
                r"R$_{\odot}$",
                r"M$_\odot$",
                r"(Gyr)",
                r"(Gyr)",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = np.array([i.split("-M")[-1][:-4] for i in filelist]).astype(
            np.float64
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] *= 1e9

        return mass, cooling_model, column_names, column_units

    def _panei07_formatter(self, model):
        """
        A formatter to load the Panei et al. 2007 WD cooling model

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.

        """

        # He core models
        if model == "lpcode_he_da_07":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/panei07/*He.SDSS")
            )

        # CO core models
        if model == "lpcode_co_da_07":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/panei07/*CO.SDSS")
            )

        # Prepare the array column dtype
        column_key = np.array(
            ("Teff", "logg", "lum", "age", "u", "g", "r", "i", "z")
        )
        column_key_formatted = np.array(
            (
                r"log(T$_{\mathrm{eff}})$",
                "log(g)",
                "Luminosity",
                "log(Age)",
                "u",
                "g",
                "r",
                "i",
                "z",
            )
        )
        column_key_unit = np.array(
            (
                "(K)",
                r"(cm/s$^2$)",
                r"L$_{\odot}$",
                r"(Gyr)",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split(".")[-2][:5] for i in filelist]).astype(
                np.float64
            )
            / 100000.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] *= 1e9

        return mass, cooling_model, column_names, column_units

    def _renedo10_formatter(self, model):
        """
        A formatter to load the Renedo et al. 2010 WD cooling model from
        http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html

        Two metallicity for DA are available: Z=0.01 and Z=0.001

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.

        """

        # Solar metallicity model
        if model == "lpcode_co_da_10_z001":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/renedo10/*z001.trk")
            )

        # Low metallicity model
        if model == "lpcode_co_da_10_z0001":
            filelist = glob.glob(
                os.path.join(self.THIS_FILE, "wd_cooling/renedo10/*z0001.trk")
            )

        # Prepare the array column dtype
        column_key = np.array(
            (
                "lum",
                "Teff",
                "logTc",
                "logrhoc",
                "age",
                "mass",
                "lumpp",
                "lumcno",
                "lumhe",
                "lumnu",
                "logH",
                "logg",
                "rsun",
            )
        )
        column_key_formatted = np.array(
            (
                "log(Luminosity)",
                r"log(T$_{\mathrm{eff}})$",
                r"log(T$_{\mathrm{c}})$",
                r"log($\rho_{\mathrm{c}})$",
                "log(Age)",
                "Mass",
                r"log($L_{PP}$)",
                r"log($L_{CNO}$)",
                r"log($L_{He}$)",
                r"log($L_{\nu}$)",
                r"log(M$_{H, tot}$)",
                "log(g)",
                "Radius",
            )
        )
        column_key_unit = np.array(
            (
                "erg/s",
                "(K)",
                "(K)",
                r"(g/cm$^3$)",
                r"(Gyr)",
                r"M$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"L$_{\odot}$",
                r"M$_{\odot}$",
                r"(cm/s$^2$)",
                r"E$_{\odot}$",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split("_")[1][-4:] for i in filelist]).astype(
                np.float64
            )
            / 1000.0
        )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] *= 1e6

        return mass, cooling_model, column_names, column_units

    def _salaris10_formatter(self, model, mass_range="all"):
        """
        A formatter to load the Salaris et al. 2010 WD cooling model from

        Parameters
        ----------
        model: str
            Name of the cooling model as in the `model_list`.
        mass_range: str (Default: 'all')
            The mass range in which the cooling model should return.
            The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.

        """

        # DA model with phase separation
        if model == "basti_co_da_10":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/salaris10/*DAsep.sdss"
                )
            )

        # DB model with phase separation
        if model == "basti_co_db_10":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/salaris10/*DBsep.sdss"
                )
            )

        # DA model without phase separation
        if model == "basti_co_da_10_nps":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/salaris10/*DAnosep.sdss"
                )
            )

        # DB model without phase separation
        if model == "basti_co_db_10_nps":
            filelist = glob.glob(
                os.path.join(
                    self.THIS_FILE, "wd_cooling/salaris10/*DBnosep.sdss"
                )
            )

        # Prepare the array column dtype
        column_key = np.array(
            ("age", "mass", "Teff", "lum", "u", "g", "r", "i", "z")
        )
        column_key_formatted = np.array(
            (
                "log(Age)",
                "Mass",
                r"log(T$_{\mathrm{eff}})$",
                "Luminosity",
                "u",
                "g",
                "r",
                "i",
                "z",
            )
        )
        column_key_unit = np.array(
            (
                "(Gyr)",
                r"M$_{\odot}$",
                "(K)",
                r"L$_{\odot}$",
                "mag",
                "mag",
                "mag",
                "mag",
                "mag",
            )
        )
        column_type = np.array(([np.float64] * len(column_key)))
        dtype = [(i, j) for i, j in zip(column_key, column_type)]

        column_names = {}
        column_units = {}
        for i, j, k in zip(column_key, column_key_formatted, column_key_unit):
            column_names[i] = j
            column_units[i] = k

        # Get the mass from the file name
        mass = (
            np.array([i.split("COOL")[-1][:3] for i in filelist]).astype(
                np.float64
            )
            / 100.0
        )

        if mass_range == "all":
            pass
        elif mass_range == "intermediate":
            mask_intermediate = (mass >= 0.5) & (mass <= 1.0)
            mass = mass[mask_intermediate]
            filelist = np.array(filelist)[mask_intermediate]
        elif mass_range == "high":
            mask_high = mass > 1.0
            mass = mass[mask_high]
            filelist = np.array(filelist)[mask_high]
        else:
            raise ValueError(
                "Unknown mass range requested. Please choose from"
                "'all', 'intermediate' or 'high' for bedard20 models."
            )

        # Create an empty array for holding the cooling models
        cooling_model = np.array(([""] * len(mass)), dtype="object")

        for i, filepath in enumerate(filelist):

            cooling_model[i] = np.loadtxt(filepath, skiprows=1, dtype=dtype)

            # Convert the luminosity into erg/s
            cooling_model[i]["lum"] = (
                10.0 ** cooling_model[i]["lum"] * 3.826e33
            )

            # Convert the age to yr
            cooling_model[i]["age"] = 10.0 ** cooling_model[i]["age"]

        return mass, cooling_model, column_names, column_units

    def set_low_mass_cooling_model(self, model):
        """
        Set the WD cooling model.

        Parameters
        ----------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:

            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_he_da_07' - Panei et al. 2007 He DA
            4. 'lpcode_co_da_07' - Panei et al. 2007 CO DA
            5. 'lpcode_he_da_09' - Althaus et al. 2009 He DA

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        """

        if model in self.low_mass_cooling_model_list:

            self.cooling_models["low_mass_cooling_model"] = model

        else:

            raise ValueError("Please provide a valid model.")

    def set_intermediate_mass_cooling_model(self, model):
        """
        Set the WD cooling model.

        Parameters
        ----------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:

            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_co_da_10_z001' - Renedo et al. 2010 CO DA Z=0.01
            4. 'lpcode_co_da_10_z0001' - Renedo et al. 2010 CO DA Z=0.001
            5. 'lpcode_co_da_15_z00003' - Althaus et al. 2015 DA Z=0.00003
            6. 'lpcode_co_da_15_z0001' - Althaus et al. 2015 DA Z=0.0001
            7. 'lpcode_co_da_15_z0005' - Althaus et al. 2015 DA Z=0.0005
            8. 'lpcode_co_da_17_y04' - Althaus et al. 2017 DB Y=0.4
            9. 'lpcode_co_db_17' - Camisassa et al. 2017 DB
            10. 'basti_co_da_10' - Salaris et al. 2010 CO DA
            11. 'basti_co_db_10' - Salaris et al. 2010 CO DB
            12. 'basti_co_da_10_nps' - Salaris et al. 2010 CO DA,
                no phase separation
            13. 'basti_co_db_10_nps' - Salaris et al. 2010 CO DB,
                no phase separation

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        """

        if model in self.intermediate_mass_cooling_model_list:

            self.cooling_models["intermediate_mass_cooling_model"] = model

        else:

            raise ValueError("Please provide a valid model.")

    def set_high_mass_cooling_model(self, model):
        """
        Set the WD cooling model.

        Parameters
        ----------
        model: str (Default: 'montreal_co_da_20')
            Choice of WD cooling model:

            1. 'montreal_co_da_20' - Bedard et al. 2020 CO DA
            2. 'montreal_co_db_20' - Bedard et al. 2020 CO DB
            3. 'lpcode_one_da_07' - Althaus et al. 2007 ONe DA
            4. 'lpcode_one_da_19' - Camisassa et al. 2019 ONe DA
            5. 'lpcode_one_db_19' - Camisassa et al. 2019 ONe DB
            6. 'basti_co_da_10' - Salaris et al. 2010 CO DA
            7. 'basti_co_db_10' - Salaris et al. 2010 CO DB
            8. 'basti_co_da_10_nps' - Salaris et al. 2010 CO DA,
                no phase separation
            9. 'basti_co_db_10_nps' - Salaris et al. 2010 CO DB,
                no phase separation
            10. 'mesa_one_da_18' - Lauffer et al. 2018 ONe DA
            11. 'mesa_one_db_18' - Lauffer et al. 2018 ONe DB

            The naming convention follows this format:
            [model]_[core composition]_[atmosphere]_[publication year]
            where a few models continue to have extra property description
            terms trailing after the year, currently they are either the
            progenitor metallicity or the (lack of) phase separation in the
            evolution model.

        """

        if model in self.high_mass_cooling_model_list:

            self.cooling_models["high_mass_cooling_model"] = model

        else:

            raise ValueError("Please provide a valid model.")

    def _itp2D_gradient(self, f, val1, val2, frac=1e-6):
        """
        A function to find the gradient in the direction in the first dimension
        of a 2D function at a given coordinate.

        Parameters
        ----------
        f: callable function
            A 2D function
        val1: float
            The first input value accepted by f. The gradient is computed in
            this direction.
        val2: float
            The first input value accepted by f.
        frac: float (Default: 1e-6)
            The small fractional increment of val1.

        Return
        ------
        Gradient in the direction of val1.

        """

        if not callable(f):
            raise TypeError("f has to be a callable function.")

        increment = val1 * frac / 2.0
        grad = np.asarray(
            (f(val1 + increment, val2) - f(val1 - increment, val2))
            / (increment * 2.0)
        ).reshape(-1)

        # cooling((L+1), m) - cooling(L, m) is always negative
        grad[grad > 0.0] = 0.0
        grad[np.isnan(grad)] = 0.0

        return grad

    def compute_cooling_age_interpolator(
        self,
        interpolator="CT",
        kwargs_for_RBF={},
        kwargs_for_CT={},
    ):
        """
        Compute the callable CloughTocher2DInterpolator of the cooling time
        of WDs. It needs to use float64 or it runs into float-point error
        at very faint lumnosity.

        Parameters
        ----------
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

        """

        # Set the low mass cooling model, i.e. M < 0.5 M_sun
        mass_low, cooling_model_low, _, _ = self.get_cooling_model(
            self.cooling_models["low_mass_cooling_model"], mass_range="low"
        )

        # Set the intermediate mass cooling model, i.e. 0.5 < M < 1.0 M_sun
        (
            mass_intermediate,
            cooling_model_intermediate,
            _,
            _,
        ) = self.get_cooling_model(
            self.cooling_models["intermediate_mass_cooling_model"],
            mass_range="intermediate",
        )

        # Set the high mass cooling model, i.e. 1.0 < M M_sun
        mass_high, cooling_model_high, _, _ = self.get_cooling_model(
            self.cooling_models["high_mass_cooling_model"], mass_range="high"
        )

        # Gather all the models in different mass ranges

        # Reshaping the WD mass array to match the shape of the other two.
        mass_low = (
            np.concatenate(
                np.array(
                    [
                        [mass_low[i]] * len(model["age"])
                        for i, model in enumerate(cooling_model_low)
                    ],
                    dtype=object,
                )
            )
            .T.ravel()
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and age
        luminosity_low = (
            np.concatenate([i["lum"] for i in cooling_model_low])
            .reshape(-1)
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and luminosity
        age_low = (
            np.concatenate([i["age"] for i in cooling_model_low])
            .reshape(-1)
            .astype(np.float64)
        )

        # Reshaping the WD mass array to match the shape of the other two.
        mass_intermediate = (
            np.concatenate(
                np.array(
                    [
                        [mass_intermediate[i]] * len(model["age"])
                        for i, model in enumerate(cooling_model_intermediate)
                    ],
                    dtype=object,
                )
            )
            .T.ravel()
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and age
        luminosity_intermediate = (
            np.concatenate([i["lum"] for i in cooling_model_intermediate])
            .reshape(-1)
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and luminosity
        age_intermediate = (
            np.concatenate([i["age"] for i in cooling_model_intermediate])
            .reshape(-1)
            .astype(np.float64)
        )

        # Reshaping the WD mass array to match the shape of the other two.
        mass_high = (
            np.concatenate(
                np.array(
                    [
                        [mass_high[i]] * len(model["age"])
                        for i, model in enumerate(cooling_model_high)
                    ],
                    dtype=object,
                )
            )
            .T.ravel()
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and age
        luminosity_high = (
            np.concatenate([i["lum"] for i in cooling_model_high])
            .reshape(-1)
            .astype(np.float64)
        )

        # The luminosity of the WD at the corresponding mass and luminosity
        age_high = (
            np.concatenate([i["age"] for i in cooling_model_high])
            .reshape(-1)
            .astype(np.float64)
        )

        self.cooling_model_grid = np.concatenate(
            (cooling_model_low, cooling_model_intermediate, cooling_model_high)
        )

        self.mass = np.concatenate((mass_low, mass_intermediate, mass_high))
        self.luminosity = np.concatenate(
            (luminosity_low, luminosity_intermediate, luminosity_high)
        )
        self.age = np.concatenate((age_low, age_intermediate, age_high))

        # Configure interpolator for the cooling models
        _kwargs_for_CT = {
            "fill_value": float("-inf"),
            "tol": 1e-10,
            "maxiter": 100000,
            "rescale": True,
        }
        _kwargs_for_CT.update(**kwargs_for_CT)

        _kwargs_for_RBF = {
            "neighbors": None,
            "smoothing": 0.0,
            "kernel": "thin_plate_spline",
            "epsilon": None,
            "degree": None,
        }
        _kwargs_for_RBF.update(**kwargs_for_RBF)

        if interpolator.upper() == "CT":

            # Interpolate with the scipy CloughTocher2DInterpolator
            self.cooling_interpolator = CloughTocher2DInterpolator(
                (np.log10(self.luminosity), self.mass),
                self.age,
                **_kwargs_for_CT,
            )

        elif interpolator.upper() == "RBF":

            # Interpolate with the scipy RBFInterpolator
            _cooling_interpolator = RBFInterpolator(
                np.stack((np.log10(self.luminosity), self.mass), -1),
                self.age,
                **_kwargs_for_RBF,
            )

            def cooling_interpolator(x0, x1):

                _x0 = np.asarray(x0)
                _x1 = np.asarray(x1)

                length0 = len(x0)

                return _cooling_interpolator(
                    np.asarray([_x0, _x1], dtype="object").reshape(length0, 2)
                )

            self.cooling_interpolator = cooling_interpolator

        else:

            raise ValueError(
                "Interpolator should be CT or RBF, {} is given.".format(
                    interpolator
                )
            )

        self.dLdt = self._itp2D_gradient(
            self.cooling_interpolator, np.log10(self.luminosity), self.mass
        )

        finite_mask = np.isfinite(self.dLdt)

        if interpolator.upper() == "CT":

            self.cooling_rate_interpolator = CloughTocher2DInterpolator(
                (
                    np.log10(self.luminosity)[finite_mask],
                    self.mass[finite_mask],
                ),
                self.dLdt[finite_mask],
                **_kwargs_for_CT,
            )

        elif interpolator.upper() == "RBF":

            # Interpolate with the scipy RBFInterpolator
            self.cooling_rate_interpolator = RBFInterpolator(
                np.stack(
                    (
                        np.log10(self.luminosity)[finite_mask],
                        self.mass[finite_mask],
                    ),
                    -1,
                ),
                self.dLdt[finite_mask],
                **_kwargs_for_RBF,
            )

        else:

            raise ValueError(
                "Interpolator should be CT or RBF, {} is given.".format(
                    interpolator
                )
            )
