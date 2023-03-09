#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Some plotting functions"""

import os

import numpy as np
from matplotlib import pyplot as plt

from .atmosphere_model_reader import AtmosphereModelReader
from .cooling_model_reader import CoolingModelReader


class DummyAtm:
    """dummy class to load atmosphere reader if needed"""

    def __init__(self):
        self.reader = None


class DummyCm:
    """dummy class to load cooling model reader if needed"""

    def __init__(self):
        self.reader = None


# Create dummy object, only load the respective readers when needed.
__dummy_atm = DummyAtm()
__dummy_cm = DummyCm()

plt.rc("font", size=18)
plt.rc("legend", fontsize=12)


def _preset_figure(xlabel, ylabel, title, figsize):
    fig = plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig, fig.gca()


def list_cooling_model():
    """
    Print the formatted list of available cooling models.

    """
    if __dummy_cm.reader is None:
        __dummy_cm.reader = CoolingModelReader()

    return __dummy_cm.reader.list_cooling_model()


def list_cooling_parameters(model):
    """
    Print the formatted list of parameters available for the specified cooling
    models.

    Parameters
    ----------
    model: str
        Name of the cooling model as in the `model_list`.

    """
    if __dummy_cm.reader is None:
        __dummy_cm.reader = CoolingModelReader()

    return __dummy_cm.reader.list_cooling_parameters(model)


def list_atmosphere_parameters():
    """
    Print the formatted list of parameters available from the atmophere
    models.

    """
    if __dummy_atm.reader is None:
        __dummy_atm.reader = AtmosphereModelReader()

    return __dummy_atm.reader.list_atmosphere_parameters()


def plot_atmosphere_model(
    x="G3_BP-G3_RP",
    y="G3",
    atmosphere="H",
    independent=["logg", "Teff"],
    independent_values=[
        np.linspace(7.0, 9.0, 5),
        np.power(10.0, np.linspace(3.185, 5.165, 100)),
    ],
    interpolator="CT",
    contour=True,
    figsize=(8, 8),
    invert_xaxis=False,
    invert_yaxis=False,
    title=None,
    display=True,
    savefig=False,
    folder=None,
    filename=None,
    ext=["png"],
    fig=None,
    kwargs_for_plot={"marker": "+"},
    kwargs_for_contour={"levels": 100},
    kwargs_for_colorbar={},
):
    """
    Parameters
    ----------
    x: str (Default: 'G3_BP-G3_RP')
        Model parameter(s) of the abscissa. Two formats are supported:
        (1) single parameter (2) two parameters delimited with a '-' for the
        colour in those filters.
    y: str (Default: 'G3')
        Model parameter of the ordinate. Same as x.
    atmosphere: list of str (Default: 'H')
        Choose to plot from the pure hydrogen atmosphere model or pure
        helium atmosphere model. Only 1 atmosphere model can be plotted
        at a time, if both models are to be plotted, reuse the Figure object
        returned and overplot on it.
    independent: list of str (Default: ['logg', 'Teff'])
        Independent variables to be interpolated in the atmosphere model.
    independent_values: list of list or list of arrays (Default:
        [np.linspace(7.0, 9.0, 5), np.power(10., np.linspace(3.185, 5.165, 100))])
        The coordinates to be interpolated and plotted.
    interpolator: str (Default: 'CT')
        Choice of interpolator between CT and RBF.
    contour: bool (Default: True)
        Set to True to plot the contour levels.
    figsize: array of size 2 (Default: (8, 8))
        Set the dimension of the figure.
    invert_xaxis: bool (Default: False)
        Set to invert the abscissa.
    invert_yaxis: bool (Default: False)
        Set to invert the ordinate.
    title: str (Default: None)
        Set the title of the figure.
    display: bool (Default: True)
        Set to display the figure.
    savefig: bool (Default: False)
        Set to save the figure.
    folder: str (Default: None)
        The relative or absolute path to destination, the current working
        directory will be used if None.
    filename: str (Default: None)
        The filename of the figure. The Default filename will be used
        if None.
    ext: str (Default: ['png'])
        Image type to be saved, multiple extensions can be provided. The
        supported types are those available in `matplotlib.pyplot.savefig`.
    fig: matplotlib.figure.Figure (Default: None)
        Overplotting on an existing Figure.
    kwargs_for_plot: dict (Default: {'marker': '+'})
        Keywords for matplotlib.pyplot.plot().
    kwargs_for_contour: dict (Default: {'levels': 100})
        Keywords for matplotlib.pyplot.tricontourf().
    kwargs_for_colorbar: dict (Default: {})
        Keywords for matplotlib.pyplot.colorbar().

    """

    if __dummy_atm.reader is None:
        __dummy_atm.reader = AtmosphereModelReader()

    x = x.split("-")
    y = y.split("-")

    if len(x) == 2:
        x_name = (
            __dummy_atm.reader.column_names[x[0]]
            + r" $-$ "
            + __dummy_atm.reader.column_names[x[1]]
            + " / "
            + __dummy_atm.reader.column_units[x[0]]
        )

    else:
        x_name = (
            __dummy_atm.reader.column_names[x[0]]
            + " / "
            + __dummy_atm.reader.column_units[x[0]]
        )

    if len(y) == 2:
        y_name = (
            __dummy_atm.reader.column_names[y[0]]
            + r" $-$ "
            + __dummy_atm.reader.column_names[y[1]]
            + " / "
            + __dummy_atm.reader.column_units[y[0]]
        )

    else:
        y_name = (
            __dummy_atm.reader.column_names[y[0]]
            + " / "
            + __dummy_atm.reader.column_units[y[0]]
        )

    if title is None:
        if atmosphere in ["H", "h", "hydrogen", "Hydrogen", "da", "DA"]:
            title = "DA (Montreal)"

        if atmosphere in ["He", "he", "helium", "Helium", "db", "DB"]:
            title = "DB (Montreal)"

    x_out = []
    y_out = []

    for i_v in independent_values[0]:
        if len(x) == 2:
            x0_itp = __dummy_atm.reader.interp_am(
                dependent=x[0],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            x1_itp = __dummy_atm.reader.interp_am(
                dependent=x[1],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            x_out.append(
                x0_itp(i_v, independent_values[1])
                - x1_itp(i_v, independent_values[1])
            )

        else:
            x_itp = __dummy_atm.reader.interp_am(
                dependent=x[0],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            x_out.append(x_itp(i_v, independent_values[1]))

        if len(y) == 2:
            y0_itp = __dummy_atm.reader.interp_am(
                dependent=y[0],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            y1_itp = __dummy_atm.reader.interp_am(
                dependent=y[1],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            y_out.append(
                y0_itp(i_v, independent_values[1])
                - y1_itp(i_v, independent_values[1])
            )

        else:
            y_itp = __dummy_atm.reader.interp_am(
                dependent=y[0],
                atmosphere=atmosphere,
                independent=independent,
                interpolator=interpolator,
            )
            y_out.append(y_itp(i_v, independent_values[1]))

    if fig is not None:
        axes = fig.gca()

    else:
        fig, axes = _preset_figure(x_name, y_name, title, figsize)

    for i, (x_i, y_i) in enumerate(zip(x_out, y_out)):
        label = (
            __dummy_atm.reader.column_names[independent[0]]
            + f" = {independent_values[0][i]:.2f}"
        )
        axes.plot(x_i, y_i, label=label, **kwargs_for_plot)

    if contour:
        contourf_ = axes.tricontourf(
            np.array(x_out).flatten(),
            np.array(y_out).flatten(),
            np.array(
                [independent_values[1]] * len(independent_values[0])
            ).flatten(),
            **kwargs_for_contour,
        )
        fig.colorbar(contourf_, **kwargs_for_colorbar)

    plt.grid()
    plt.legend()

    if invert_xaxis:
        axes.invert_xaxis()

    if invert_yaxis:
        axes.invert_yaxis()

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
                _filename = title + "_" + y_name + "_" + x_name + "." + _e

            else:
                _filename = filename + "." + _e

            plt.savefig(os.path.join(_folder, _filename))

    if display:
        plt.show()

    return fig


def plot_cooling_model(
    model="montreal_co_da_20",
    x="age",
    y="lum",
    log_x=True,
    log_y=True,
    mass="all",
    figsize=(8, 8),
    invert_xaxis=False,
    invert_yaxis=False,
    title=None,
    display=True,
    savefig=False,
    folder=None,
    filename=None,
    ext=["png"],
    fig=None,
    kwargs_for_plot={"marker": "+"},
):
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
        6. 'lpcode_co_da_10_z001' - Renedo et al. 2010 CO DA Z=0.01
        7. 'lpcode_co_da_10_z0001' - Renedo et al. 2010 CO DA Z=0.001
        8. 'lpcode_co_da_15_z00003' - Althaus et al. 2015 DA Z=0.00003
        9. 'lpcode_co_da_15_z0001' - Althaus et al. 2015 DA Z=0.0001
        10. 'lpcode_co_da_15_z0005' - Althaus et al. 2015 DA Z=0.0005
        11. 'lpcode_co_da_17_y04' - Althaus et al. 2017 DB Y=0.4
        12. 'lpcode_co_db_17' - Camisassa et al. 2017 DB
        13. 'basti_co_da_10' - Salaris et al. 2010 CO DA
        14. 'basti_co_db_10' - Salaris et al. 2010 CO DB
        15. 'basti_co_da_10_nps' - Salaris et al. 2010 CO DA,
            no phase separation
        16. 'basti_co_db_10_nps' - Salaris et al. 2010 CO DB,
            no phase separation
        17. 'lpcode_one_da_07' - Althaus et al. 2007 ONe DA
        18. 'lpcode_one_da_19' - Camisassa et al. 2019 ONe DA
        19. 'lpcode_one_db_19' - Camisassa et al. 2019 ONe DB
        20. 'mesa_one_da_18' - Lauffer et al. 2018 ONe DA
        21. 'mesa_one_db_18' - Lauffer et al. 2018 ONe DB

        The naming convention follows this format:
        [model]_[core composition]_[atmosphere]_[publication year]
        where a few models continue to have extra property description
        terms trailing after the year, currently they are either the
        progenitor metallicity or the (lack of) phase separation in the
        evolution model.
    x: str (Default: 'age')
        Model parameter(s) of the abscissa.
    y: str (Default: 'lum')
        Model parameter of the ordinate.
    log_x: bool (Default: True)
        Set to True to log the abscissa.
    log_y: bool (Default: True)
        Set to True to log the ordinate.
    mass: str (Default: 'all')
        A list of mass in which the cooling model should return.
        Default is 'all', this is the only accept str.
    figsize: array of size 2 (Default: (8, 8))
        Set the dimension of the figure.
    invert_xaxis: bool (Default: False)
        Set to invert the abscissa.
    invert_yaxis: bool (Default: False)
        Set to invert the ordinate.
    title: str (Default: None)
        Set the title of the figure.
    display: bool (Default: True)
        Set to display the figure.
    savefig: bool (Default: False)
        Set to save the figure.
    folder: str (Default: None)
        The relative or absolute path to destination, the current working
        directory will be used if None.
    filename: str (Default: None)
        The filename of the figure. The Default filename will be used
        if None.
    ext: str (Default: ['png'])
        Image type to be saved, multiple extensions can be provided. The
        supported types are those available in `matplotlib.pyplot.savefig`.
    fig: matplotlib.figure.Figure (Default: None)
        Overplotting on an existing Figure.
    kwargs_for_plot={'marker': '+'}):
        Keywords for matplotlib.pyplot.plot().

    """

    if __dummy_cm.reader is None:
        __dummy_cm.reader = CoolingModelReader()

    (
        _mass_list,
        cooling_model,
        column_names,
        column_units,
    ) = __dummy_cm.reader.get_cooling_model(model)

    x_name = column_names[x]

    if column_units[x] != "":
        x_name = x_name + " / " + column_units[x]

    y_name = column_names[y]

    if column_units[y] != "":
        y_name = y_name + " / " + column_units[y]

    if title is None:
        title = f"Cooling Model - {model}"

    if fig is not None:
        axes = fig.gca()

    else:
        fig, axes = _preset_figure(x_name, y_name, title, figsize)

    if mass == "all":
        mass_list = _mass_list

    else:
        mass_list = mass

    x_out = []
    y_out = []

    for i, mass in enumerate(mass_list):
        x_out.append(cooling_model[i][x])
        y_out.append(cooling_model[i][y])

        label = "Mass = {mass:.2f}"
        axes.plot(x_out[i], y_out[i], label=label, **kwargs_for_plot)

    if log_x:
        axes.set_xscale("log")

    if log_y:
        axes.set_yscale("log")

    if invert_xaxis:
        axes.invert_xaxis()

    if invert_yaxis:
        axes.invert_yaxis()

    plt.grid()
    plt.legend()
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
                _filename = title + "_" + y_name + "_" + x_name + "." + _e

            else:
                _filename = filename + "." + _e

            plt.savefig(os.path.join(_folder, _filename))

    if display:
        plt.show()

    return fig
