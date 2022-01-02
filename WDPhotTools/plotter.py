import numpy as np
from matplotlib import pyplot as plt

from .atmosphere_model_reader import atm_reader
from .cooling_model_reader import list_cooling_model as lcm
from .cooling_model_reader import list_cooling_parameters as lcp
from .cooling_model_reader import get_cooling_model


class Dummy:
    pass


__dummy = Dummy()
__dummy.ar = None

plt.rc('font', size=18)
plt.rc('legend', fontsize=12)


def _preset_figure(xlabel, ylabel, title, figsize):
    fig = plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig, fig.gca()


def list_cooling_model():
    '''
    Print the formatted list of available cooling models.

    '''
    return lcm()


def list_cooling_parameters(model):
    '''
    Print the formatted list of parameters available for the specified cooling
    models.

    Parameters
    ----------
    model: str
        Name of the cooling model as in the `model_list`.

    '''
    return lcp(model)


def list_atmosphere_parameters():
    '''
    Print the formatted list of parameters available from the atmophere
    models.

    '''
    if __dummy.ar is None:

        __dummy.ar = atm_reader()

    return __dummy.ar.list_atmosphere_parameters()


def plot_atmosphere_model(x='G3_BP-G3_RP',
                          y='G3',
                          atmosphere='H',
                          independent=['logg', 'Teff'],
                          independent_values=[
                              np.linspace(7.0, 9.0, 5),
                              10.**np.linspace(3.17610, 5.17609, 101)
                          ],
                          contour=True,
                          figsize=(8, 8),
                          title=None,
                          display=True,
                          savefig=False,
                          filename=None,
                          ext=['png'],
                          fig=None,
                          kwargs_for_plot={'marker': '+'},
                          kwargs_for_contour={'levels': 100},
                          kwargs_for_colorbar={}):
    """
    Paramters
    ---------
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
    independent_values: list of list or list of arrays
        (Default: [np.linspace(7.0, 9.0, 11),
         10.**np.linspace(3.17610, 5.17609, 101)])
    contour: bool (Default: True)
        Set to True to plot the contour levels.
    figsize: array of size 2 (Default: (8, 8))
        Set the dimension of the figure.
    title: str (Default: None)
        Set the title of the figure.
    display: bool (Default: True)
        Set to display the figure.
    savefig: bool (Default: False)
        Set to save the figure.
    filename: str (Default: None)
        The filename (relative path) of the figure.
    ext: str (Default: ['png'])
        Image type to be saved, multiple extensions can be provided. The
        supported types are those available in `matplotlib.pyplot.savefig`.
    fig: matplotlib.figure.Figure (Default: None)
        Overplotting on an existing Figure.
    **kwargs_for_plot: dict (Default: {'marker': '+'})

    **kwargs_for_contour: dict (Default: {'levels': 100})

    **kwargs_for_colorbar: dict (Default: {})


    """

    if __dummy.ar is None:

        __dummy.ar = atm_reader()

    x = x.split('-')
    y = y.split('-')

    if len(x) == 2:

        x_name = __dummy.ar.column_names[
            x[0]] + r' $-$ ' + __dummy.ar.column_names[
                x[1]] + ' / ' + __dummy.ar.column_units[x[0]]

    else:

        x_name = __dummy.ar.column_names[
            x[0]] + ' / ' + __dummy.ar.column_units[x[0]]

    if len(y) == 2:

        y_name = __dummy.ar.column_names[
            y[0]] + r' $-$ ' + __dummy.ar.column_names[
                y[1]] + ' / ' + __dummy.ar.column_units[y[0]]

    else:

        y_name = __dummy.ar.column_names[
            y[0]] + ' / ' + __dummy.ar.column_units[y[0]]

    if title is None:

        if atmosphere in ['H', 'h', 'hydrogen', 'Hydrogen', 'da', 'DA']:

            title = 'DA (Montreal)'

        if atmosphere in ['He', 'he', 'helium', 'Helium', 'db', 'DB']:

            title = 'DB (Montreal)'

    x_out = []
    y_out = []

    for i_v in independent_values[0]:

        if len(x) == 2:

            x0_itp = __dummy.ar.interp_atm(dependent=x[0],
                                           atmosphere=atmosphere,
                                           independent=independent)
            x1_itp = __dummy.ar.interp_atm(dependent=x[1],
                                           atmosphere=atmosphere,
                                           independent=independent)
            x_out.append(
                x0_itp(i_v, independent_values[1]) -
                x1_itp(i_v, independent_values[1]))

        else:

            x_itp = __dummy.ar.interp_atm(dependent=x[0],
                                          atmosphere=atmosphere,
                                          independent=independent)
            x_out.append(x_itp(i_v, independent_values[1]))

        if len(y) == 2:

            y0_itp = __dummy.ar.interp_atm(dependent=y[0],
                                           atmosphere=atmosphere,
                                           independent=independent)
            y1_itp = __dummy.ar.interp_atm(dependent=y[1],
                                           atmosphere=atmosphere,
                                           independent=independent)
            y_out.append(
                y0_itp(i_v, independent_values[1]) -
                y1_itp(i_v, independent_values[1]))

        else:

            y_itp = __dummy.ar.interp_atm(dependent=y[0],
                                          atmosphere=atmosphere,
                                          independent=independent)
            y_out.append(y_itp(i_v, independent_values[1]))

    if fig is not None:

        ax = fig.gca()

    else:

        fig, ax = _preset_figure(x_name, y_name, title, figsize)

    for i in range(len(x_out)):

        label = __dummy.ar.column_names[independent[0]] + ' = {:.2f}'.format(
            independent_values[0][i])
        ax.plot(x_out[i], y_out[i], label=label, **kwargs_for_plot)

    if __dummy.ar.column_units[y[0]] == 'mag':

        ax.invert_yaxis()

    if contour:

        contourf_ = ax.tricontourf(
            np.array(x_out).flatten(),
            np.array(y_out).flatten(),
            np.array([independent_values[1]] *
                     len(independent_values[0])).flatten(),
            **kwargs_for_contour)
        fig.colorbar(contourf_, **kwargs_for_colorbar)

    plt.grid()
    plt.legend()
    plt.tight_layout()

    if savefig:

        if isinstance(ext, str):

            ext = [ext]

        # Loop through the ext list to save figure into each image type
        for e in ext:

            if filename is None:

                _filename = title + '_' + y_name + '_' + x_name + '.' + e

            else:

                _filename = filename + '.' + e

            plt.savefig(_filename)

    if display:

        plt.show()

    return fig


def plot_cooling_model(model='montreal_co_da_20',
                       x='age',
                       y='lum',
                       log_x=True,
                       log_y=True,
                       mass='all',
                       figsize=(8, 8),
                       title=None,
                       display=True,
                       savefig=False,
                       filename=None,
                       ext=['png'],
                       fig=None,
                       kwargs_for_plot={'marker': '+'}):
    '''
    Set the WD cooling model.

    Parameter
    ---------
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
        13. 'basti_co_da_10' - Salari et al. 2010 CO DA
        14. 'basti_co_db_10' - Salari et al. 2010 CO DB
        15. 'basti_co_da_10_nps' - Salari et al. 2010 CO DA,
                                    no phase separation
        16. 'basti_co_db_10_nps' - Salari et al. 2010 CO DB,
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
    mass_range: str (Default: 'all')
        The mass range in which the cooling model should return.
        The ranges are defined as <0.5, 0.5-1.0 and >1.0 solar masses.
    figsize: array of size 2 (Default: (8, 8))
        Set the dimension of the figure.
    title: str (Default: None)
        Set the title of the figure.
    display: bool (Default: True)
        Set to display the figure.
    savefig: bool (Default: False)
        Set to save the figure.
    filename: str (Default: None)
        The filename (relative path) of the figure.
    ext: str (Default: ['png'])
        Image type to be saved, multiple extensions can be provided. The
        supported types are those available in `matplotlib.pyplot.savefig`.
    fig: matplotlib.figure.Figure (Default: None)
        Overplotting on an existing Figure.
    kwargs_for_plot={'marker': '+'}):

    '''

    _mass_list, cooling_model, column_names, column_units = get_cooling_model(
        model)

    x_name = column_names[x]

    if column_units[x] != '':

        x_name = x_name + ' / ' + column_units[x]

    y_name = column_names[y]

    if column_units[y] != '':

        y_name = y_name + ' / ' + column_units[y]

    if title is None:

        title = 'Cooling Model - {}'.format(model)

    if fig is not None:

        ax = fig.gca()

    else:

        fig, ax = _preset_figure(x_name, y_name, title, figsize)

    if mass == 'all':

        mass_list = _mass_list

    else:

        mass_list = mass

    x_out = []
    y_out = []

    for i, m in enumerate(mass_list):

        x_out.append(cooling_model[i][x])
        y_out.append(cooling_model[i][y])

        label = 'Mass = {:.2f}'.format(m)
        ax.plot(x_out[i], y_out[i], label=label, **kwargs_for_plot)

    if log_x:

        ax.set_xscale('log')

    if log_y:

        ax.set_yscale('log')

    plt.grid()
    plt.legend()
    plt.tight_layout()

    if savefig:

        if isinstance(ext, str):

            ext = [ext]

        # Loop through the ext list to save figure into each image type
        for e in ext:

            if filename is None:

                _filename = title + '_' + y_name + '_' + x_name + '.' + e

            else:

                _filename = filename + '.' + e

            plt.savefig(_filename)

    if display:

        plt.show()

    return fig
