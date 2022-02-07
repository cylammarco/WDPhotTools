import numpy as np
from unittest.mock import patch

from WDPhotTools import plotter
from WDPhotTools import cooling_model_reader as cmr

cr = cmr.cm_reader()


def test_list_everything():
    plotter.list_atmosphere_parameters()
    plotter.list_cooling_model()
    for i in cr.model_list.keys():
        plotter.list_cooling_parameters(i)


# Not displaying
def test_plot_atmosphere_model_with_ext_as_str():
    plotter.plot_atmosphere_model(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_plot_atmosphere_model",
        ext="png",
    )


@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_model_with_ext_as_str(mock_show):
    plotter.plot_atmosphere_model(
        display=True,
        savefig=True,
        folder="test_output",
        filename="test_plot_atmosphere_model",
        ext="png",
    )


@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_model(mock_show):
    plotter.plot_atmosphere_model(
        display=True,
        savefig=True,
        folder="test_output",
        filename="test_plot_atmosphere_model",
        ext=["png", "pdf"],
    )


@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_model_different_filters(mock_show):
    plotter.plot_atmosphere_model(
        x="B-V", y="U", invert_yaxis=True, display=True
    )


@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_model_color_color_diagram(mock_show):
    plotter.plot_atmosphere_model(x="U-V", y="B-V", display=True)


@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_model_with_differnt_independent_variables(mock_show):
    plotter.plot_atmosphere_model(
        independent=["logg", "Mbol"],
        independent_values=[
            np.linspace(7.0, 9.0, 5),
            np.linspace(2.0, 18.0, 101),
        ],
        invert_xaxis=True,
        invert_yaxis=True,
        display=True,
    )


@patch("matplotlib.pyplot.show")
def test_plot_2_atmosphere_models(mock_show):
    fig = plotter.plot_atmosphere_model(display=True, title=" ")
    plotter.plot_atmosphere_model(
        atmosphere="He",
        invert_yaxis=True,
        contour=False,
        display=True,
        title="DA + DB (Montreal)",
        fig=fig,
    )


@patch("matplotlib.pyplot.show")
def test_plot_cooling_model(mock_show):
    plotter.plot_cooling_model(
        display=True,
        savefig=True,
        folder="test_output",
        filename="cooling_model",
        ext="png",
    )


@patch("matplotlib.pyplot.show")
def test_plot_cooling_model_invert_axis(mock_show):
    plotter.plot_cooling_model(
        x="r",
        y="logg",
        mass=np.arange(0.5, 1.0, 0.1),
        invert_xaxis=True,
        invert_yaxis=True,
        display=True,
        savefig=True,
        folder="test_output",
        filename="cooling_model_r_logg",
        ext=["png", "pdf"],
    )


# YKW 19JAN2022 1
@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_models_he_title_none(mock_show):
    fig = plotter.plot_atmosphere_model(display=True, title=" ")
    plotter.plot_atmosphere_model(
        atmosphere="He",
        invert_yaxis=True,
        contour=False,
        display=True,
        title=None,
        folder="test_output",
        fig=fig,
    )


# YKW 19JAN2022 2
@patch("matplotlib.pyplot.show")
def test_plot_atmosphere_models_none_folder_savefig(mock_show):
    plotter.plot_atmosphere_model(
        display=True,
        savefig=True,
        folder=None,
        filename="test_plot_atmosphere_model",
        ext="png",
    )


# YKW 19JAN2022 3
@patch("matplotlib.pyplot.show")
def test_plot_cooling_model_fig_none(mock_show):
    fig = plotter.plot_cooling_model(display=True, title=" ")
    plotter.plot_cooling_model(
        display=True,
        savefig=True,
        folder="test_output",
        filename="cooling_model_ykw_1",
        ext="png",
        fig=fig,
    )


# YKW 19JAN2022 4
@patch("matplotlib.pyplot.show")
def test_plot_cooling_model_folder_none(mock_show):
    fig = plotter.plot_cooling_model(display=True, title=" ")
    plotter.plot_cooling_model(
        display=True,
        savefig=True,
        folder="test_output",
        filename="cooling_model_ykw_1",
        ext="png",
        fig=fig,
    )


# YKW 19JAN2022 5
def test_plot_atmosphere_models_lenx_not_2():
    plotter.plot_atmosphere_model(x="G3_BP", display=True, savefig=False)
