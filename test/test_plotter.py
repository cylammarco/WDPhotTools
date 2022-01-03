import numpy as np
from WDPhotTools import plotter
from WDPhotTools.cooling_model_reader import model_list


def test_list_everything():
    plotter.list_atmosphere_parameters()
    plotter.list_cooling_model()
    for i in model_list.keys():
        plotter.list_cooling_parameters(i)


def test_plot_atmosphere_model():
    plotter.plot_atmosphere_model(display=False)


def test_plot_atmosphere_model_different_filters():
    plotter.plot_atmosphere_model(x='U', y='B-V', display=False)


def test_plot_atmosphere_model_with_differnt_independent_variables():
    plotter.plot_atmosphere_model(independent=['logg', 'Mbol'],
                                  independent_values=[
                                      np.linspace(7.0, 9.0, 5),
                                      np.linspace(2.0, 18.0, 101)
                                  ],
                                  invert_yaxis=True,
                                  display=False)


def test_plot_2_atmosphere_models():
    fig = plotter.plot_atmosphere_model(display=False, title=' ')
    plotter.plot_atmosphere_model(atmosphere='He',
                                  invert_yaxis=True,
                                  contour=False,
                                  display=False,
                                  title='DA + DB (Montreal)',
                                  fig=fig)


def test_plot_cooling_model():
    plotter.plot_cooling_model(display=False)
