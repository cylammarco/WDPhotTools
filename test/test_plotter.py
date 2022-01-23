import numpy as np
from WDPhotTools import plotter
from WDPhotTools.cooling_model_reader import model_list


def test_list_everything():
    plotter.list_atmosphere_parameters()
    plotter.list_cooling_model()
    for i in model_list.keys():
        plotter.list_cooling_parameters(i)


def test_plot_atmosphere_model_with_ext_as_str():
    plotter.plot_atmosphere_model(display=False,
                                  savefig=True,
                                  folder='test_output',
                                  filename='test_plot_atmosphere_model',
                                  ext='png')


def test_plot_atmosphere_model():
    plotter.plot_atmosphere_model(display=False,
                                  savefig=True,
                                  folder='test_output',
                                  filename='test_plot_atmosphere_model',
                                  ext=['png', 'pdf'])


def test_plot_atmosphere_model_different_filters():
    plotter.plot_atmosphere_model(x='B-V',
                                  y='U',
                                  invert_yaxis=True,
                                  display=False)


def test_plot_atmosphere_model_color_color_diagram():
    plotter.plot_atmosphere_model(x='U-V', y='B-V', display=False)


def test_plot_atmosphere_model_with_differnt_independent_variables():
    plotter.plot_atmosphere_model(independent=['logg', 'Mbol'],
                                  independent_values=[
                                      np.linspace(7.0, 9.0, 5),
                                      np.linspace(2.0, 18.0, 101)
                                  ],
                                  invert_xaxis=True,
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
    plotter.plot_cooling_model(display=False,
                               savefig=True,
                               folder='test_output',
                               filename='cooling_model',
                               ext='png')


def test_plot_cooling_model_invert_axis():
    plotter.plot_cooling_model(x='r',
                               y='logg',
                               mass=np.arange(0.5, 1.0, 0.1),
                               invert_xaxis=True,
                               invert_yaxis=True,
                               display=False,
                               savefig=True,
                               folder='test_output',
                               filename='cooling_model_r_logg',
                               ext=['png', 'pdf'])

# YKW 19JAN2022 1
def test_plot_atmosphere_models_he_title_none():
    fig = plotter.plot_atmosphere_model(display=False, title=' ')
    plotter.plot_atmosphere_model(atmosphere='He',
                                  invert_yaxis=True,
                                  contour=False,
                                  display=False,
                                  title=None,
                                  fig=fig)

# YKW 19JAN2022 2
def test_plot_atmosphere_models_none_folder_savefig():
    plotter.plot_atmosphere_model(display=False,
                                  savefig=True,
                                  folder=None,
                                  filename='test_plot_atmosphere_model',
                                  ext='png')

# YKW 19JAN2022 3
def test_plot_cooling_model_fig_none():
    fig = plotter.plot_cooling_model(display=False, title=' ')
    plotter.plot_cooling_model(display=False,
                               savefig=True,
                               folder='test_output',
                               filename='cooling_model_ykw_1',
                               ext='png',
                               fig=fig)

# YKW 19JAN2022 4
def test_plot_cooling_model_folder_none():
    fig = plotter.plot_cooling_model(display=False, title=' ')
    plotter.plot_cooling_model(display=False,
                               savefig=True,
                               filename='cooling_model_ykw_2',
                               ext='png',
                               fig=fig)

# YKW 19JAN2022 5
def test_plot_atmosphere_models_lenx_not_2():
    plotter.plot_atmosphere_model(x='G3_BP',
                                  display=False,
                                  savefig=False)

# YKW 23JAN2022 1
def test_plot_cooling_model_filename_none():
    fig = plotter.plot_cooling_model(display=False, title=' ')
    plotter.plot_cooling_model(display=False,
                               savefig=True,
                               filename=None,
                               ext='png',
                               fig=fig)

# YKW 23JAN2022 2
def test_plot_atmosphere_models_filename_none():
    fig = plotter.plot_atmosphere_model(display=False, title=' ')
    plotter.plot_atmosphere_model(atmosphere='He',
                                  invert_yaxis=True,
                                  contour=False,
                                  display=False,
                                  title=None,
                                  filename=None,
                                  fig=fig)