from WDPhotTools import plotter
from WDPhotTools.cooling_model_reader import model_list


def test_list_everything():
    plotter.list_atmosphere_parameters()
    plotter.list_cooling_model()
    for i in model_list.keys():
        plotter.list_cooling_parameters(i)
