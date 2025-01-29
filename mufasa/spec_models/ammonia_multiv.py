from __future__ import print_function
__author__ = 'mcychen'

from .SpecModels import AmmoniaModel

nh3model = AmmoniaModel()

def nh3_multi_v_model_generator(n_comp, linenames=None):
    return nh3model.multi_v_model_generator(n_comp)

def ammonia_multi_v(xarr, *args, **kwargs):
    return nh3model.multi_v_spectrum(xarr, *args)

def _ammonia_spectrum(xarr, tex, tau_dict, width, xoff_v, line_names, background_ta=0.0, fillingfraction=None,
                      return_components=False):
    return nh3model._single_spectrum(xarr, tex, tau_dict, width, xoff_v, background_ta=background_ta)

def T_antenna(Tbright, nu):
    return nh3model.T_antenna(Tbright, nu)