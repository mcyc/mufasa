from __future__ import print_function
__author__ = 'mcychen'

from .SpecModels import N2HplusModel

n2hp_model = N2HplusModel()

def n2hp_multi_v_model_generator(n_comp, linenames = None):
    return n2hp_model.multi_v_model_generator(n_comp)

def n2hp_multi_v(xarr, *args, **kwargs):
    return n2hp_model.multi_v_spectrum(xarr, *args)

def _n2hp_spectrum(xarr, tex, tau_dict, width, xoff_v, line_names, background_ta=0.0, fillingfraction=None,
                      return_components=False):
    return n2hp_model._single_spectrum(xarr, tex, tau_dict, width, xoff_v, background_ta=background_ta)

def T_antenna(Tbright, nu):
    return n2hp_model.T_antenna(Tbright, nu)