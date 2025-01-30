"""
`Deprecated` - will be removed in v1.6.0.

.. deprecated:: 1.5.0
    This module is deprecated and will be removed in v1.6.0.
    Use instances of :class:`AmmoniaModel <mufasa.spec_models.SpecModels.AmmoniaModel>` instead.
"""

#===============================================================================================

import warnings

warnings.warn(
    "The `ammonia_multiv.py` module is deprecated and will be removed in v1.6.0."
    "Use the 'SpecModels.AmmoniaModel' class instead.",
    DeprecationWarning,
    stacklevel=2
)

#===============================================================================================

from .SpecModels import AmmoniaModel

nh3model = AmmoniaModel()

def nh3_multi_v_model_generator(n_comp, linenames=None):
    return nh3model.multi_v_model_generator(n_comp)

def ammonia_multi_v(xarr, *args, **kwargs):
    return nh3model.multi_v_spectrum(xarr, *args)

def _ammonia_spectrum(xarr, tex, tau_dict, width, xoff_v, line_names, background_ta=0.0, fillingfraction=None,
                      return_components=False):
    return nh3model._single_spectrum_hf(xarr, tex, tau_dict, width, xoff_v, background_ta=background_ta)

def T_antenna(Tbright, nu):
    return nh3model.T_antenna(Tbright, nu)