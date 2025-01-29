"""
nh3_deblended.py

DEPRECATED: This module is deprecated and will be removed in v1.6.0.
Use the deblend() method of the 'SpecModels.AmmoniaModel' class instead.

"""

#=======================================================================================================================

from pyspeckit.spectrum.models import hyperfine
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict)

import warnings

warnings.warn(
    "The `nh3_deblended.py` module is deprecated and will be removed in v1.6.0."
    "Use the deblend() of a 'SpecModels.AmmoniaModel' object instead.",
    DeprecationWarning,
    stacklevel=2
)


#=======================================================================================================================


tau0 = 1.0

# represent the tau profile of nh3 spectra as a single Gaussian for each individual velocity slab
nh3_vtau_deblended = {linename:
            hyperfine.hyperfinemodel({0:0},
                                     {0:0.0},
                                     {0:freq_dict[linename]},
                                     {0:tau0},
                                     {0:1},
                                    )
            for linename in line_names}


def nh3_vtau_singlemodel_deblended(xarr, Tex, tau, xoff_v, width, linename = 'oneone'):
    # the parameters are in the order of vel, width, tex, tau for each velocity component
    return nh3_vtau_deblended[linename].hyperfine(xarr, Tex=Tex, tau=tau, xoff_v=xoff_v, width=width)