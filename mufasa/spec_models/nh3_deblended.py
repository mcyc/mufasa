__author__ = 'mcychen'


"""
==================================================================
Ammonia inversion transition: deblended fitter (Hyperfine-removed)
==================================================================
"""

#=======================================================================================================================

from pyspeckit.spectrum.models import hyperfine
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict, aval_dict, ortho_dict, voff_lines_dict,
                                                         tau_wts_dict, line_labels)

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