
from .HyperfineModel import HyperfineModel

class AmmoniaModel(HyperfineModel):
    """
    Model class for ammonia multi-component spectral fitting.

    Inherits from BaseModel and sets ammonia-specific molecular constants and line names.
    """

    # Set ammonia-specific molecular constants and default line names
    from .molecular_constants import nh3_constants
    molecular_constants = nh3_constants

    #from pyspeckit.spectrum.models.ammonia\
    #    import (freq_dict, voff_lines_dict, tau_wts_dict) #line_names

    # Set ammonia-specific molecular constants and default line names
    #molecular_constants = {
    #    'freq_dict': freq_dict,
    #    'voff_lines_dict': voff_lines_dict,
    #    'tau_wts_dict': tau_wts_dict
    #}

    def __init__(self, line_names=['oneone']):
        """
        Initialize the AmmoniaModel with specific line names.

        Parameters
        ----------
        line_names : list of str, optional
            List of ammonia line names (default is ['oneone']).
        """
        super().__init__(line_names)


class N2HplusModel(HyperfineModel):
    """
    Model class for N2H+ (Diazenylium) multi-component spectral fitting.

    Inherits from BaseModel and sets N2H+-specific molecular constants and line names.
    """

    #from .n2hp_constants import (freq_dict, voff_lines_dict, tau_wts_dict)
    from .molecular_constants import n2hp_constants

    # Set N2H+-specific molecular constants and default line names
    '''
    molecular_constants = {
        'freq_dict': freq_dict,
        'voff_lines_dict': voff_lines_dict,
        'tau_wts_dict': tau_wts_dict
    }
    '''
    molecular_constants =n2hp_constants

    def __init__(self, line_names=['onezero']):
        """
        Initialize the N2HplusModel with specific line names.

        Parameters
        ----------
        line_names : list of str, optional
            List of N2H+ line names (default is ['onezero']).
        """
        super().__init__(line_names=line_names)

