
from .BaseModels import BaseModel

class AmmoniaModel(BaseModel):
    """
    Model class for ammonia multi-component spectral fitting.

    Inherits from BaseModel and sets ammonia-specific molecular constants and line names.
    """

    from pyspeckit.spectrum.models.ammonia\
        import (line_names, freq_dict, voff_lines_dict, tau_wts_dict)

    # Set ammonia-specific molecular constants and default line names
    molecular_constants = {
        'freq_dict': freq_dict,
        'voff_lines_dict': voff_lines_dict,
        'tau_wts_dict': tau_wts_dict
    }
    line_names = ['oneone']  # Default line name for ammonia

    def __init__(self, line_names=None):
        """
        Initialize the AmmoniaModel with specific line names.

        Parameters
        ----------
        line_names : list of str, optional
            List of ammonia line names (default is ['oneone']).
        """
        super().__init__(self.molecular_constants, line_names=line_names)



class N2HplusModel(BaseModel):
    """
    Model class for N2H+ (Diazenylium) multi-component spectral fitting.

    Inherits from BaseModel and sets N2H+-specific molecular constants and line names.
    """

    from n2hp_constants import (freq_dict, voff_lines_dict, tau_wts_dict)

    # Set N2H+-specific molecular constants and default line names
    molecular_constants = {
        'freq_dict': freq_dict,
        'voff_lines_dict': voff_lines_dict,
        'tau_wts_dict': tau_wts_dict
    }
    line_names = ['onezero']  # Default line name for N2H+

    def __init__(self, line_names=None):
        """
        Initialize the N2HplusModel with specific line names.

        Parameters
        ----------
        line_names : list of str, optional
            List of N2H+ line names (default is ['onezero']).
        """
        super().__init__(self.molecular_constants, line_names=line_names)

