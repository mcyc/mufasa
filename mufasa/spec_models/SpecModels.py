"""
Spectral models for multi-component fitting, including ammonia (NH₃) and aiazenylium (N₂H⁺).
"""

from .HyperfineModel import HyperfineModel

class AmmoniaModel(HyperfineModel):
    """
    Ammonia (NH₃) spectral model for multi-component fitting.

    Inherits from `HyperfineModel` and applies ammonia-specific molecular constants.
    """
    from .m_constants import nh3_constants as _nh3_constants
    _molecular_constants = _nh3_constants

    def __init__(self, line_names=['oneone']):
        """
        Initialize the ammonia model.

        Parameters
        ----------
        line_names : list of str, optional
            Ammonia spectral line names (default: ['oneone']).
        """
        super().__init__(line_names)


class N2HplusModel(HyperfineModel):
    """
    N₂H⁺ (Diazenylium) spectral model for multi-component fitting.

    Inherits from `HyperfineModel` and applies N₂H⁺-specific molecular constants.
    """
    from .m_constants import n2hp_constants as _n2hp_constants
    _molecular_constants = _n2hp_constants

    def __init__(self, line_names=['onezero']):
        """
        Initialize the N₂H⁺ model.

        Parameters
        ----------
        line_names : list of str, optional
            N₂H⁺ spectral line names (default: ['onezero']).
        """
        super().__init__(line_names=line_names)
