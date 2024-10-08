#Exceptions and Warnings (:mod:`numpy.exceptions`)
#=================================================

class SNRMaskError(Exception):
    """SNR Mask has no valid pixel.

    This is raised whenever the snr_min provided results in masks with zero valid pixel

    """
    pass


class FitTypeError((LookupError)):
    """Fitttype provided is not valid.

    This is raised whenever the fittype specified by the user is invalid

    """
    pass