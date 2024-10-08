#Exceptions and Warnings (:mod:`numpy.exceptions`)
#=================================================

class SNRMaskError(Exception):
    """SNR Mask has no valid pixel.

    This is raised whenever the snr_min provided results in masks with zero valid pixel

    """
    pass


class FitTypeError(LookupError):
    """Fitttype provided is not valid.

    This is raised whenever the fittype specified by the user is invalid

    """
    pass

class StartFitError(Exception):
    """Fitting failed from the beginning

    This is raised whenever pcube.fiteach raises the follwoing:
    AssertionError: The first fitted pixel did not yield a fit. Please try starting from a different pixel.

    """
    pass