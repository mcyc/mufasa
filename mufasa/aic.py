from __future__ import absolute_import
__author__ = 'mcychen'

import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube

from . import multi_v_fit as mvf

# a place to calculate AIC values

#=======================================================================================================================

def fits_comp_AICc(cubepath, modpath1, modpath2, aiccpath, likelihoodpath=None):
    """
    A wrapper function to calculate corrected Akaike Information Criterion (AICc) values
    and save them to a FITS file.

    Parameters
    ----------
    cubepath : str
        Path to the data cube FITS file.
    modpath1 : str
        Path to the first model FITS file.
    modpath2 : str
        Path to the second model FITS file.
    aiccpath : str
        Path to save the resulting AICc FITS file.
    likelihoodpath : str, optional
        Path to save the likelihood FITS file, if provided.

    Returns
    -------
    None
    """

    cube = SpectralCube.read(cubepath)
    mod1, hdr1 = fits.getdata(modpath1, header = True)
    mod2, hdr2 = fits.getdata(modpath2, header = True)

    aicc1, aicc2 = get_comp_AICc(cube, mod1, mod2, p1 = 4, p2 = 8)

    hdr_new = cube.wcs.celestial.to_header()
    hdr_new['PLANE1'] = "AICc values for the 1 component fit model"
    hdr_new['PLANE2'] = "AICc values for the 2 component fit model"

    aicccube = fits.PrimaryHDU(data=np.array([aicc1, aicc2]), header=hdr_new)
    aicccube.writeto(aiccpath, overwrite=True)

    if likelihoodpath is not None:
        likelyhood = (aicc1 - aicc2) / 2.0
        fits.writeto(likelihoodpath, likelyhood, cube.wcs.celestial.to_header(), overwrite=True)


def fits_comp_chisq(cubepath, modpath1, modpath2, savepath, reduced=True):
    """
    Calculate and save chi-squared values for the given cube and model fits.

    Parameters
    ----------
    cubepath : str
        Path to the data cube FITS file.
    modpath1 : str
        Path to the first model FITS file.
    modpath2 : str
        Path to the second model FITS file.
    savepath : str
        Path to save the resulting chi-squared FITS file.
    reduced : bool, optional
        Whether to calculate reduced chi-squared values. Default is True.

    Returns
    -------
    None
    """
    cube = SpectralCube.read(cubepath)
    mod1, hdr1 = fits.getdata(modpath1, header = True)
    mod2, hdr2 = fits.getdata(modpath2, header = True)

    hdr_new = cube.wcs.celestial.to_header()
    hdr_new['PLANE1'] = "reduced chi-squared values for the 1 component fit model"
    hdr_new['PLANE2'] = "reduced chi-squared values for the 2 component fit model"

    mask1 = mod1 > 0
    mask2 = mod2 > 0
    mask = np.logical_or(mask1, mask2)

    # expand of 20 is same as that used to calculate aic value
    chi1 = mvf.get_chisq(cube, mod1, expand=20, reduced = reduced, usemask = True, mask = mask)
    chi2 = mvf.get_chisq(cube, mod2, expand=20, reduced = reduced, usemask = True, mask = mask)

    chicube = fits.PrimaryHDU(data=np.array([chi1, chi2]), header=cube.wcs.celestial.to_header())
    chicube.writeto(savepath, overwrite=True)


def get_comp_AICc(cube, model1, model2, p1, p2):
    """
    Calculate AICc values for two models over the same samples.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    model1 : numpy.ndarray
        The first model cube.
    model2 : numpy.ndarray
        The second model cube.
    p1 : int
        Number of parameters associated with the first model.
    p2 : int
        Number of parameters associated with the second model.

    Returns
    -------
    tuple of numpy.ndarray
        AICc values for the first and second models.
    """
    mask1 = model1 > 0
    mask2 = model2 > 0
    mask = np.logical_or(mask1, mask2)

    chi1, N1 = mvf.get_chisq(cube, model1, expand=20, reduced = False, usemask = True, mask = mask)
    chi2, N2 = mvf.get_chisq(cube, model2, expand=20, reduced = False, usemask = True, mask = mask)

    # I need a way to double check that N1 and N2 are the same (just in case)
    aicc1 = AICc(chi1, p1, N1)
    aicc2 = AICc(chi2, p2, N1)

    return aicc1, aicc2


def AIC(rss, p, N):
    """
    Calculate the Akaike Information Criterion (AIC).

    Parameters
    ----------
    rss : numpy.ndarray
        Residual sum of squares.
    p : int
        Number of parameters.
    N : int
        Number of samples.

    Returns
    -------
    numpy.ndarray
        AIC values.
    """
    # avoid invalid math values
    N[N==0] = np.nan
    aic = N * np.log(rss/N) + 2*p
    #return np.nan_to_num(aic)
    return aic


def AICc(rss, p, N):
    """
    Calculate the corrected Akaike Information Criterion (AICc).

    Parameters
    ----------
    rss : numpy.ndarray
        Residual sum of squares.
    p : int
        Number of parameters.
    N : int
        Number of samples.

    Returns
    -------
    numpy.ndarray
        Corrected AICc values.
    """
    top = 2*p*(p+1)
    bottom = N - p - 1
    return AIC(rss, p, N) + top/bottom


def likelihood(aiccA, aiccB):
    """
    Calculate the log-likelihood of model A relative to model B.

    Parameters
    ----------
    aiccA : numpy.ndarray
        AICc values for model A.
    aiccB : numpy.ndarray
        AICc values for model B.

    Returns
    -------
    numpy.ndarray
        Log-likelihood values.
    """
    #aiccA, aiccB = np.nan_to_num(aiccA), np.nan_to_num(aiccB)
    return -1.0*(aiccA - aiccB) / 2.0


