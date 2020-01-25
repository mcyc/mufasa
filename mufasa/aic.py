__author__ = 'mcychen'

import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube

import multi_v_fit as mvf

# a place to calculate AIC values

#=======================================================================================================================

def fits_comp_AICc(cubepath, modpath1, modpath2, aiccpath, likelihoodpath = None):
    # a wrapper around fits_comp_AICc() to work with fits files

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


def fits_comp_chisq(cubepath, modpath1, modpath2, savepath, reduced = True):
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
    '''
    Acquire AICc values over the same samples, where BOTH of the models have none-zero values
    :param cube: <SpectralCube>
        The data cube
    :param model1: <numpy array>
        The 1st model cube
    :param model2: <numpy array>
        The 2nd model cube
    :param p1:
        Number of parameters associated with model 1
    :param p2:
        Number of parameters associated with model 2
    :return:
    '''

    mask1 = model1 > 0
    mask2 = model2 > 0
    mask = np.logical_or(mask1, mask2)

    chi1, N1 = mvf.get_chisq(cube, model1, expand=20, reduced = False, usemask = True, mask = mask)
    chi2, N2 = mvf.get_chisq(cube, model2, expand=20, reduced = False, usemask = True, mask = mask)

    # I need a way to double check that N1 and N2 are the same (just in case)
    aicc1 = AICc(chi1, p1, N1)
    aicc2 = AICc(chi2, p2, N1)

    return aicc1, aicc2


def AIC(chisq, p):
    '''
    Calculate the Akaike information criterion based on the provided chi-squared values
    :param chisq:
        Chi-squared values
    :param p:
        Number of parameters
    :return:
    '''
    return chisq + 2*p


def AICc(chisq, p, N):
    '''
    Calculate the corrected Akaike information criterion based on the provided chi-squared values
    corrected AIC (AICc) approaches that of the AIC value when chisq >> p^2
    :param chisq:
        Chi-squared values
    :param p:
        Number of parameters
    :param N:
    :return:
    '''
    top = 2*p*(p+1)
    bottom = N - p - 1
    return AIC(chisq, p) + top/bottom


def likelihood(aiccA, aiccB):
    # return the log likelihood of A relative to B
    return -1.0*(aiccA - aiccB) / 2.0


