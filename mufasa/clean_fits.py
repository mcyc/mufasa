__author__ = 'mcychen'

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from reproject import reproject_interp
from astropy.wcs import WCS

from mufasa import slab_sort as sb
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#=======================================================================================================================

class fit_results(object):

    def __init__(self, path_para_1c, path_lnk01):

        self.paraCubes = {}
        self.headers = {}
        self.lnkMaps = {}

        self.paraCubes['1c'], self.headers['1c'] = fits.getdata(path_para_1c, header=True)
        self.lnkMaps['10'] = fits.getdata(path_lnk01)
        self.ncompList = [1]


    def add_results(self, path_para, path_lnk, ncomp):
        if ncomp > 1:
            self.paraCubes['{}c'.format(ncomp)] = fits.getdata(path_para)
            self.headers['{}c'.format(ncomp)] = fits.getheader(path_para)
            self.lnkMaps['{}{}'.format(ncomp, ncomp-1)] = fits.getdata(path_lnk)
            self.ncompList.append(ncomp)
        else:
            print("[ERROR]: ncomp must be >1. The provide value is {}. No action taken.".format(ncomp))

#=======================================================================================================================

def clean_2comp_maps(fit_results, savename=None, vErrThresh=None, removeExtremeV=True):
    fr=fit_results
    pmaps_1c = fr.paraCubes['1c'].copy()
    pmaps_2c = fr.paraCubes['2c'].copy()
    lnk21 = fr.lnkMaps['21']
    lnk10 =  fr.lnkMaps['10']

    remove_zeros(pmaps_1c)
    remove_zeros(pmaps_2c)

    # remove pixels better modeled by noise
    mask = lnk10 < 5
    pmaps_1c[:, mask] = np.nan
    #pmaps_2c[:, mask] = np.nan

    # remove pixels best modeled by one component
    mask = lnk21 < 5
    pmaps_2c[:, mask] = np.nan

    if removeExtremeV:
        # remove pixels with fitted vlsr & sigma that are 'stuck' at the extreme values
        mask_exmv_1c = extremeV_mask(pmaps_1c)
        mask_exmv_2c = extremeV_mask(pmaps_2c)

        pmaps_1c[:, mask_exmv_1c] = np.nan
        pmaps_2c[:, mask_exmv_2c] = np.nan

    if vErrThresh is not None:
        # remove pixels with fitted vlsr & sigma above the specified thresholds
        mask_thr_1c = above_ErrV_Thresh(pmaps_1c, vErrThresh)
        mask_thr_2c = above_ErrV_Thresh(pmaps_2c, vErrThresh)

        pmaps_1c[:, mask_thr_1c] = np.nan
        pmaps_2c[:, mask_thr_2c] = np.nan

    # replace two comp fit with 1 comp fit over where lnk21 < 5 and where the cleaned 1 comp map is finite
    #mmask = np.all(np.isfinite(pmaps_1c), axis=0)
    #mmask = np.logical_or(mmask, np.all(np.isfinite(pmaps_2c), axis=0))
    #mmask = np.logical_and(mmask, lnk21 < 5)
    #return mmask

    mmask = ~np.all(np.isfinite(pmaps_2c), axis=0)
    pmaps_2c[0:4, mmask] = pmaps_1c[0:4, mmask]  # fitted parameters
    pmaps_2c[8:12, mmask] = pmaps_1c[4:8, mmask]  # fitted errors


    if savename is not None:
        fits.writeto(savename, data=pmaps_2c, header=fr.headers['2c'], overwrite=True)

    return pmaps_2c


def remove_zeros(pmaps):
    pmaps[pmaps==0] = np.nan


def extremeV_mask(pmaps):
    # find pixels with extreme fitted vlsr or sigma values

    ncomp = pmaps.shape[0]/8

    # create a list of indices for all vlsr and sigma maps
    iList = []
    for i in range(ncomp):
        iList.append(i*4)     # add vlsr
        iList.append(i*4+1)   # add sigma

    # create an empty mask
    mmask = np.zeros(pmaps[0].shape, dtype='bool')
    for i in iList:
        mask1 = pmaps[i] == np.nanmin(pmaps[i])
        #print(np.sum(mask1))
        mask2 = pmaps[i] == np.nanmax(pmaps[i])
        #print(np.sum(mask2))
        mask = np.logical_or(mask1, mask2)
        mmask = np.logical_or(mmask, mask)

    return mask


def above_ErrV_Thresh(pmaps, thresh):
    # return a mask indicating the pixels with vlsr and sigma errors above the threshold

    ncomp = pmaps.shape[0]/8

    # create a list of indices for all vlsr and sigma maps
    iList = []
    for i in range(ncomp):
        iList.append(4*(i+ncomp))     # add vlsr error
        iList.append(4*(i+ncomp)+1)   # add sigma error

    # create an empty mask
    mmask = np.zeros(pmaps[0].shape, dtype='bool')
    for i in iList:
        mmask = np.logical_or(mmask, pmaps[i] > thresh)

    return mmask