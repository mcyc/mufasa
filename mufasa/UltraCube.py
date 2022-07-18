from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

#======================================================================================================================#
import os
import numpy as np

from spectral_cube import SpectralCube
import pyspeckit
import multiprocessing
import gc
from astropy import units as u
import scipy.ndimage as nd

from . import aic
from . import multi_v_fit as mvf
from . import convolve_tools as cnvtool

#======================================================================================================================#

class UltraCube(object):

    def __init__(self, cubefile=None, cube=None, snr_min=None, rmsfile=None, cnv_factor=2):
        '''
        # a data frame work to handel multiple component fits and their results
        Parameters
        ----------
        filename : str, optional
            The name of 3D FITS file to load
        '''

        # to hold pyspeckit cubes for fitting
        self.pcubes = {}
        self.residual_cubes = {}
        self.rms_maps = {}
        self.Tpeak_maps = {}
        self.chisq_maps = {}
        self.rchisq_maps = {}
        self.rss_maps = {}
        self.NSamp_maps = {}
        self.AICc_maps = {}
        self.master_model_mask = None
        self.snr_min = None
        self.cnv_factor = cnv_factor
        self.n_cores = multiprocessing.cpu_count()

        if cubefile is not None:
            self.cubefile = cubefile
            self.load_cube(cubefile)
            return
        else:
            if hasattr(cube, 'spectral_axis'):
                # Load from a SpectralCube instance
                self.cube = cube

        if not snr_min is None:
            self.snr_min = kwargs['snr_min']


        if not rmsfile is None:
            self.rmsfile = kwargs['rmsfile']



    def load_cube(self, fitsfile):
        # loads SpectralCube
        self.cube = SpectralCube.read(fitsfile)


    def convolve_cube(self, savename=None, factor=None, edgetrim_width=5):
        # convolved the SpectralCube to a resolution X times the factor specified
        if factor is None:
            factor = self.cnv_factor
        self.cube_cnv = convolve_sky_byfactor(self.cube, factor, savename, edgetrim_width=edgetrim_width)


    def get_cnv_cube(self, filename=None):
        # load the convolve cube if the file exist, create one if otherwise
        if filename is None:
            self.convolve_cube(factor=self.cnv_factor)
        elif os.path.exists(filename):
            self.cube_cnv = SpectralCube.read(fitsfile)
        else:
            print("[WARNING]: the specified file does not exist.")


    def fit_cube(self, ncomp, **kwargs):
        # currently limited to NH3 (1,1) 2-slab fit

        if not 'multicore' in kwargs:
            kwargs['multicore'] = multiprocessing.cpu_count()

        if not 'snr_min' in kwargs:
            kwargs['snr_min'] = self.snr_min

        from collections.abc import Iterable
        if not isinstance(ncomp, Iterable):
            ncomp = [ncomp]

        for nc in ncomp:
            self.pcubes[str(nc)] = mvf.cubefit_gen(self.cube, ncomp=nc, **kwargs)

            if hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)].get_modelcube(multicore=self.n_cores) > 0
                self.include_model_mask(mod_mask)
            gc.collect()


    def include_model_mask(self, mask):
        # update the mask that shows were all the models are non-zero

        if self.master_model_mask is None:
            self.master_model_mask = mask
        else:
            self.master_model_mask = np.logical_or(self.master_model_mask, mask)

    def save_fit(self, savename, ncomp):
        # note, this implementation currently relies on
        if hasattr(self.pcubes[str(ncomp)], 'parcube'):
            save_fit(self.pcubes[str(ncomp)], savename, ncomp)
        else:
            print("[WARNING]: no fit was performed and thus no file will saved")


    def load_model_fit(self, filename, ncomp):
        self.pcubes[str(ncomp)] = load_model_fit(self.cube, filename, ncomp)
        # update model mask
        mod_mask = self.pcubes[str(ncomp)].get_modelcube(multicore=self.n_cores) > 0
        print("{}comp model mask size: {}".format(ncomp, np.sum(mod_mask)) )
        gc.collect()
        self.include_model_mask(mod_mask)


    def get_residual(self, ncomp):
        compID = str(ncomp)
        model = self.pcubes[compID].get_modelcube(multicore=self.n_cores)
        self.residual_cubes[compID] = get_residual(self.cube, model)
        gc.collect()
        return self.residual_cubes[compID]


    def get_rms(self, ncomp):
        compID = str(ncomp)
        if not compID in self.residual_cubes:
            self.get_residual(ncomp)

        self.rms_maps[compID] = get_rms(self.residual_cubes[compID])
        return self.rms_maps[compID]

    def get_rss(self, ncomp, mask=None):
        # residual sum of squares
        if mask is None:
            mask = self.master_model_mask
        # note: a mechanism is needed to make sure NSamp is consistient across the models
        self.rss_maps[str(ncomp)], self.NSamp_maps[str(ncomp)] = \
            calc_rss(self, ncomp, usemask=True, mask=mask, return_size=True)


    def get_Tpeak(self, ncomp):
        compID = str(ncomp)
        model = self.pcubes[compID].get_modelcube(multicore=self.n_cores)
        self.Tpeak_maps[compID] = get_Tpeak(model)
        return self.Tpeak_maps[compID]


    def get_chisq(self, ncomp, mask=None):
        if mask is None:
            mask = self.master_model_mask
        # note: a mechanism is needed to make sure NSamp is consistient across
        self.chisq_maps[str(ncomp)], self.NSamp_maps[str(ncomp)] = \
            calc_chisq(self, ncomp, reduced=False, usemask=True, mask=mask)


    def get_reduced_chisq(self, ncomp):
        # no mask is passed insnr_mask, and thus is not meant for model comparision
        compID = str(ncomp)
        self.rchisq_maps[compID]= \
            calc_chisq(self, ncomp, reduced=True, usemask=True, mask=None)
        return self.rchisq_maps[compID]


    def get_AICc(self, ncomp, update=True, **kwargs):

        compID = str(ncomp)
        if not compID in self.chisq_maps:
            self.get_rss(ncomp, **kwargs)

        # note that zero component is assumed to have no free-parameter (i.e., no fitting)
        p = ncomp*4

        AICc_map = aic.AICc(rss=self.rss_maps[compID], p=p, N=self.NSamp_maps[compID])

        if update:
            self.AICc_maps[compID] = AICc_map
        return AICc_map


    def get_AICc_likelihood(self, ncomp1, ncomp2):
        return calc_AICc_likelihood(self, ncomp1, ncomp2)


    def get_best_residual(self, cubetype=None):
        return None



class UCubePlus(UltraCube):
    # create a subclass of UltraCube that holds the directory information
    #__init__(self, cubefile=None, cube=None, snr_min=None, rmsfile=None, cnv_factor=2)

    def __init__(self, cubefile, cube=None, paraNameRoot=None, paraDir=None, **kwargs): # snr_min=None, rmsfile=None, cnv_factor=2):
        # super(UCube, self).__init__(cubefile=cubefile)
        UltraCube.__init__(self, cubefile, cube, **kwargs) # snr_min, rmsfile, cnv_factor)

        self.cubeDir = os.path.dirname(cubefile)

        if paraNameRoot is None:
            # use the cube file name as the basis
            self.paraNameRoot = "{}_paramaps".format(os.path.splitext(os.path.basename(cubefile))[0])
        else:
            self.paraNameRoot = paraNameRoot

        if paraDir is None:
            self.paraDir = "{}/para_maps".format(self.cubeDir)
        else:
            self.paraDir = paraDir

        if not os.path.exists(self.paraDir):
            os.makedirs(self.paraDir)

        self.paraPaths = {}



    def get_model_fit(self, ncomp, update=True, **kwargs):
        # load the model fits if it exist, else

        for nc in ncomp:
            if not str(nc) in self.paraPaths:
                self.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(self.paraDir, self.paraNameRoot, nc)

        if update:
            # re-fit the cube
            for nc in ncomp:
                #if update or (not os.path.isfile(self.paraPaths[str(nc)])):
                self.fit_cube(ncomp=[nc], **kwargs)
                gc.collect()
                self.save_fit(self.paraPaths[str(nc)], nc)
                gc.collect()

        for nc in ncomp:
            path = self.paraPaths[str(nc)]
            self.load_model_fit(path, nc)




#======================================================================================================================#

def fit_cube(cube, **kwargs):
    return mvf.cubefit_gen(cube, **kwargs)


def save_fit(pcube, savename, ncomp):
    # specifically save ammonia multi-component model with the right fits header
    mvf.save_pcube(pcube, savename, ncomp)


def load_model_fit(cube, filename, ncomp):
    # currently only loads ammonia multi-component model
    pcube = pyspeckit.Cube(cube=cube)

    # reigster fitter
    linename = 'oneone'
    from . import ammonia_multiv as ammv

    fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])
    pcube.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)

    pcube.load_model_fit(filename, npars=fitter.npars, fittype='nh3_multi_v')
    gc.collect()
    return pcube


def convolve_sky_byfactor(cube, factor, savename=None, **kwargs):
    return cnvtool.convolve_sky_byfactor(cube, factor, savename, **kwargs)


#======================================================================================================================#
# UltraCube based methods


def calc_rss(ucube, compID, usemask=True, mask=None, return_size=True):
    # calculate residual sum of squares

    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(multicore=ucube.n_cores)

    gc.collect()
    return get_rss(cube, modcube, expand=20, usemask=usemask, mask=mask, return_size=return_size)


def calc_chisq(ucube, compID, reduced=False, usemask=False, mask=None):

    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(multicore=ucube.n_cores)

    gc.collect()
    return get_chisq(cube, modcube, expand=20, reduced=reduced, usemask=usemask, mask=mask)


def calc_AICc_likelihood(ucube, ncomp_A, ncomp_B, ucube_B=None):
    # return the log likelihood of the A model relative to the B model

    if not ucube_B is None:
        # if a second UCube is provide for model comparison, use their common mask and calculate AICc values
        # without storing/updating them in the UCubes
        mask = np.logical_or(ucube.master_model_mask, ucube_B.master_model_mask)
        AICc_A = ucube.get_AICc(ncomp_A, update=False, mask=mask)
        AICc_B = ucube_B.get_AICc(ncomp_B, update=False, mask=mask)
        return aic.likelihood(AICc_A, AICc_B)

    if not str(ncomp_A) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_A)

    if not str(ncomp_B) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_B)

    NSampEqual = ucube.NSamp_maps[str(ncomp_A)] == ucube.NSamp_maps[str(ncomp_B)]

    if np.nansum(~NSampEqual) != 0:
        print("[WARNING]: Number of samples do not match. Recalculating AICc values")
        ucube.get_AICc(ncomp_A)
        ucube.get_AICc(ncomp_B)

    gc.collect()
    return aic.likelihood(ucube.AICc_maps[str(ncomp_A)], ucube.AICc_maps[str(ncomp_B)])


#======================================================================================================================#
# statistics tools

'''
def get_aic(chisq, p, N=None):
    # calculate AIC or AICc values
    if N is None:
        return aic.AIC(chisq, p)
    else:
        return aic.AICc(chisq, p, N)
'''


def get_rss(cube, model, expand=20, usemask = True, mask = None, return_size=True):
    '''
    Calculate residual sum of squares (RSS)

    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension

    reduced : boolean
        Whether or not to return the reduced chi-squared value or not

    mask: boolean array
        A mask stating which array elements the chi-squared values are calculated from
    '''

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    residual = get_residual(cube, model)

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    mask = expand_mask(mask, expand)
    mask = mask.astype(np.float)

    # note: using nan-sum may walk over some potential bad pixel cases
    rss = np.nansum((residual * mask)**2, axis=0)

    if return_size:
        return rss, np.nansum(mask, axis=0)
    else:
        return rss



def get_chisq(cube, model, expand=20, reduced = True, usemask = True, mask = None):
    '''
    cube : SpectralCube

    model: numpy array

    expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension

    reduced : boolean
        Whether or not to return the reduced chi-squared value or not

    usemask: boolean
        Whether or not to mask out some parts of the data.
        If no mask is provided, it masks out samples with model values of zero.

    mask: boolean array
        A mask stating which array elements the chi-squared values are calculated from
    '''

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    residual = get_residual(cube, model)

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    mask = expand_mask(mask, expand)
    mask = mask.astype(np.float)

    # note: using nan-sum may walk over some potential bad pixel cases
    chisq = np.nansum((residual * mask) ** 2, axis=0)

    if reduced:
        # assuming n_size >> n_parameters
        chisq /= np.nansum(mask, axis=0)

    rms = get_rms(residual)
    chisq /= rms ** 2

    gc.collect()

    if reduced:
        # return the reduce chi-squares values
        return chisq
    else:
        # return the ch-squared values and the number of data points used
        return chisq, np.nansum(mask, axis=0)



def get_masked_moment(cube, model, order=0, expand=10, mask=None):
    '''
    create masked moment using either the model
    :param cube: SpectralCube
    :param model: numpy array
    :param order:
    :param expand : int
        Expands the region where the residual is evaluated by this many channels in the spectral dimension
    :param boolean array
        A mask stating which array elements the chi-squared values are calculated from
    :return:
    '''

    if mask is None:
        mask = model > 0
    else:
        mask = np.logical_and(mask, np.isfinite(model))

    # get mask over where signal is stronger than the median
    peak_T = np.nanmax(model, axis=0)
    med_peak_T = np.nanmedian(peak_T)
    mask_highT_2d = peak_T > med_peak_T

    mask_lowT = mask.copy()
    mask_lowT[:, mask_highT_2d] = False

    # get all the spectral channels greater than 10% of the median peak
    specmask = model > med_peak_T*0.1
    specmask = np.any(specmask, axis=(1,2))

    # adopte those spectral channles for low signal regions
    mask_lowT[specmask, :] = True
    mask[:, ~mask_highT_2d] = mask_lowT[:, ~mask_highT_2d]

    # get pixels that aren't modeled
    #mask_s = np.zeros(mask.shape, dtype=np.bool)
    #mask_s[: ~np.all(mask, axis=0)] =

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    mask = expand_mask(mask, expand)
    mask = mask.astype(np.float)


    '''
    # expand in all directions instead
    #selem = np.ones(shape=(expand, expand, expand), dtype=np.bool)
    #mask = nd.binary_dilation(mask, selem)
    mask = nd.binary_dilation(mask, iterations=expand)

    # pixels with less than expand number of spectral chanels
    mask_s = np.zeros(mask.shape, dtype=np.bool)
    mask_s[:, np.sum(mask, axis=0) < expand] = True
    mask_s = expand_mask(mask_s, expand)

    mask = np.logical_or(mask, mask_s)
    '''

    maskcube = cube.with_mask(mask.astype(bool))
    maskcube = maskcube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    mom = maskcube.moment(order=order)
    return mom


def expand_mask(mask, expand):

    # adds a buffer of size set by the expand keyword to a 2D mask,
    selem = np.ones(expand,dtype=np.bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    return mask


def get_rms(residual):
    # get robust estimate of the rms from the fit residual
    diff = residual - np.roll(residual, 2, axis=0)
    #print("finite diff cube size: {}".format(np.sum(np.isfinite(diff))))
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5
    #print("finite rms map size: {}".format(np.sum(np.isfinite(rms))))
    gc.collect()
    return rms


def get_residual(cube, model):
    # calculate the residual of the fit to the cube
    residual = cube.filled_data[:].value - model
    gc.collect()
    return residual


def get_Tpeak(model):
    return np.nanmax(model, axis=0)
