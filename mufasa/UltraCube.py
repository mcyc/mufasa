from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

#======================================================================================================================#
import os
import warnings
import numpy as np

from spectral_cube import SpectralCube
# prevent any spectral-cube related warnings from being displayed.
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning, append=True)

import pyspeckit
import gc
from astropy import units as u
import scipy.ndimage as nd

from . import aic
from . import multi_v_fit as mvf
from . import convolve_tools as cnvtool
from .utils.multicore import validate_n_cores
#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

class UltraCube(object):

    def __init__(self, cubefile=None, cube=None, fittype=None, snr_min=None, rmsfile=None, cnv_factor=2, n_cores=True):
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
        self.snr_min = 0.0
        self.cnv_factor = cnv_factor
        self.n_cores = validate_n_cores(n_cores)
        self.fittype = fittype

        if cubefile is not None:
            self.cubefile = cubefile
            self.load_cube(cubefile)
            return
        else:
            if hasattr(cube, 'spectral_axis'):
                # Load from a SpectralCube instance
                self.cube = cube

        if not snr_min is None:
            self.snr_min = snr_min


        if not rmsfile is None:
            self.rmsfile = rmsfile


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
            self.cube_cnv = SpectralCube.read(filename)
        else:
            logger.warning("the specified convolved cube file does not exist.")


    def fit_cube(self, ncomp, simpfit=False, **kwargs):
        '''
        currently limited to NH3 (1,1) 2-slab fit

        kwargs are those used for pyspeckit.Cube.fiteach
        '''

        if not 'multicore' in kwargs:
            kwargs['multicore'] = self.n_cores 

        if not 'snr_min' in kwargs:
            kwargs['snr_min'] = self.snr_min
        try:
            from collections import Iterable
        except ImportError:
            # for backwards compatibility
            from collections.abc import Iterable
        if not isinstance(ncomp, Iterable):
            ncomp = [ncomp]

        for nc in ncomp:
            self.pcubes[str(nc)] = fit_cube(self.cube, fittype=self.fittype, simpfit=simpfit, ncomp=nc, **kwargs)

            if hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)].get_modelcube(multicore=kwargs['multicore']) > 0
                self.include_model_mask(mod_mask)
            gc.collect()

    def include_model_mask(self, mask):
        # update the mask that shows were all the models are non-zero

        if self.master_model_mask is None:
            self.master_model_mask = mask
        else:
            self.master_model_mask = np.logical_or(self.master_model_mask, mask)

    def reset_model_mask(self, ncomps, multicore=True):
        #reset and re-generate master_model_mask for all the components in ncomps
        self.master_model_mask = None
        for nc in ncomps:
            if hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)].get_modelcube(multicore=multicore) > 0
                self.include_model_mask(mod_mask)
            gc.collect()


    def save_fit(self, savename, ncomp):
        # note, this implementation currently relies on
        if hasattr(self.pcubes[str(ncomp)], 'parcube'):
            save_fit(self.pcubes[str(ncomp)], savename, ncomp)
        else:
            logger.warning("no fit was performed and thus no file will be saved")


    def load_model_fit(self, filename, ncomp, multicore=None):
        if multicore is None: multicore = self.n_cores
        self.pcubes[str(ncomp)] = load_model_fit(self.cube, filename, ncomp,self.fittype)
        # update model mask
        mod_mask = self.pcubes[str(ncomp)].get_modelcube(multicore=self.n_cores) > 0
        logger.debug("{}comp model mask size: {}".format(ncomp, np.sum(mod_mask)) )
        gc.collect()
        self.include_model_mask(mod_mask)


    def get_residual(self, ncomp, multicore=None):
        if multicore is None: multicore = self.n_cores
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
        # only include pixels with samples
        mask = self.NSamp_maps[str(ncomp)] == 0
        self.NSamp_maps[str(ncomp)][mask] = np.nan
        self.rss_maps[str(ncomp)][mask] = np.nan

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
        # recalculate AICc fresh if update is True
        if update or not compID in self.chisq_maps:
            # start the calculation fresh
            compID = str(ncomp)
            # note that zero component is assumed to have no free-parameter (i.e., no fitting)
            p = ncomp * 4
            self.get_rss(ncomp, **kwargs)
            self.AICc_maps[compID] = aic.AICc(rss=self.rss_maps[compID], p=p, N=self.NSamp_maps[compID])
        return self.AICc_maps[compID]


    def get_AICc_likelihood(self, ncomp1, ncomp2):
        return calc_AICc_likelihood(self, ncomp1, ncomp2)


    def get_best_residual(self, cubetype=None):
        return None


class UCubePlus(UltraCube):
    # create a subclass of UltraCube that holds the directory information

    def __init__(self, cubefile, cube=None, paraNameRoot=None, paraDir=None,fittype=None, **kwargs): # snr_min=None, rmsfile=None, cnv_factor=2):
        # super(UCube, self).__init__(cubefile=cubefile)

        UltraCube.__init__(self, cubefile, cube, fittype, **kwargs) # snr_min, rmsfile, cnv_factor)

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
        '''
        load the model fits if it exists, else

        kwargs are passed to pyspeckit.Cube.fiteach (if update)
        '''

        for nc in ncomp:
            if not str(nc) in self.paraPaths:
                self.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(self.paraDir, self.paraNameRoot, nc)

        if update:
            # re-fit the cube
            for nc in ncomp:
                if 'conv' in self.paraPaths[str(nc)]:
                    logger.info(f'Fitting convolved cube for {nc} component(s)')
                else:
                    logger.info(f'Fitting cube for {nc} component(s)')
                #if update or (not os.path.isfile(self.paraPaths[str(nc)])):
                if 'multicore' not in kwargs:
                    kwargs['multicore'] = self.n_cores
                self.fit_cube(ncomp=[nc], **kwargs)
                gc.collect()
                self.save_fit(self.paraPaths[str(nc)], nc)
                gc.collect()
        else:
            if 'conv' in self.paraPaths[str(nc)]:
                logger.info(f'Loading convolved cube fits for {nc} component(s)')
            else:
                logger.info(f'Loading fits for {nc} component(s)')

        for nc in ncomp:
            path = self.paraPaths[str(nc)]
            self.load_model_fit(path, nc)


#======================================================================================================================#

def fit_cube(cube, fittype, simpfit=False, **kwargs):
    '''
    kwargs are those used for pyspeckit.Cube.fiteach
    '''
    if simpfit:
        # fit the cube with the provided guesses and masks with no pre-processing
        return mvf.cubefit_simp(cube, fittype=fittype, **kwargs)
    else:
        return mvf.cubefit_gen(cube, fittype=fittype, **kwargs)


def save_fit(pcube, savename, ncomp):
    # specifically save ammonia multi-component model with the right fits header
    mvf.save_pcube(pcube, savename, ncomp)


def load_model_fit(cube, filename, ncomp, fittype):
    # currently only loads ammonia multi-component model
    pcube = pyspeckit.Cube(cube=cube)

    # reigster fitter
    if fittype == 'nh3_multi_v':
        linename = 'oneone'
        from .spec_models import ammonia_multiv as ammv
        fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])

    elif fittype == 'n2hp_multi_v':
        linename = 'onezero'
        from .spec_models import n2hp_multiv as n2hpmv
        fitter = n2hpmv.n2hp_multi_v_model_generator(n_comp=ncomp, linenames=[linename])

    pcube.specfit.Registry.add_fitter(fittype, fitter, fitter.npars)
    pcube.xarr.velocity_convention = 'radio'

    pcube.load_model_fit(filename, npars=fitter.npars,fittype=fittype)
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


def calc_AICc(ucube, compID, mask, mask_plane=None, return_NSamp=True):
    # calculate AICc value withouth change the internal ucube data

    if isinstance(compID, int):
        p = compID * 4
        compID = str(compID)
    elif isinstance(compID, str):
        p = int(compID) * 4

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(multicore=ucube.n_cores)

    # get the rss value and sample size
    rss_map, NSamp_map = get_rss(ucube.cube, modcube, expand=20, usemask=True, mask=None, return_size=True, return_mask=False)
    # ensure AICc is only calculated where models exits
    nmask = NSamp_map == 0
    NSamp_map[nmask] = np.nan
    rss_map[nmask] = np.nan
    AICc_map = aic.AICc(rss=rss_map, p=p, N=NSamp_map)

    if return_NSamp:
        return AICc_map, NSamp
    else:
        return AICc_map


def calc_AICc_likelihood(ucube, ncomp_A, ncomp_B, ucube_B=None, multicore=True):
    # return the log likelihood of the A model relative to the B model

    if not ucube_B is None:
        # if a second UCube is provide for model comparison, use their common mask and calculate AICc values
        # without storing/updating them in the UCubes
        mask = np.logical_or(ucube.master_model_mask, ucube_B.master_model_mask)
        AICc_A = calc_AICc(ucube, compID=ncomp_A, mask=mask, mask_plane=None, return_NSamp=False)
        AICc_B = calc_AICc(ucube_B, compID=ncomp_B, mask=mask, mask_plane=None, return_NSamp=False)
        return aic.likelihood(AICc_A, AICc_B)

    if not str(ncomp_A) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_A)

    if not str(ncomp_B) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_B)

    NSamp_mapA = ucube.NSamp_maps[str(ncomp_A)] # since (np.nan == np.nan) is False, this threw the below warning unnecessarily
    NSamp_mapB = ucube.NSamp_maps[str(ncomp_B)]

    if not np.array_equal(NSamp_mapA, NSamp_mapB, equal_nan=True):
        logger.warning("Number of samples do not match. Recalculating AICc values")
        #reset the master component mask first
        ucube.reset_model_mask(ncomps=[comp_A, ncomp_B], multicore=multicore)
        ucube.get_AICc(ncomp_A)
        ucube.get_AICc(ncomp_B)

    gc.collect()
    return aic.likelihood(ucube.AICc_maps[str(ncomp_A)], ucube.AICc_maps[str(ncomp_B)])


#======================================================================================================================#
# statistics tools

def get_rss(cube, model, expand=20, usemask = True, mask = None, return_size=True, return_mask=False):
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

    if expand > 0:
        mask = expand_mask(mask, expand)
    mask = mask.astype(float)

    # note: using nan-sum may walk over some potential bad pixel cases
    rss = np.nansum((residual * mask)**2, axis=0)

    returns = (rss,)

    if return_size:
        returns += (np.nansum(mask, axis=0),)
    if return_mask:
        returns += mask
    return returns


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

    if expand > 0:
        mask = expand_mask(mask, expand)
    mask = mask.astype(float)

    # note: using nan-sum may walk over some potential bad pixel cases
    chisq = np.nansum((residual * mask) ** 2, axis=0)

    if reduced:
        # assuming n_size >> n_parameters
        reduction = np.nansum(mask, axis=0) # avoid division by zero
        reduction[reduction == 0] = np.nan
        chisq /= reduction

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

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.

    if expand > 0:
        mask = expand_mask(mask, expand)
    mask = mask.astype(float)

    maskcube = cube.with_mask(mask.astype(bool))
    maskcube = maskcube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    mom = maskcube.moment(order=order)
    return mom


def expand_mask(mask, expand):

    # adds a buffer of size set by the expand keyword to a 2D mask,
    selem = np.ones(expand,dtype=bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    return mask


def get_rms(residual):
    # get robust estimate of the rms from the fit residual
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5
    gc.collect()
    return rms


def get_residual(cube, model):
    # calculate the residual of the fit to the cube
    residual = cube.filled_data[:].value - model
    gc.collect()
    return residual


def get_Tpeak(model):
    return np.nanmax(model, axis=0)
