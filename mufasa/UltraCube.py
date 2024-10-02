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
from copy import copy

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
        #self.rss_maps = {}
        #self.NSamp_maps = {}
        #self.AICc_maps = {}
        for nc in ncomps:
            if nc > 0 and hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                #self.pcubes[str(nc)]._modelcube = self.pcubes[str(nc)].get_modelcube(update=True, multicore=multicore)
                self.pcubes[str(nc)].get_modelcube(update=True, multicore=multicore)
                mod_mask = self.pcubes[str(nc)]._modelcube > 0
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


    def get_rss(self, ncomp, mask=None, planemask=None, update=True, expand=20):
        # residual sum of squares
        if mask is None:
            mask = self.master_model_mask
        # note: a mechanism is needed to make sure NSamp is consistient across the models
        rrs, nsamp = calc_rss(self, ncomp, usemask=True, mask=mask, return_size=True, update_cube=update,
                              planemask=planemask, expand=expand)

        # only include pixels with samples
        mask = nsamp < 1
        nsamp[mask] = np.nan
        # only if rss value is valid
        mask = np.logical_or(mask, rrs <= 0)
        rrs[mask] = np.nan
        if planemask is None:
            self.rss_maps[str(ncomp)] = rrs
            self.NSamp_maps[str(ncomp)] = nsamp
        else:
            self.rss_maps[str(ncomp)][planemask] = rrs
            self.NSamp_maps[str(ncomp)][planemask] = nsamp

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


    def get_AICc(self, ncomp, update=False, planemask=None, expand=20, **kwargs):
        # recalculate AICc fresh if update is True
        compID = str(ncomp)
        if update or not compID in self.AICc_maps:
            # start the calculation fresh
            # note that zero component is assumed to have no free-parameter (i.e., no fitting)
            p = ncomp * 4
            self.get_rss(ncomp, update=update, planemask=planemask, expand=expand, **kwargs)
            if planemask is None or not compID in self.AICc_maps:
                self.AICc_maps[compID] = aic.AICc(rss=self.rss_maps[compID], p=p, N=self.NSamp_maps[compID])
            else:
                self.AICc_maps[compID][planemask] = aic.AICc(rss=self.rss_maps[compID][planemask], p=p, N=self.NSamp_maps[compID][planemask])
        return self.AICc_maps[compID]


    def get_AICc_likelihood(self, ncomp1, ncomp2, **kwargs):
        return calc_AICc_likelihood(self, ncomp1, ncomp2, **kwargs)

    def get_all_lnk_maps(self, ncomp_max=2, rest_model_mask=True, multicore=True):
        return get_all_lnk_maps(self, ncomp_max=ncomp_max, rest_model_mask=rest_model_mask, multicore=multicore)

    def get_best_2c_parcube(self, multicore=True, lnk21_thres=5, lnk20_thres=5, lnk10_thres=5, return_lnks=True):
        kwargs = dict(multicore=multicore, lnk21_thres=lnk21_thres, lnk20_thres=lnk20_thres,
                      lnk10_thres=lnk10_thres, return_lnks=return_lnks)
        return get_best_2c_parcube(self, **kwargs)

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

def calc_rss(ucube, compID, usemask=True, mask=None, return_size=True, update_cube=False, planemask=None, expand=20):
    # calculate residual sum of squares

    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        #ucube.pcubes[compID]._modelcube = ucube.pcubes[compID].get_modelcube(update=update_cube, multicore=ucube.n_cores)
        #modcube = ucube.pcubes[compID]._modelcube
        modcube = ucube.pcubes[compID].get_modelcube(update=update_cube, multicore=ucube.n_cores)

    gc.collect()
    return get_rss(cube, modcube, expand=expand, usemask=usemask, mask=mask, return_size=return_size,
                   planemask=planemask)


def calc_chisq(ucube, compID, reduced=False, usemask=False, mask=None, expand=20):

    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(multicore=ucube.n_cores)

    gc.collect()
    return get_chisq(cube, modcube, expand=expand, reduced=reduced, usemask=usemask, mask=mask)


def calc_AICc(ucube, compID, mask, mask_plane=None, return_NSamp=True, expand=20):
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
        modcube = ucube.pcubes[compID].get_modelcube(update=True, multicore=ucube.n_cores)

    # get the rss value and sample size
    rss_map, NSamp_map = get_rss(ucube.cube, modcube, expand=expand, usemask=True, mask=mask, return_size=True, return_mask=False)
    # ensure AICc is only calculated where models exits
    #nmask = np.isnan(rss_map)
    #nmask = np.logical_or(NSamp_map == 0)
    #NSamp_map[nmask] = np.nan
    #rss_map[nmask] = np.nan
    #ucube.rss_maps[str(ncomp)] = rss_map
    #ucube.NSamp_maps[str(ncomp)] = NSamp_map
    AICc_map = aic.AICc(rss=rss_map, p=p, N=NSamp_map)

    if return_NSamp:
        return AICc_map, NSamp_map
    else:
        return AICc_map


def calc_AICc_likelihood(ucube, ncomp_A, ncomp_B, ucube_B=None, multicore=True, expand=0):
    # return the log likelihood of the A model relative to the B model
    # currently, expand is only used if ucube_B is provided

    if not ucube_B is None:
        # if a second UCube is provide for model comparison, use their common mask and calculate AICc values
        # without storing/updating them in the UCubes
        # reset model masks first
        ucube.reset_model_mask(ncomps=[ncomp_A], multicore=multicore)
        ucube_B.reset_model_mask(ncomps=[ncomp_B], multicore=multicore)

        mask = np.logical_or(ucube.master_model_mask, ucube_B.master_model_mask)
        AICc_A = calc_AICc(ucube, compID=ncomp_A, mask=mask, mask_plane=None, return_NSamp=False, expand=expand)
        AICc_B = calc_AICc(ucube_B, compID=ncomp_B, mask=mask, mask_plane=None, return_NSamp=False, expand=expand)
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
        ucube.reset_model_mask(ncomps=[ncomp_A, ncomp_B], multicore=multicore)
        ucube.get_AICc(ncomp_A, update=True)
        ucube.get_AICc(ncomp_B, update=True)

    gc.collect()
    lnk = aic.likelihood(ucube.AICc_maps[str(ncomp_A)], ucube.AICc_maps[str(ncomp_B)])
    # ensure the likihood map doesn't include where there are no samples
    # lnk[np.isnan(ucube.NSamp_maps[str(ncomp_A)])] = np.nan
    return lnk

def get_all_lnk_maps(ucube, ncomp_max=2, rest_model_mask=True, multicore=True):
    if rest_model_mask:
        ucube.reset_model_mask(ncomps=[2, 1], multicore=multicore)

    if ncomp_max <=1:
        lnk10 = ucube.get_AICc_likelihood(1, 0)
        return lnk10

    if ncomp_max <= 2:
        lnk21 = ucube.get_AICc_likelihood(2, 1)
        lnk20 = ucube.get_AICc_likelihood(2, 0)
        lnk10 = ucube.get_AICc_likelihood(1, 0)
        return lnk10, lnk20, lnk21

    else:
        pass

def get_best_2c_parcube(ucube, multicore=True, lnk21_thres=5, lnk20_thres=5, lnk10_thres=5, return_lnks=True, include_1c=True):
    # get the best 2c model justified by AICc lnk

    lnk10, lnk20, lnk21 = get_all_lnk_maps(ucube, ncomp_max=2, multicore=multicore)

    parcube = copy(ucube.pcubes['2'].parcube)
    errcube = copy(ucube.pcubes['2'].errcube)

    mask = np.logical_and(lnk21 > lnk21_thres, lnk20 > lnk20_thres)
    # logger.info("pixels better fitted by 2-comp: {}".format(np.sum(mask)))
    if include_1c:
        parcube[:4, ~mask] = copy(ucube.pcubes['1'].parcube[:4, ~mask])
        errcube[:4, ~mask] = copy(ucube.pcubes['1'].errcube[:4, ~mask])
        parcube[4:8, ~mask] = np.nan
        errcube[4:8, ~mask] = np.nan

    else:
        parcube[:, ~mask] = np.nan
        errcube[:, ~mask] = np.nan

    mask = lnk10 > lnk10_thres
    parcube[:, ~mask] = np.nan
    errcube[:, ~mask] = np.nan

    if return_lnks:
        return parcube, errcube, lnk10, lnk20, lnk21
    else:
        return parcube, errcube

#======================================================================================================================#
# statistics tools

def get_rss(cube, model, expand=20, usemask = True, mask = None, return_size=True, return_mask=False, include_nosamp=True, planemask=None):
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
            # may want to change this for future models that includes absorptions
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    if include_nosamp:
        # if there no mask in a given pixel, fill it in with combined spectral mask
        nsamp_map = np.nansum(mask, axis=0)
        mm = nsamp_map <= 0
        try:
            max_y, max_x = np.where(nsamp_map == np.nanmax(nsamp_map))
            spec_mask_fill = copy(mask[:,max_y[0], max_x[0]])
        except:
            spec_mask_fill = np.any(mask, axis=(1,2))
        mask[:, mm] = spec_mask_fill[:, np.newaxis]

    # assume flat-baseline model even if no model exists
    model[np.isnan(model)] = 0

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    if expand > 0:
        mask = expand_mask(mask, expand)
    mask = mask.astype(float)

    if planemask is None:
        residual = get_residual(cube, model)

        # note: using nan-sum may walk over some potential bad pixel cases
        rss = np.nansum((residual * mask)**2, axis=0)
        rss[rss == 0] = np.nan

    else:
        residual = get_residual(cube, model, planemask=planemask)
        mask_temp = mask
        mask = mask[:,planemask]

        # note: using nan-sum may walk over some potential bad pixel cases
        rss = np.nansum((residual * mask) ** 2, axis=0)
        rss[rss == 0] = np.nan

        '''
        rss_temp = np.zeros(planemask.shape)
        rss_temp[:] = np.nan
        rss_temp[planemask] = rss
        rss = rss_temp
        mask = mask_temp
        '''

    returns = (rss,)

    if return_size:
        nsamp = np.nansum(mask, axis=0)
        nsamp[np.isnan(rss)] = np.nan
        returns += (nsamp,)
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


def get_residual(cube, model, planemask=None):
    # calculate the residual of the fit to the cube
    if planemask is None:
        residual = cube.filled_data[:].value - model
    else:
        residual = cube.filled_data[:].value[:,planemask] - model[:,planemask]
    gc.collect()
    return residual


def get_Tpeak(model):
    return np.nanmax(model, axis=0)
