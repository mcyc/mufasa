"""
The `mufasa.multi_v_fit` module provides functionality for preprocessing, masking, and refining
guesses for spectral line fitting, as well as utility functions for calculating
moments, signal-to-noise ratios, and handling parameter constraints.
"""

from __future__ import print_function
from __future__ import absolute_import

__author__ = 'mcychen'

import numpy as np
import pyspeckit
import astropy.io.fits as fits
import copy
import os, errno
import multiprocessing
import warnings
from datetime import datetime
import gc

from astropy import units as u
from astropy.units import UnitConversionError
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import mad_std

from scipy import ndimage as ndi

from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import Projection

from pyspeckit.spectrum.units import SpectroscopicAxis

from skimage.morphology import remove_small_objects, disk, opening, binary_erosion, remove_small_holes, binary_dilation
from dask.array.core import Array as daskArray
import dask.array

from . import signals
from . import moment_guess as momgue
from . import guess_refine as g_refine
from .exceptions import SNRMaskError, FitTypeError, StartFitError
from .utils.multicore import validate_n_cores
from .utils import interpolate, dask_utils
from . import __version__
from .spec_models.meta_model import MetaModel

# =======================================================================================================================

from .utils.mufasa_log import get_logger

logger = get_logger(__name__)


# =======================================================================================================================

def get_chisq(cube, model, expand=20, reduced=True, usemask=True, mask=None, rest_value=None):

    if rest_value is not None:
        cube = cube.with_spectral_unit(u.Hz, rest_value=rest_value)

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    residual = cube.filled_data[:].value - model

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand, dtype=np.bool)
    selem.shape += (1, 1,)
    mask = ndi.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask) ** 2, axis=0)

    if reduced:
        chisq /= np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    # (alternatively, we can use mad_std from astropy?)
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2 ** 0.5

    chisq /= rms ** 2

    if reduced:
        # return the reduce chi-squares values
        return chisq

    else:
        # return the ch-squared values and the number of data points used
        return chisq, np.sum(mask, axis=0)


def register_pcube(pcube, mod_info):

    # unit check and conversion
    if hasattr(pcube, "unit"):
        if pcube.unit is None or pcube.unit == '':
            logger.warning("The cube doesn't have a unit. The fitting will proceed assume it has a unit of K")
            pcube.unit = 'K'
        elif pcube.unit != 'K':
            raise ValueError(f"The cube should have units of K rather than {pcube.unit}")
    else:
        logger.warning("The cube doesn't have a unit. The fitting will proceed assume it has a unit of K")

    # spectral unit check and conversion
    if pcube.xarr.refX is None or np.isnan(pcube.wcs.wcs.restfrq):
        # Specify the reference rest frequency if not present
        logger.warning("The cube has no reference rest frequency."
                       " The rest frequency of the spectral model will be used instead")
        pcube.xarr.refX = mod_info.freq_dict[linename] * u.Hz

    if pcube.xarr.velocity_convention is None:
        pcube.xarr.velocity_convention = 'radio'

    # always register the fitter just in case different lines are used
    fitter = mod_info.fitter
    pcube.specfit.Registry.add_fitter(mod_info.fittype, fitter, fitter.npars)
    logger.debug("number of parameters is {0}".format(fitter.npars))
    logger.debug("the line to fit is {0}".format(mod_info.linenames))
    return pcube

# setup_cube is depreciated
'''
def setup_cube(cube, mod_info, return_spectral_cube=False):

    if hasattr(cube, 'spectral_axis'):
        # if it's a SpectralCube object
        pcube = pyspeckit.Cube(cube=cube)

        if isinstance(cube._data, dask.array.Array):
            logger.info("SpectralCube uses dask")
            # load the entire array onto pcube
            pcube.data = cube._data.compute()
            pcube.data = cube._data.compute()


        if False:# #isinstance(cube._data, np.ndarray):
            logger.info("SpectralCube does not use dask")
            pcube = pyspeckit.Cube(cube=cube)

        elif False: # isinstance(cube._data, dask.array.Array):
            logger.info("SpectralCube uses dask")

            if cube.spectral_axis.flags['OWNDATA']:
                xarr = SpectroscopicAxis(cube.spectral_axis,
                                         unit=cube.spectral_axis.unit,
                                         refX=cube.wcs.wcs.restfrq,
                                         refX_unit='Hz')
            else:
                xarr = SpectroscopicAxis(cube.spectral_axis.copy(),
                                         unit=cube.spectral_axis.unit,
                                         refX=cube.wcs.wcs.restfrq,
                                         refX_unit='Hz')

            pcube = pyspeckit.Cube(cube=cube._data.compute(), xarr=xarr, header=cube.header)

            if (cube.unit in ('undefined', u.dimensionless_unscaled)
                    and 'BUNIT' in cube._meta):
                pcube.unit = cube._meta['BUNIT']
            else:
                pcube.unit = cube.unit

    elif isinstance(cube, str):
        cubename = cube
        if return_spectral_cube:
            cube = SpectralCube.read(cube)
            # pcube = pyspeckit.Cube(filename=cubename)
            pcube = pyspeckit.Cube(cube=cube)

        else:
            pcube = pyspeckit.Cube(filename=cubename)

    pcube.unit = "K"
    if np.isnan(pcube.wcs.wcs.restfrq):
        # Specify the rest frequency not present
        pcube.xarr.refX = mod_info.freq_dict[linename] * u.Hz
    pcube.xarr.velocity_convention = 'radio'

    # always register the fitter just in case different lines are used
    fitter = mod_info.fitter
    pcube.specfit.Registry.add_fitter(mod_info.fittype, fitter, fitter.npars)
    logger.debug("number of parameters is {0}".format(fitter.npars))
    logger.debug("the line to fit is {0}".format(mod_info.linenames))

    if return_spectral_cube:
        # the following check on rest-frequency may not be necessarily for GAS, but better be safe than sorry
        # note: this assume the data cube has the right units
        if np.isnan(cube._wcs.wcs.restfrq):
            # Specify the rest frequency not present
            cube = cube.with_spectral_unit(u.Hz, rest_value=mod_info.rest_value)
        cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')

        return pcube, cube
    else:
        return pcube
'''

def cubefit_simp(cube, pcube, ncomp, guesses, multicore=None, maskmap=None, fittype='nh3_multi_v', snr_min=None, **kwargs):
    # a simper version of cubefit_gen that assumes good user provided guesses

    logger.debug("using cubefit_simp")

    # get information on the spectral model
    mod_info = MetaModel(fittype, ncomp)
    freq_dict = mod_info.freq_dict
    linename = mod_info.linenames

    Texmin = mod_info.Texmin
    Texmax = mod_info.Texmax
    sigmin = mod_info.sigmin
    sigmax = mod_info.sigmax
    taumax = mod_info.taumax
    taumin = mod_info.taumin
    eps = mod_info.eps

    # deprecated and no longer needed
    #pcube = setup_cube(cube, mod_info, return_spectral_cube=False)

    register_pcube(pcube, mod_info)

    multicore = validate_n_cores(multicore)

    # get the masking for the fit
    footprint_mask = np.any(np.isfinite(cube._data), axis=0)
    planemask = np.any(np.isfinite(guesses), axis=0)

    peaksnr, planemask, kwargs = handle_snr(pcube, snr_min, planemask, **kwargs)

    if maskmap is not None:
        maskmap = maskmap & planemask & footprint_mask
    else:
        maskmap = planemask & footprint_mask

    logger.debug(f"maskmap type: {type(maskmap)}")
    maskmap = maskmap.compute()

    v_peak_hwidth = 10
    v_guess = guesses[::4]
    v_guess[v_guess == 0] = np.nan

    v_median, v_99p, v_1p = get_vstats(v_guess)

    vmax = v_median + v_peak_hwidth
    vmin = v_median - v_peak_hwidth

    # use percentile limits padded with sigmax if these values are more relaxed than the v_peak window
    if v_99p + sigmax > vmax:
        vmax = v_99p + sigmax
    if v_1p - sigmax < vmin:
        vmin = v_1p - sigmax

    def impose_lim(data, min=None, max=None, eps=0):
        if min is not None:
            mask = data < min
            data[mask] = min + eps
        if max is not None:
            mask = data > max
            data[mask] = max - eps

    # impose parameter limits on the guesses
    impose_lim(guesses[::4], vmin, vmax, eps)
    impose_lim(guesses[1::4], sigmin, sigmax, eps)
    impose_lim(guesses[2::4], Texmin, Texmax, eps)
    impose_lim(guesses[3::4], taumin, taumax, eps)

    kwargs['start_from_point'] = get_start_point(maskmap, weight=None)

    logger.info(f"Number of cores used for cubefit_simp: {multicore}")

    # add all the pcube.fiteach kwargs)
    kwargs['multicore'] = multicore
    kwargs['maskmap'] = maskmap
    kwargs['use_neighbor_as_guess'] = False
    kwargs['limitedmax'] = [True, True, True, True] * ncomp
    kwargs['maxpars'] = [vmax, sigmax, Texmax, taumax] * ncomp
    kwargs['limitedmin'] = [True, True, True, True] * ncomp
    kwargs['minpars'] = [vmin, sigmin, Texmin, taumin] * ncomp
    kwargs = set_pyspeckit_verbosity(**kwargs)

    pcube.fiteach(fittype=fittype, guesses=guesses, **kwargs)

    return pcube


def cubefit_gen(cube, pcube, ncomp=2, paraname=None, modname=None, chisqname=None, guesses=None, errmap11name=None,
                multicore=None, mask_function=None, snr_min=0.0, momedgetrim=True, saveguess=False,
                fittype='nh3_multi_v', **kwargs):

    logger.debug('Using cubefit_gen')

    # get information on the spectral model
    mod_info = MetaModel(fittype, ncomp)
    freq_dict = mod_info.freq_dict
    linename = mod_info.linenames

    Texmin = mod_info.Texmin
    Texmax = mod_info.Texmax
    sigmin = mod_info.sigmin
    sigmax = mod_info.sigmax
    taumax = mod_info.taumax
    taumin = mod_info.taumin
    eps = mod_info.eps

    register_pcube(pcube, mod_info)

    if errmap11name is not None:
        errmap = fits.getdata(errmap11name)
    else:
        # a quick way to estimate RMS as long as the noise dominates the spectrum by channels
        errmap = None

    linewidth_sigma = False
    trim = 3

    # trim the edges by 3 pixels
    cube = signals.trim_cube_edge(cube, trim=trim)
    m0, m1, m2, rms = signals.get_moments(cube, window_hwidth=mod_info.window_hwidth,
                                          linewidth_sigma=linewidth_sigma, trim=None,
                                          central_win_hwidth=mod_info.central_win_hwidth,
                                          return_rms=True)
    # release memories
    if isinstance(cube._data, dask.array.Array):
        cube._data = dask_utils.persist_and_clean(cube._data)

    peaksnr = signals.get_snr(cube, rms=rms)

    # use min and max assuming the moment 1 is robust
    vmax = np.nanmax(m1).value + sigmax / 2
    vmin = np.nanmin(m1).value - sigmax / 2

    logger.debug("velocity limit: ({:.3f}, {:.3f})".format(vmin, vmax))

    if 'planemask' in kwargs:
        planemask = kwargs['planemask']
        planemask = planemask & np.any(cube.mask.include(), axis=0)
    else:
        planemask = np.any(cube.mask.include(), axis=0)

    if 'maskmap' in kwargs:
        logger.debug("including user specified mask as a base")
        planemask = np.logical_and(kwargs['maskmap'], planemask)

    if 'signal_cut' in kwargs:
        logger.warning("signal_cut will be set to zero for pcube.fiteach() in cubefit_gen."
                       " The snr_min is used to determine which pixels to fit prior to calling fiteach()")

    if snr_min > 0:
        snr_mask = peaksnr > snr_min
        if snr_min >= 3:
            mom_mask = np.isfinite(m1)
            planemask = planemask & snr_mask & mom_mask
        else:
            planemask = planemask & snr_mask

    if planemask.sum() < 1:
        msg = "The provided snr_min={} results in no valid pixels to fit; fitting terminated".format(snr_min)
        raise SNRMaskError(msg)

    if mask_function is not None:
        msg = "\'mask_function\' is now deprecation, and will be removed in the next version"
        warnings.warn(msg, DeprecationWarning)
        logger.warning(msg)

    # interpolate the moment maps
    mmm = interpolate.iter_expand(np.array([m0, m1, m2]), mask=planemask)
    m0, m1, m2 = mmm[0], mmm[1], mmm[2]
    gc.collect()

    logger.info("velocity fitting limits: ({}, {})".format(np.round(vmin, 2), np.round(vmax, 2)))

    # get the guesses based on moment maps
    # tex and tau guesses are chosen to reflect low density, diffusive gas that are likley to have low SNR
    gg = momgue.moment_guesses(m1, m2, ncomp, sigmin=sigmin, moment0=m0, linetype=mod_info.linetype,
                               mom0_floor=None)

    del m0, m1, m2
    gc.collect()

    if isinstance(cube._data, dask.array.Array):
        cube._data = dask_utils.persist_and_clean(cube._data)

    if guesses is None:
        guesses = gg

    else:
        # fill in the blanks with convolved interpolation then moment guesses
        guesses[guesses == 0] = np.nan
        gmask = np.isfinite(guesses)

        mom_mask = np.isfinite(gg)

        # get interpolated results
        kernel = Gaussian2DKernel(5)  # large kernel size because regions outside the guesses are likely noisy
        guesses_smooth = guesses.copy()
        for i, gsm in enumerate(guesses_smooth):
            guesses_smooth[i] = convolve(guesses[i], kernel, boundary='extend')

        guesses[~gmask] = guesses_smooth[~gmask]
        guesses[~mom_mask] = np.nan
        gmask = np.isfinite(guesses)

        # fill in the rest of the guesses with moment guesses
        guesses[~gmask] = gg[~gmask]

        # fill in the failed sigma guesses with interpotaed guesseses
        gmask = guesses[1::4] < sigmin
        guesses[1::4][gmask] = guesses_smooth[1::4][gmask]

        logger.debug("provided guesses accepted")

    # re-normalize the degenerated tau & text for the purpose of estimate guesses
    guesses[3::4], guesses[2::4] = g_refine.tautex_renorm(guesses[3::4], guesses[2::4], tau_thresh=taumin,
                                                          nu=mod_info.rest_value.to("GHz").value)

    # The guesses should be fine in the first case, but just in case, make sure the guesses are confined within the
    # appropriate limits
    guesses[::4][guesses[::4] > vmax] = vmax
    guesses[::4][guesses[::4] < vmin] = vmin
    guesses[1::4][guesses[1::4] > sigmax] = sigmax
    guesses[1::4][guesses[1::4] < sigmin] = sigmin + eps
    guesses[2::4][guesses[2::4] > Texmax] = Texmax
    guesses[2::4][guesses[2::4] < Texmin] = Texmin
    guesses[3::4][guesses[3::4] > taumax] = taumax
    guesses[3::4][guesses[3::4] < taumin] = taumin

    if saveguess:
        save_guesses(pcube, paraname, guesses, savename)

    kwargs['maskmap'] = planemask & np.any(cube.mask.include(), axis=0) & np.all(np.isfinite(guesses), axis=0)
    kwargs['maskmap'] = kwargs['maskmap'] & match_pcube_mask(pcube, kwargs['maskmap'])
    if isinstance(kwargs['maskmap'], daskArray):
        kwargs['maskmap'] = kwargs['maskmap'].compute()

    mask_size = np.sum(kwargs['maskmap'])

    if mask_size < 1:
        logger.warning("maskmap has no pixels, no fitting will be performed")
        return pcube
    elif np.all(np.isfinite(guesses), axis=0).sum() < 1:
        logger.warning("guesses has no pixels, no fitting will be performed")
        return pcube

    logger.info("Final mask size for the fitting: {0} pixels".format(mask_size))

    multicore = validate_n_cores(multicore)
    logger.info(f"Number of cores used for cubefit_gen: {multicore}")

    # get the start point
    if "start_from_point" in kwargs:
        logger.warning("user start point will not be used")
    start_point = get_start_point(kwargs['maskmap'], weight=peaksnr)
    kwargs['start_from_point'] = start_point

    # finalize the argruments to be passed into fiteach
    # signal_cut is set to zero, since a signal cut has already been done
    kwargs_core = dict(fittype=fittype,
                       guesses=guesses,
                       limitedmax=[True, True, True, True] * ncomp,
                       maxpars=[vmax, sigmax, Texmax, taumax] * ncomp,
                       limitedmin=[True, True, True, True] * ncomp,
                       minpars=[vmin, sigmin, Texmin, taumin] * ncomp,
                       multicore=multicore,
                       use_neighbor_as_guess=False,
                       guesses_are_moments=False,
                       signal_cut=0)
    kwargs = {**kwargs, **kwargs_core}
    kwargs = set_pyspeckit_verbosity(**kwargs)

    # start fitting the cube
    try:
        # fitting the cube
        idx = kwargs['start_from_point']
        logger.debug("guesses at start, 1st try: {}".format(guesses[:, idx[1], idx[0]]))
        logger.debug("peaksnr at start, 1st try: {}".format(peaksnr[idx[1], idx[0]]))
        pcube.fiteach(**kwargs)
    except ValueError as e:
        msg = "The starting fit position is somehow not among the valid, likely due to a bug. " \
              "ValueError message from pyspeckit: " + e.__str__()
        logger.warning(msg)
        retry_fit(pcube, kwargs, start_point, peaksnr=peaksnr)
    except AssertionError as e:
        msg = "The starting fit position is somehow not among the valid, likely due to a bug. " \
              "AssertionError message from pyspeckit: " + e.__str__()
        logger.warning(msg)
        retry_fit(pcube, kwargs, start_point, peaksnr=peaksnr)

    if paraname != None:
        save_pcube(pcube, paraname, ncomp=ncomp)

    if modname != None:
        model = SpectralCube(pcube.get_modelcube(), pcube.wcs, header=cube.header)
        model.write(modname, overwrite=True)

    if chisqname != None:
        chisq = get_chisq(cube, pcube.get_modelcube(), expand=20, rest_value=mod_info.rest_value)
        chisqfile = fits.PrimaryHDU(data=chisq, header=cube.wcs.celestial.to_header())
        chisqfile.writeto(chisqname, overwrite=True)

    return pcube


# ===============================================================================

def default_masking(snr, snr_min=5.0):
    # a snr masking fucntion that perform some quailty controls

    if snr_min is None:
        planemask = np.isfinite(snr)
    else:
        planemask = (snr > snr_min)

    if planemask.sum() > 100:
        # to create a less noisy mask further
        planemask = remove_small_holes(planemask, area_threshold=9)
        planemask = remove_small_objects(planemask, min_size=25)
        planemask = opening(planemask, disk(1))

    return (planemask)


def match_pcube_mask(pcube, maskmap):
    # a function to ensure the mask passed into pcube.fiteach will be valid
    # using the same code as pcube.fiteach to return a maskmap that should be valid when passed into fiteach
    if not hasattr(pcube.mapplot, 'plane'):
        logger.debug("pcube.mapplot has no attribute plane. Creating one right now.")
        pcube.mapplot.makeplane()
    if isinstance(pcube.mapplot.plane, np.ma.core.MaskedArray):
        OK = ((~pcube.mapplot.plane.mask) &
              maskmap.astype('bool')).astype('bool')
    else:
        if isinstance(pcube.mapplot.plane, daskArray):
            pcube.mapplot.plane = pcube.mapplot.plane.compute()
        OK = (np.isfinite(pcube.mapplot.plane) &
              maskmap.astype('bool')).astype('bool')
    return OK


def retry_fit(pcube, kwargs, old_start_point, peaksnr=None):
    # a module to retry a fit with a different start point
    # remove the last failed pixel from the mask before trying again

    kwargs['maskmap'][old_start_point[1], old_start_point[0]] = False
    kwargs['start_from_point'] = get_start_point(kwargs['maskmap'], weight=None)
    idx = kwargs['start_from_point']
    logger.debug("guesses at start, 2nd try: {}".format(kwargs['guesses'][:, idx[1], idx[0]]))
    if peaksnr is not None:
        logger.debug("peaksnr (not used) at start, 2st try: {}".format(peaksnr[idx[1], idx[0]]))

    try:
        pcube.fiteach(**kwargs)
    except ValueError as e:
        msg = "The first two attempts to fit a pixel did not yield a fit. Likely due to a lack of signal or poor guesses. " \
              "ValueError message from pyspeckit: " + e.__str__()
        logger.error(msg)
        raise StartFitError(msg)
    except AssertionError as e:
        msg = "The first two attempts to fit a pixel did not yield a fit. Likely due to a lack of signal or poor guesses. " \
              "AssertionError message from pyspeckit: " + e.__str__()
        logger.error(msg)
        raise StartFitError(msg)


def make_header(ndim, ref_header):

    # initilizing a new, empthy header
    hdr_new = fits.Header()

    hdr_new.set(keyword='DATE', value=datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), comment=f"Written by MUFASA v{__version__}")

    if ndim==3:
        keylist = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'OBJECT',
                   'BMAJ', 'BMIN', 'BPA', 'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRPIX3',
                   'CDELT1', 'CDELT2', 'CDELT3', 'CUNIT1', 'CUNIT2', 'CUNIT3',
                   'CTYPE1', 'CTYPE2', 'CTYPE3', 'CRVAL1', 'CRVAL2', 'CRVAL3',
                   'LONPOLE', 'LATPOLE', 'WCSNAME', 'MJDREF', 'RADESYS',
                   'EQUINOX', 'SPECSYS']
    elif ndim==2:
        keylist = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'OBJECT',
                   'BMAJ', 'BMIN', 'BPA', 'WCSAXES', 'CRPIX1', 'CRPIX2',
                   'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2',
                   'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                   'LONPOLE', 'LATPOLE', 'WCSNAME', 'MJDREF',
                   'RADESYS', 'EQUINOX', 'SPECSYS']

    for key in keylist:
        # copy cards from the reference header if they exist
        if key in ref_header:
            hdr_new.append(copy.copy(ref_header.cards[key]))

    # make sure the dimension of the header is written correctly
    comments_dict = dict(NAXIS='Number of array dimensions', WCSAXES='Number of coordinate axes')
    for key, comment in comments_dict.items():
        if (key in hdr_new) and (hdr_new[key] != ndim):
            hdr_new.set(keyword=key, value=ndim, comment=comment)

    hdr_new.add_history("====================================================================")
    hdr_new.add_history(f"Written by MUFASA v{__version__} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hdr_new.add_comment("====================================================================")
    hdr_new.add_comment("MUFASA citation: Chen et al. (2020), DOI: 10.3847/1538-4357/ab7378")
    hdr_new.add_comment("MUFASA is available on GitHub at: https://github.com/mcyc/mufasa")

    return hdr_new


def save_pcube(pcube, savename, ncomp, npara=4, header_note=None):
    # a method to save the fitted parameter cube with relavent header information

    # create a new header using the pcube's cube header as a template
    hdr_new = make_header(ndim=3, ref_header=pcube.header)

    if header_note is None:
        header_note = f'Parameter maps derived from {ncomp}-comp model fits'

    hdr_new.set(keyword='NOTES', value=header_note, comment=None, before='DATE')

    # modify the units of the plane accordingly
    hdr_new.set(keyword ='CDELT3', value=1, comment='[unitless] Coordinate increment at reference point')
    hdr_new.set(keyword ='CTYPE3', value='FITPAR', comment='fitted parameters')
    hdr_new.set(keyword ='CRVAL3', value=0, comment='[unitless] Coordinate value at reference point')
    hdr_new.set(keyword='CRPIX3', value=1, comment='[unitless] Pixel coordinate of reference point')
    hdr_new.set(keyword ='CUNIT3', value='None', comment='[unitless] Units of coordinate increment and value')

    if ncomp == 1:
        hdr_new.set(keyword ='PLANE0', value= 'VELOCITY', comment='[km/s] vlsr')
        hdr_new.set(keyword ='PLANE1', value= 'SIGMA', comment='[km/s] velocity dispersion')
        hdr_new.set(keyword ='PLANE2', value  = 'TEX', comment='[K] excitation temperature')
        hdr_new.set(keyword ='PLANE3', value  = 'TAU', comment='[unitless] peak optical depth')

        hdr_new.set(keyword ='PLANE4', value  = 'eVELOCITY', comment='[km/s] estimated error of vlsr')
        hdr_new.set(keyword ='PLANE5', value  = 'eSIGMA', comment='[km/s] estimated error velocity dispersion')
        hdr_new.set(keyword ='PLANE6', value = 'eTEX', comment='[K] estimated error excitation temperature')
        hdr_new.set(keyword ='PLANE7', value = 'eTAU', comment='[unitless] estimated error of peak optical depth')

    elif ncomp > 1:
        for i in range(0, ncomp):
            hdr_new.set(keyword=f'PLANE{i * npara + 0}', value=f'VELOCITY_{i + 1}',
                        comment=f'[km/s] vlsr of component {i+1}')

            hdr_new.set(keyword=f'PLANE{i * npara + 1}', value=f'SIGMA_{i + 1}',
                        comment=f'[km/s] velocity dispersion of component {i + 1}')

            hdr_new.set(keyword=f'PLANE{i * npara + 2}', value=f'TEX_{i + 1}',
                        comment=f'[K] excitation temperature of component {i + 1}')

            hdr_new.set(keyword=f'PLANE{i * npara + 3}', value=f'TAU_{i + 1}',
                        comment=f'[unitless] peak optical depth of component {i + 1}')

        # the loop is split into two so the numbers will be written in ascending order
        for i in range(0, ncomp):
            hdr_new.set(keyword=f'PLANE{(ncomp + i) * npara + 0}', value=f'eVELOCITY_{i + 1}',
                        comment=f'[km/s] estimated error of component {i+1} vlsr')

            hdr_new.set(keyword=f'PLANE{(ncomp + i) * npara + 1}', value=f'eSIGMA_{i + 1}',
                        comment=f'[km/s] estimated error of component {i + 1} velocity dispersion')

            hdr_new.set(keyword=f'PLANE{(ncomp + i) * npara + 2}', value=f'eTEX_{i + 1}',
                        comment=f'[K] estimated error of component {i + 1} excitation temperature')

            hdr_new.set(keyword=f'PLANE{(ncomp + i) * npara + 3}', value=f'eTAU_{i + 1}',
                        comment=f'[unitless] estimated error of component {i + 1} peak optical depth')

    fitcubefile = fits.PrimaryHDU(data=np.concatenate([pcube.parcube, pcube.errcube]), header=hdr_new)
    fitcubefile.writeto(savename, overwrite=True)
    logger.debug("{} saved.".format(savename))


def save_guesses(pcube, paraname, guesses, savename):
    # a module to save guesses for diagnostic purposes

    # create a new header using the pcube's cube header as a template
    hdr_new = make_header(ndim=3, ref_header=pcube.header)

    # modify the units of the plane accordingly
    hdr_new.set(keyword ='CDELT3', value=1, comment='[unitless] Coordinate increment at reference point')
    hdr_new.set(keyword ='CTYPE3', value='FITPAR', comment='fitted parameters')
    hdr_new.set(keyword ='CRVAL3', value=0, comment='[unitless] Coordinate value at reference point')
    hdr_new.set(keyword='CRPIX3', value=1, comment='[unitless] Pixel coordinate of reference point')
    hdr_new.set(keyword ='CUNIT3', value='None', comment='[unitless] Units of coordinate increment and value')

    savedir = "{0}/{1}".format(os.path.dirname(paraname), "guesses")

    try:
        os.makedirs(savedir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    savename = "{0}_guesses.fits".format(os.path.splitext(paraname)[0], "parameter_maps")
    savename = "{0}/{1}".format(savedir, os.path.basename(savename))

    fitcubefile = fits.PrimaryHDU(data=guesses, header=hdr_new)
    fitcubefile.writeto(savename, overwrite=True)


##################

def set_pyspeckit_verbosity(**kwargs):
    if 'verbose_level' not in kwargs:
        from .utils.mufasa_log import DEBUG, INFO
        if logger.getEffectiveLevel() < DEBUG:
            kwargs['verbose_level'] = 4
        elif logger.getEffectiveLevel() <= INFO:
            kwargs['verbose_level'] = 1
        else:
            kwargs['verbose_level'] = 0

    if (kwargs['verbose_level'] == 0) and ('verbose' not in kwargs):
        kwargs['verbose'] = False
        # pyspeckit defaults to verbose=True
        # if verbose=True and verbose_level=0 it prints output

    return kwargs


##################
def get_vstats(velocities, signal_mask=None):
    m1 = velocities
    mask = np.isfinite(m1)
    if signal_mask is not None:
        mask = np.logical_and(mask, signal_mask)
    m1_good = m1[mask]

    v_median = np.median(m1_good)
    v_99p = np.percentile(m1_good, 99)
    v_1p = np.percentile(m1_good, 1)
    return v_median, v_99p, v_1p


def get_start_point(maskmap, weight=None, return_xy=True):
    # if return_xy, return index in the order of xy (pcube.fiteach's convention) instead of yx
    # start_from_point is ordered in yx before it was returned upstream
    if weight is not None:
        if isinstance(weight, Projection):
            weight = weight.value
        # use the max position in weight within the mask as a start point
        weight = np.nan_to_num(weight * maskmap)
        try:
            start_from_point = np.unravel_index(weight.argmax(), weight.shape)  # get start_from_point in the yx order
            logger.debug("starting point obtained from using the weight: {}".format(start_from_point))
            logger.debug("value of maskmap at startpoint: {}".format(maskmap[start_from_point]))
            if not maskmap[start_from_point]:
                msg = "The start point is not within maskmap. This is a bug that needs to be addressed."
                logger.error(msg)
                raise IndexError(msg)
        except IndexError:
            logger.debug("Trouble getting good start point with weight. Trying again without the weight.")
            indx_g = np.argwhere(maskmap)
            start_from_point = (indx_g[0, 0], indx_g[0, 1])
    else:
        indx_g = np.argwhere(maskmap)
        start_from_point = (indx_g[0, 0], indx_g[0, 1])
    logger.debug("starting point: {}".format(start_from_point))
    if return_xy:
        return (start_from_point[1], start_from_point[0])
    else:
        return start_from_point


def snr_estimate(pcube, errmap, smooth=True):
    if hasattr(pcube, 'cube'):
        cube = pcube.cube
    else:
        # assume pcube is just a "regular" cube
        cube = pcube

    if errmap is None:
        # a quick way to estimate RMS as long as the noise dominates the spectrum by channels
        errmap = mad_std(cube, axis=0, ignore_nan=True)

    err_med = np.nanmedian(errmap)
    logger.debug("median rms: {:.5f}".format(err_med))

    # mask out pixels that are too noisy (in this case, 3 times the median rms in the cube)
    err_mask = errmap < err_med * 3.0

    snr = cube / errmap
    peaksnr = np.nanmax(snr, axis=0)

    if smooth:
        # the snr map can be noisy, so a little smoothing
        kernel = Gaussian2DKernel(1)
        peaksnr = convolve(peaksnr, kernel)

    return peaksnr, snr, errmap, err_mask


def handle_snr(pcube, snr_min, planemask, return_errmask=False, **kwargs):
    if snr_min is None:
        if 'signal_cut' not in kwargs:
            snr_min = 0.0
        else:
            snr_min = kwargs['signal_cut']
    elif 'signal_cut' in kwargs:
        if kwargs['signal_cut'] != snr_min:
            if kwargs['signal_cut'] > snr_min:
                snr_min = kwargs['signal_cut']
            else:
                kwargs['signal_cut'] = snr_min
            logger.warning("snr_min and signal_cut provided do not match."
                           " The higher value of the two, {}, is adopted".format(snr_min))
    else:
        kwargs['signal_cut'] = 0

    if 'errmap' not in kwargs:
        errmap = None

    peaksnr, snr, emap, err_mask = snr_estimate(pcube, errmap, smooth=True)

    if snr_min > 0:
        snr_mask = peaksnr > snr_min
        planemask = np.logical_and(planemask, snr_mask)

    if planemask.sum() < 1:
        msg = "The provided snr_min={} results in no valid pixels to fit; fitting terminated.".format(snr_min)
        raise SNRMaskError(msg)

    if return_errmask:
        return peaksnr, planemask, kwargs, err_mask
    else:
        return peaksnr, planemask, kwargs