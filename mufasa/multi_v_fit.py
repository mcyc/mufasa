from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

import numpy as np
import pyspeckit
import astropy.io.fits as fits
import copy
import os, errno
from os import path
import multiprocessing

from astropy import units as u
from astropy.stats import mad_std
from pyspeckit.spectrum.units import SpectroscopicAxis
from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict
from pyspeckit.spectrum.models import ammonia
import scipy.ndimage as nd

from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
from skimage.morphology import remove_small_objects,disk,opening,binary_erosion,remove_small_holes #, closing
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import mad_std

from . import ammonia_multiv as ammv
from . import moment_guess as momgue
#=======================================================================================================================
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#=======================================================================================================================
# the current implementation only fits the 1-1 lines
line_names = ["oneone"]

#=======================================================================================================================
# set global constants

# set the fit parameter limits (consistent with GAS DR1) for NH3 fits
Texmin = 3.0  # K; a more reasonable lower limit (5 K T_kin, 1e3 cm^-3 density, 1e13 cm^-2 column, 3km/s sigma)
Texmax = 40  # K; DR1 T_k for Orion A is < 35 K. T_k = 40 at 1e5 cm^-3, 1e15 cm^-2, and 0.1 km/s yields Tex = 37K
sigmin = 0.07  # km/s
sigmax = 2.5  # km/s; for Larson's law, a 10pc cloud has sigma = 2.6 km/s
#taumax = 100.0  # a reasonable upper limit for GAS data. At 10K and 1e5 cm^-3 & 3e15 cm^-2 -> 70
taumax = 30.0  # when the satellite hyperfine lines becomes optically thick
taumin = 0.1  # 0.2   # note: at 1e3 cm^-3, 1e13 cm^-2, 1 km/s linewidth, 40 K -> 0.15
eps = 0.001  # a small perturbation that can be used in guesses

#=======================================================================================================================

def get_multiV_models(paraname, refcubename, n_comp = 2, savename = None, snrname = None, rms = 0.15, rmspath = None,
                      linename = "oneone"):
    '''
    Creates a fits file containing the model cubes of individual components stacked into a hypercube
    :param paraname:
    :param refcubename:
    :param n_comp:
    :param savename:
    :param snrname:
    :param rms:
    :param rmspath:
    :return:
    '''

    para, hdr = fits.getdata(paraname, header = True)

    pcube = pyspeckit.Cube(refcubename)
    xarr = pcube.xarr

    cubes = [pcube.cube.copy()
             for i in np.arange(n_comp)]
    cubes = np.array(cubes)
    cubes[:] = np.nan

    # remove the error components
    n_para = n_comp*4
    para = para[:n_para]
    assert para.shape[0] == n_para

    yy,xx = np.indices(para.shape[1:])
    nanvals = np.any(~np.isfinite(para),axis=0)
    isvalid = np.any(para, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        models = [ammonia._ammonia_spectrum(xarr.as_unit('GHz'), tex=tex, tau_dict={linename:tau}, width=width, xoff_v=vel, fortho=0.0,
                                            line_names = [linename])
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        cubes[:,:,y,x] = models

    for xy in ProgressBar(list(valid_pixels)):
        print(int(xy[0]), int(xy[1]))
        model_a_pixel(xy)

    if savename != None:
        f_name, f_extension = path.splitext(savename)
        for i, data in enumerate(cubes):
            fname = "{0}_v{1}_{2}".format(f_name, i, f_extension)
            model = SpectralCube(data, pcube.wcs, header=pcube.header)
            model.write(fname, overwrite=True)

    if snrname != None:
        # calculate the peak temperature
        Tpeak = np.array([np.nanmax(cube, axis=0)
                 for cube in cubes])

        if rmspath is not None:
            rmsdata = fits.getdata(rmspath)
            if rmsdata.shape == Tpeak[0].shape:
                rms = rmsdata
            else:
                logger.warning("The shape of the rms map ({0}) does not match the shape of the emission map {1}." \
                      " An uniform rms value of: {2} has been adopted instead".format(rmsdata.shape, Tpeak[0].shape, rms))

        snr = Tpeak/rms
        snrfile = fits.PrimaryHDU(data=snr, header=pcube.header)

        for i in np.arange(n_comp*8)+1:
            key = 'PLANE{0}'.format(i)
            if key in hdr:
                hdr.remove(key)

        snrfile.header.set('CDELT3',1)
        snrfile.header.set('CTYPE3','FITPAR')
        snrfile.header.set('PLANE1','SNR_0')
        snrfile.header.set('PLANE2','SNR_1')
        snrfile.header.set('NAXIS3',n_comp*8)
        snrfile.writeto(snrname, overwrite=True)

    return cubes


def get_SNR(paraname, savename = None, rms = 0.15, n_comp = 2, linename='oneone'):
    '''
    Take a multiple velocity componet fit and produce a signal to noise ratio of the two velocity components
    :param paraname:
    :param savename:
    :param rms:
    :param n_comp:
    :return:
    '''

    para, hdr = fits.getdata(paraname, header = True)
    n_para = n_comp*4

    # remove the error components
    para = para[:n_para]
    assert para.shape[0] == n_para

    yy,xx = np.indices(para.shape[1:])
    nanvals = np.any(~np.isfinite(para),axis=0)
    isvalid = np.any(para, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    # Create a synthetic X-dimension in km/s
    vres = 0.07
    vpad = 0.5
    vmax, vmin = np.argmax([para[0],para[4]]), np.argmin([para[0],para[4]])
    vmax = vmax + vpad
    vmin = vmin - vpad
    n_samp = (vmax - vmin)/vres

    xarr = np.linspace(vmin, vmax, int(n_samp) + 1, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict[linename]*u.Hz).as_unit(u.GHz)

    peakT = np.zeros((n_comp, para.shape[1], para.shape[2]))

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        models = [ammonia._ammonia_spectrum(xarr.as_unit('GHz'), tex=tex, tau_dict={linename:tau}, width=width,
                                            xoff_v=vel, fortho=0.0,line_names = [linename])
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]
        peakT[:,y,x] = np.nanmax(models, axis = 1)

    for xy in ProgressBar(list(valid_pixels)):
        print(int(xy[0]), int(xy[1]))
        model_a_pixel(xy)

    if savename != None:

        for i in np.arange(n_comp*8)+1:
            key = 'PLANE{0}'.format(i)
            if key in hdr:
                hdr.remove(key)

        newfits = fits.PrimaryHDU(data=peakT/rms, header=hdr)
        newfits.header.set('CDELT3',1)
        newfits.header.set('CTYPE3','FITPAR')
        newfits.header.set('PLANE1','SNR_0')
        newfits.header.set('PLANE2','SNR_1')
        newfits.header.set('NAXIS3',n_comp*8)
        newfits.writeto(savename ,overwrite=True)

    return peakT/rms


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

    cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict['oneone']*u.Hz)

    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)


    residual = cube.filled_data[:].value-model

    # This calculates chisq over the region where the fit is non-zero
    # plus a buffer of size set by the expand keyword.

    selem = np.ones(expand,dtype=np.bool)
    selem.shape += (1,1,)
    mask = nd.binary_dilation(mask, selem)
    mask = mask.astype(np.float)
    chisq = np.sum((residual * mask)**2, axis=0)

    if reduced:
        chisq /= np.sum(mask, axis=0)

    # This produces a robust estimate of the RMS along every line of sight:
    # (alternatively, we can use mad_std from astropy?)
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5

    chisq /= rms**2

    if reduced:
        # return the reduce chi-squares values
        return chisq

    else:
        # return the ch-squared values and the number of data points used
        return chisq, np.sum(mask, axis=0)


def main_hf_moments(maskcube, window_hwidth, v_atpeak=None, signal_mask=None):
    '''
    # find moments for the main hyperfine lines
    # (moments, especially moment 2, computed with the satellite lines are less useful in terms of the kinematics)

    :param maskcube: Spectral Cube
        the Spectral Cube to make the moments from

    :param window_hwidth: float
        half-width of the window (in km/s) to be used to isolate the main hyperfine lines from the rest of the spectrum

    :param signal_mask: boolean ndarray
        the mask indicating which pixels to include in the initial search for the spectral signal in sum of all spectra

    -------
    :return: m0
    :return: m1
    :return: m2
    '''
    return momgue.window_moments(maskcube, window_hwidth, v_atpeak=v_atpeak, signal_mask=signal_mask)


def moment_guesses(moment1, moment2, ncomp, sigmin=0.07, tex_guess=3.2, tau_guess=0.5, moment0=None):
    '''
    Make reasonable guesses for the multiple component fits
    :param moment1:
    :param moment2:
    :param ncomp:
    :param sigmin:
        <float> default at 0.07 km/s, the spectral resolution of the GAS channels
    :param tex_guess:
    :param tau_guess:
    :return:
    '''
    return momgue.moment_guesses(moment1, moment2, ncomp, sigmin, tex_guess, tau_guess, moment0)


def make_guesses(sigv_para_name, n_comp = 2, tex_guess =10.0, tau_guess = 0.5):
    '''
    Make 2 velocity component fit guesses based on the GAS DR1 parameter maps
    Parameters
    ----------
    sigv_para_name : str
        The file name of the DR1 parameter map
    tex_guess: float
        The guess for excitation temperature for where full DR1 model wasn't fitted
    tau_guess: float
        The guess for the ammonia (1-1) optical depth for where full DR1 model wasn't fitted
    Returns
    -------
    guesses : ndarray
        parameter guesses for the 2 velocity component fit
    '''

    para = fits.getdata(sigv_para_name)
    vlsr = para[4]
    sigma = para[3]
    tex = para[1]

    # upper and lower limit of the Tex guesses
    Tmin, Tmax = 2.8, 12.0
    tex[tex < Tmin] = Tmin
    tex[tex > Tmax] = Tmax

    # lower limit of the sigma guess
    sigmin = 0.07

    # Guess linewidth (the current recipe works okay, but potential improvements can be made).
    gs_sig = 0.5*sigma
    gs_sig[gs_sig < sigmin] = sigmin + 0.001


    # get the 1-1 optical depth from the cold_ammonia model
    # (may be a little overkill for guesses, but I guess it's okay given that it's not too computationally intensive)
    tau11 = get_singv_tau11(para.copy())

    # construct guesses
    guesses = np.zeros((4*n_comp,)+para.shape[1:])

    if n_comp == 1:
        # provide guesses for where vlsr and sigma has been fitted in DR1 (has a larger footprint than the full 5 para fit)
        guesses[0,:,:] = vlsr           # v centriod
        guesses[1,:,:] = sigma          # linewidth
        guesses[2,:,:] = tex_guess      # T_ex
        guesses[3,:,:] = tau_guess      # tau

        # provide guesses for where tex and tau (and other parameters) has been fitted in DR1
        has_col = para[2] > 0
        guesses[2,has_col] = tex[has_col]               # T_ex
        guesses[3,has_col] = tau11[has_col]             # tau

    if n_comp == 2:
        # provide guesses for where vlsr and sigma has been fitted in DR1 (has a larger footprint than the full 5 para fit)
        guesses[0,:,:] = vlsr - 0.25*sigma      # v0 centriod
        guesses[1,:,:] = gs_sig                 # v0 width
        guesses[2,:,:] = tex_guess              # v0 T_ex
        guesses[3,:,:] = tau_guess*0.25         # v0 tau
        guesses[4,:,:] = vlsr + 0.25*sigma      # v1 centriod
        guesses[5,:,:] = gs_sig                 # v1 width
        guesses[6,:,:] = tex_guess              # v1 T_ex
        guesses[7,:,:] = tau_guess*0.75         # v1 tau

        # provide guesses for where tex and tau (and other parameters) has been fitted in DR1
        # note: the initial guess for the 1st component tau (i.e., the background component) is lower than the 2nd
        # component tau by design to give the fit a better chance of not falling into a local chi-squared minimum where
        # the second component is unphysically thick.
        has_col = para[2] > 0
        guesses[2,has_col] = tex[has_col]                    # v0 T_ex
        guesses[3,has_col] = tau11[has_col]*0.25             # v0 tau
        guesses[6,has_col] = tex[has_col]                    # v1 T_ex
        guesses[7,has_col] = tau11[has_col]*0.75             # v1 tau

    if n_comp > 2:
        raise Exception("guesses for > 2 components have not been implemented!")
        return None

    return guesses


def get_singv_tau11(singv_para):
    '''
    Take a GAS DR1 parameter maps and return optical depth of the 1-1 line.
    Parameters
    ----------
    sigv_para : str or ndarray
        The GAS DR1 parameter cube (i.e., maps), either as a file name or as a 3D ndarray cube.
    Returns
    -------
    tau11 : ndarray
        A map of model optical depths for ammonia (1-1)
    '''

    # Note: the efficiency could benifit from multi-core processing

    if type(singv_para) == str:
        parcube = fits.getdata(singv_para)

    else:
        parcube = singv_para

    # Create a dummy spectral-axis in km/s as a place holder to acquire tau
    xarr = np.linspace(0.0, 10.0, 10, endpoint = True)
    xarr = SpectroscopicAxis(xarr*u.km/u.s, velocity_convention='radio', refX=freq_dict['oneone']*u.Hz).as_unit(u.GHz)

    # set ntot elements with zero values to NaN
    parcube[:,parcube[2] == 0.0] = np.nan

    yy,xx = np.indices(parcube.shape[1:])
    nanvals = np.any(~np.isfinite(parcube),axis=0)
    isvalid = np.any(parcube, axis=0) & ~nanvals
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    tau11 = np.zeros(parcube.shape[1:])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        kwargs = {'tkin': parcube[0,y,x],
                  'tex': parcube[1,y,x],
                  'ntot': parcube[2,y,x],
                  'width': parcube[3,y,x],
                  'xoff_v': parcube[4,y,x],
                  'fortho': parcube[5,y,x],
                  'return_tau': True,
                  }
        tau = ammonia.cold_ammonia(xarr, **kwargs)
        tau11[y,x] = tau['oneone']

    for xy in ProgressBar(list(valid_pixels)):
        model_a_pixel(xy)

    return tau11


def cubefit_simp(cube, ncomp, guesses, multicore = None, maskmap=None, linename="oneone", **kwargs):
    # a simper version of cubefit_gen that assumes good user provided guesses

    logger.info("using cubefit_simp")

    if hasattr(cube, 'spectral_axis'):
        pcube = pyspeckit.Cube(cube=cube)

    else:
        cubename = cube
        pcube = pyspeckit.Cube(filename=cubename)

    pcube.unit="K"

    # the following check on rest-frequency may not be necessarily for GAS, but better be safe than sorry
    # note: this assume the data cube has the right units

    if pcube.wcs.wcs.restfrq == np.nan:
        # Specify the rest frequency not present
        pcube.xarr.refX = freq_dict[linename]*u.Hz
    pcube.xarr.velocity_convention = 'radio'

    # always register the fitter just in case different lines are used
    fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])
    pcube.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)

    if multicore is None:
        # use n-1 cores on the computer
        multicore= multiprocessing.cpu_count() - 1

        if multicore < 1:
            multicore = 1

    # get the masking for the fit
    footprint_mask = np.any(np.isfinite(cube._data), axis=0)
    planemask = np.any(np.isfinite(guesses), axis=0)

    if maskmap is not None:
        logger.info("using user specified mask")
        maskmap *= planemask * footprint_mask
    else:
        maskmap = planemask * footprint_mask

    if 'start_from_point' not in kwargs:
        logger.info("using automated starting point")
        indx_g = np.argwhere(maskmap)
        start_from_point = (indx_g[0,1], indx_g[0,0])
        #start_from_point = (indx_g[0, 0], indx_g[0, 1]) # probably the wrong order
        logger.info("starting point: {}".format(start_from_point))
        kwargs['start_from_point'] = start_from_point

    if 'signal_cut' not in kwargs:
        kwargs['signal_cut'] = 0.0

    if 'errmap' not in kwargs:
        kwargs['errmap'] = mad_std(pcube.cube, axis=0, ignore_nan = True)

    v_peak_hwidth = 10
    v_guess = guesses[::4]
    v_guess[v_guess == 0] = np.nan

    '''
    v_median = np.nanmedian(v_guess)
    v_guess_finite = v_guess[np.isfinite(v_guess)]
    v_99p = np.percentile(v_guess_finite, 99)
    v_1p = np.percentile(v_guess_finite, 1)
    '''

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

    # add all the pcube.fiteach kwargs)
    kwargs['multicore'] = multicore
    kwargs['use_neighbor_as_guess'] = False
    kwargs['limitedmax'] = [True, True, True, True] * ncomp
    kwargs['maxpars'] = [vmax, sigmax, Texmax, taumax] * ncomp
    kwargs['limitedmin'] = [True, True, True, True] * ncomp
    kwargs['minpars'] = [vmin, sigmin, Texmin, taumin] * ncomp

    pcube.fiteach(fittype='nh3_multi_v', guesses=guesses, maskmap=maskmap, **kwargs)

    '''
    try:
        pcube.fiteach(fittype='nh3_multi_v', guesses=guesses, maskmap=maskmap, **kwargs)

    except AssertionError:
        # if the start_from_point is invalid
        pcube.fiteach(fittype='nh3_multi_v', guesses=guesses, maskmap=maskmap, **kwargs)
        indx_g = np.argwhere(maskmap)
        start_from_point = (indx_g[0,1], indx_g[0,0])
        kwargs['start_from_point'] = start_from_point
    '''

    return pcube


def cubefit_gen(cube, ncomp=2, paraname = None, modname = None, chisqname = None, guesses = None, errmap11name = None,
            multicore = None, mask_function = None, snr_min=0.0, linename="oneone", momedgetrim=True, saveguess=False,
            **kwargs):
    '''
    Perform n velocity component fit on the GAS ammonia 1-1 data.
    (This should be the function to call for all future codes if it has been proven to be reliable)
    # note: the method can probably be renamed to cubefit()

    Parameters
    ----------
    cube : str
        The file name of the ammonia 1-1 cube or a SpectralCube object
    ncomp : int
        The number of components one wish to fit. Default is 2
    paraname: str
        The output file name of the
    Returns
    -------
    pcube : 'pyspeckit.cubes.SpectralCube.Cube'
        Pyspeckit cube object containing both the fit and the original data cube
    '''
    logger.info('Using cubefit_gen')

    if hasattr(cube, 'spectral_axis'):
        pcube = pyspeckit.Cube(cube=cube)

    else:
        cubename = cube
        cube = SpectralCube.read(cubename)
        pcube = pyspeckit.Cube(filename=cubename)

    pcube.unit="K"


    # the following check on rest-frequency may not be necessarily for GAS, but better be safe than sorry
    # note: this assume the data cube has the right units
    if cube._wcs.wcs.restfrq == np.nan:
        # Specify the rest frequency not present
        cube = cube.with_spectral_unit(u.Hz, rest_value = freq_dict[linename]*u.Hz)
    cube = cube.with_spectral_unit(u.km/u.s,velocity_convention='radio')

    if pcube.wcs.wcs.restfrq == np.nan:
        # Specify the rest frequency not present
        pcube.xarr.refX = freq_dict[linename]*u.Hz
    pcube.xarr.velocity_convention = 'radio'

    # always register the fitter just in case different lines are used
    fitter = ammv.nh3_multi_v_model_generator(n_comp = ncomp, linenames=[linename])
    pcube.specfit.Registry.add_fitter('nh3_multi_v', fitter, fitter.npars)
    logger.info("number of parameters is {0}".format(fitter.npars))
    logger.info("the line to fit is {0}".format(linename))

    # Specify a width for the expected velocity range in the data
    #v_peak_hwidth = 3.0 # km/s (should be sufficient for GAS Orion, but may not be enough for KEYSTONE)
    v_peak_hwidth = 4.0  # km/s (should be sufficient for GAS Orion, but may not be enough for KEYSTONE)

    if errmap11name is not None:
        errmap11 = fits.getdata(errmap11name)
    else:
        # a quick way to estimate RMS as long as the noise dominates the spectrum by channels
        errmap11 = mad_std(cube._data, ignore_nan=True, axis=0)

    err_med = np.nanmedian(errmap11)
    logger.info("median rms: {0}".format(err_med))

    # mask out pixels that are too noisy (in this case, 3 times the median rms in the cube)
    err_mask = errmap11 < err_med * 3.0

    snr = cube.filled_data[:].value / errmap11
    peaksnr = np.nanmax(snr, axis=0)

    # the snr map will inetiabley be noisy, so a little smoothing
    kernel = Gaussian2DKernel(1)
    peaksnr = convolve(peaksnr, kernel)

    footprint_mask = np.any(np.isfinite(cube._data), axis=0)


    if np.logical_and(footprint_mask.sum() > 1000, momedgetrim):
        # trim the edges by 3 pixels to guess the location of the peak emission
        logger.info("triming the edges to make moment maps")
        footprint_mask = binary_erosion(footprint_mask, disk(3))

    # the following function is copied directly from GAS
    def default_masking(snr, snr_min=5.0):
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

    if 'maskmap' in kwargs:
        planemask = kwargs['maskmap']
    elif mask_function is None:
        planemask = default_masking(peaksnr, snr_min=snr_min)
    else:
        planemask = mask_function(peaksnr, snr_min=snr_min)

    logger.info("planemask size: {0}, shape: {1}".format(planemask[planemask].sum(), planemask.shape))

    # masking for moment guesses (note that this isn't used for the actual fit)
    mask = np.isfinite(cube._data) * planemask * footprint_mask #* err_mask

    logger.info("mask size: {0}, shape: {1}".format(mask[mask].sum(), mask.shape))

    maskcube = cube.with_mask(mask.astype(bool))
    maskcube = maskcube.with_spectral_unit(u.km / u.s, velocity_convention='radio')

    if guesses is not None:
        v_guess = guesses[::4]
        v_guess[v_guess == 0] = np.nan
    else:
        v_guess = np.nan

    if np.isfinite(v_guess).sum() > 0:
        v_guess = v_guess[np.isfinite(v_guess)]
        v_median = np.median(v_guess)
        logger.info("The median of the user provided velocities is: {0}".format(v_median))
        m0, m1, m2 = main_hf_moments(maskcube, window_hwidth=v_peak_hwidth, v_atpeak=v_median)
        v_median, v_99p, v_1p = get_vstats(v_guess)
    else:
        signal_mask = default_masking(peaksnr, snr_min=10.0)
        sig_mask_size = signal_mask.sum()

        if sig_mask_size < 1:
            logger.info("No pixels with SNR > 10; try SNR>5")
            # if there's no pixel above SNR > 10, try lower the threshold to 5
            signal_mask = default_masking(peaksnr, snr_min=5.0)
            sig_mask_size = signal_mask.sum()

        if sig_mask_size < 1:
            logger.info("No pixels with SNR > 5; try SNR>3")
            # if there's no pixel above SNR > 10, try lower the threshold to 5
            signal_mask = default_masking(peaksnr, snr_min=3.0)
            sig_mask_size = signal_mask.sum()

        if sig_mask_size < 1:
            # if no pixel in the map still, use all pixels
            logger.info("No pixels with SNR > 3; using all pixels")
            signal_mask = planemask
            sig_mask_size = signal_mask.sum()

        logger.info("signal_mask size: {}".format(sig_mask_size))

        signal_mask *= err_mask


        from skimage.morphology import binary_dilation

        def adoptive_moment_maps(maskcube, seeds, window_hwidth, weights, signal_mask):
            # a method to divide the cube into different regions and moments in each region
            _, n_seeds = ndi.label(seeds)
            if n_seeds > 10:
                # if there are a large number of seeds, dilate the structure to merge the nearby structures into one
                seeds = binary_dilation(seeds, disk(5))

            return momgue.adoptive_moment_maps(maskcube, seeds, window_hwidth=window_hwidth,
                                              weights=weights, signal_mask=signal_mask)


        # find the number of structures in the signal_mask
        from scipy import ndimage as ndi

        _, n_sig_parts = ndi.label(signal_mask)

        if n_sig_parts > 1:
            # if there is more than one structure in the signal mask
            seeds = signal_mask
            m0, m1, m2 = adoptive_moment_maps(maskcube, seeds, window_hwidth=v_peak_hwidth,
                                 weights=peaksnr, signal_mask=signal_mask)

        else:
            # use err_mask if it has structures
            _, n_parts = ndi.label(~err_mask)
            if n_parts > 1:
                seeds = err_mask
                m0, m1, m2 = adoptive_moment_maps(maskcube, seeds, window_hwidth=v_peak_hwidth,
                                     weights=peaksnr, signal_mask=signal_mask)
            else:
                # use the simplest main_hf_moments
                m0, m1, m2 = main_hf_moments(maskcube, window_hwidth=v_peak_hwidth)

        # mask over robust moment guess pixels to set the velocity fitting range
        '''
        mask = np.isfinite(m1)
        mask = np.logical_and(mask, signal_mask)
        m1_good = m1[mask]

        v_median = np.median(m1_good)
        v_99p = np.percentile(m1_good, 99)
        v_1p = np.percentile(m1_good, 1)
        '''
        v_median, v_99p, v_1p = get_vstats(m1, signal_mask)

        if False:
            # save the moment maps for diagnostic purposes
            hdr_new = copy.deepcopy(pcube.header)
            hdr_new['CDELT3']= 1
            hdr_new['CTYPE3']= 'FITPAR'
            hdr_new['CRVAL3']= 0
            hdr_new['CRPIX3']= 1

            savename = "{0}_moments.fits".format(os.path.splitext(paraname)[0], "parameter_maps")
            fitcubefile = fits.PrimaryHDU(data=np.array([m0,m1,m2]), header=hdr_new)
            fitcubefile.writeto(savename ,overwrite=True)


    print("median velocity: {0}".format(v_median))

    # remove the nana values to allow np.nanargmax(m0) to operate smoothly
    m0[np.isnan(m0)] = 0.0 # I'm not sure if this is a good way to get around the sum vs nansum issue

    # define acceptable v range based on the provided or determined median velocity
    vmax = v_median + v_peak_hwidth
    vmin = v_median - v_peak_hwidth
    logger.info("median velocity: {0}".format(v_median))
    logger.info("velocity fitting limits: ({}, {})".format(np.round(vmin,2), np.round(vmax,2)))

    # use percentile limits padded with sigmax if these values are more relaxed than the v_peak window
    if v_99p + sigmax > vmax:
        vmax = v_99p + sigmax
    if v_1p - sigmax < vmin:
        vmin = v_1p - sigmax

    print("velocity fitting limits: ({}, {})".format(np.round(vmin,2), np.round(vmax,2)))

    # find the location of the peak signal (to determine the first pixel to fit if nearest neighbour method is used)
    peakloc = np.nanargmax(m0)
    ymax,xmax = np.unravel_index(peakloc, m0.shape)

    # get the guesses based on moment maps
    # tex and tau guesses are chosen to reflect low density, diffusive gas that are likley to have low SNR
    gg = moment_guesses(m1, m2, ncomp, sigmin=sigmin, moment0=m0)

    if guesses is None:
        guesses = gg

    else:
        # fill in the blanks with convolved interpolation then moment guesses
        guesses[guesses == 0] = np.nan
        gmask = np.isfinite(guesses)
        # gmask = binary_erosion(gmask)

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

        logger.info("provided guesses accepted")


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
        # save the guesses for diagnostic purposes
        hdr_new = copy.deepcopy(pcube.header)
        hdr_new['CDELT3']= 1
        hdr_new['CTYPE3']= 'FITPAR'
        hdr_new['CRVAL3']= 0
        hdr_new['CRPIX3']= 1

        savedir = "{0}/{1}".format(path.dirname(paraname), "guesses")

        try:
            os.makedirs(savedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        savename = "{0}_guesses.fits".format(path.splitext(paraname)[0], "parameter_maps")
        savename = "{0}/{1}".format(savedir,path.basename(savename))

        fitcubefile = fits.PrimaryHDU(data=guesses, header=hdr_new)
        fitcubefile.writeto(savename ,overwrite=True)

    # set some of the fiteach() inputs to that used in GAS DR1 reduction
    if not 'integral' in kwargs:
        kwargs['integral'] = False

    kwargs = set_pyspeckit_verbosity(**kwargs)

    if not 'signal_cut' in kwargs:
        kwargs['signal_cut'] = 2

    # Now fit the cube. (Note: the function inputs are consistent with GAS DR1 whenever possible)

    # use SNR masking if not provided
    if not 'maskmap' in kwargs:
        logger.info("mask not included in kwargs, generating mask")
        kwargs['maskmap'] = planemask * footprint_mask

    if np.sum(kwargs['maskmap']) < 1:
        logger.warning("maskmap has no pixels, no fitting will be performed")
        return pcube
    elif np.sum(np.isfinite(guesses)) < 1:
        logger.warning("guesses has no pixels, no fitting will be performed")
        return pcube

    if multicore is None:
        # use n-1 cores on the computer
        multicore = multiprocessing.cpu_count() - 1

        if multicore < 1:
            multicore = 1

    pcube.fiteach (fittype='nh3_multi_v', guesses=guesses,
                  start_from_point=(xmax,ymax),
                  use_neighbor_as_guess=False,
                  limitedmax=[True,True,True,True]*ncomp,
                  maxpars=[vmax, sigmax, Texmax, taumax]*ncomp,
                  limitedmin=[True,True,True,True]*ncomp,
                  minpars=[vmin, sigmin, Texmin, taumin]*ncomp,
                  multicore=multicore,
                  **kwargs
                  )

    if paraname != None:
        save_pcube(pcube, paraname, ncomp=ncomp)

    if modname != None:
        model = SpectralCube(pcube.get_modelcube(), pcube.wcs, header=cube.header)
        model.write(modname, overwrite=True)

    if chisqname != None:
        chisq = get_chisq(cube, pcube.get_modelcube(), expand=20)
        chisqfile = fits.PrimaryHDU(data=chisq, header=cube.wcs.celestial.to_header())
        chisqfile.writeto(chisqname, overwrite=True)

    return pcube


def save_pcube(pcube, savename, ncomp=2):
    # a method to save the fitted parameter cube with relavent header information

    npara = 4

    hdr_new = copy.deepcopy(pcube.header)
    for i in range (0, ncomp):
        hdr_new['PLANE{0}'.format(i*npara+0)] = 'VELOCITY_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+1)] = 'SIGMA_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+2)] = 'TEX_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+3)] = 'TAU_{0}'.format(i+1)

    # the loop is split into two so the numbers will be written in ascending order
    for i in range (0, ncomp):
        hdr_new['PLANE{0}'.format((ncomp+i)*npara +0)] = 'eVELOCITY_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format((ncomp+i)*npara +1)] = 'eSIGMA_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format((ncomp+i)*npara +2)] = 'eTEX_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format((ncomp+i)*npara +3)] = 'eTAU_{0}'.format(i+1)
    hdr_new['CDELT3']= 1
    hdr_new['CTYPE3']= 'FITPAR'
    hdr_new['CRVAL3']= 0
    hdr_new['CRPIX3']= 1

    fitcubefile = fits.PrimaryHDU(data=np.concatenate([pcube.parcube,pcube.errcube]), header=hdr_new)
    fitcubefile.writeto(savename ,overwrite=True)
    logger.info("{} saved!".format(savename))

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

