from __future__ import print_function
import numpy as np
import copy

from astropy import units as u
from spectral_cube import SpectralCube

import pyspeckit
from pyspeckit.spectrum.models.ammonia_constants import (ckms, h, kb)
from astropy.stats import mad_std

from .utils import map_divide
import multiprocessing
from .utils.multicore import validate_n_cores
#=======================================================================================================================
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#=======================================================================================================================


class LineSetup(object):
    def __init__(self, linetype='nh3', new_recipe=True):

        if linetype is 'nh3':
            from pyspeckit.spectrum.models.ammonia_constants import voff_lines_dict

            voff = voff_lines_dict['oneone']

            # define max and min values of tex and tau to use for the test
            # a spectrum with tex and tau values both below the specified minima has an intensity below the expected GAS rms

            if new_recipe:
                self.tex_max = 8.0
                self.tau_max = 5.0
                self.tex_min = 3.0
                self.tau_min = 0.1
            else:
                self.tex_max = 8.0
                self.tau_max = 1.0
                self.tex_min = 3.1
                self.tau_min = 0.3

        elif linetype is 'n2hp':
            '''
            For Julian to fill in for the N2H+ model
            '''
            from .spec_models.n2hp_constants import voff_lines_dict

            voff = voff_lines_dict['onezero']

            # define max and min values of tex and tau to use for the test
            if new_recipe:
                self.tex_max = 8.0
                self.tau_max = 5.0
                self.tex_min = 3.0
                self.tau_min = 0.1
            else:
                self.tex_max = 8.0
                self.tau_max = 1.0
                self.tex_min = 3.1
                self.tau_min = 0.3
        else:
            raise Exception("{} is an invalid linetype".format(linetype))

#=======================================================================================================================

def master_guess(spectrum, ncomp, sigmin=0.07, v_peak_hwidth=3.0, v_atpeak=None, widewVSep=False, snr_cut=3,
                 signal_mask=None, linetype='nh3'):

    m0, m1, m2 = window_moments(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_atpeak, signal_mask=signal_mask)

    # estimate the rms level, and pass to the spectrum (probably should've been performed in another function)
    rms = get_rms_prefit(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=m1)

    if m0 < snr_cut*rms:
        gg = np.zeros((ncomp * 4,) + np.array([m1], dtype=object).shape)
        gg[:] = np.nan
        return gg

    if ncomp == 2 and widewVSep:
        # use recipe that recovers two-slab spectra (warning, may not be ideal if more than 3 slabs are present)

        m0_b, m1_b, m2_b = noisemask_moment(spectrum, m1, m2, mask_sigma=4, noise_rms=rms, window_hwidth=v_peak_hwidth)

        if m0_b > snr_cut*rms:
            # if the residual spectrum has m0 that is 3 sigma above the rms noise, treat both moment as individual
            # one component parts
            gg_a = moment_guesses(np.array([m1]), np.array([m2]), ncomp=1, sigmin=sigmin, moment0=np.array([m0]),
                                  linetype=linetype)
            gg_b = moment_guesses(np.array([m1_b]), np.array([m2_b]), ncomp=1, sigmin=sigmin, moment0=np.array([m0_b]),
                                  linetype=linetype)
            gg = np.zeros((ncomp * 4,) + np.array([m1]).shape)
            gg[:4,:] = gg_a[:]
            gg[4:,:] = gg_b[:]

        else:
            gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]), linetype=linetype)

    else:
        # get the guesses based on moment maps based on "traditional" recipe
        gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]), linetype=linetype)

    return gg


# THIS FUNCTION IS NEVER  CALLED
def get_window_slab(maskcube, window_hwidth=3.0, v_atpeak=None):
    if v_atpeak is None:
        # find the peak of the integrated spectrum if v_atpeak isn't provided
        tot_spec = np.nansum(maskcube._data[:,]*maskcube.get_mask_array(), axis=(1,2))
        idx_peak = np.nanargmax(tot_spec)
        print("peak T_B: {0}".format(np.nanmax(tot_spec)))
        v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value
        print("v_atpeak: {0}".format(v_atpeak))

    vmax = v_atpeak + window_hwidth
    vmin = v_atpeak - window_hwidth

    # Extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = maskcube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
    return slab


def vmask_moments(cube, vmap, window_hwidth=3.0):
    # obtain moments with windows centered around the vlsr specified in the provided map
    cubemasked = vmask_cube(cube, vmap, window_hwidth)

    m0 = cubemasked.moment0(axis=0).value
    m1 = cubemasked.moment1(axis=0).to(u.km/u.s).value
    m2 = (np.abs(cubemasked.moment2(axis=0))**0.5).to(u.km/u.s).value

    return m0, m1, m2


def vmask_cube(cube, vmap, window_hwidth=3.0):
    spax = cube.spectral_axis.value
    spax_cube = np.ones(cube.shape) * spax[:, None, None]
    v_up = vmap + window_hwidth
    v_down = vmap - window_hwidth
    mask = np.logical_and(spax_cube > v_down, spax_cube < v_up)
    cubemasked = cube.with_mask(mask)
    return cubemasked


def window_moments(spec, window_hwidth=4.0, v_atpeak=None, signal_mask=None):
    '''
    :param spec:
        A spectrum or cube. Can be either pyspeckit objects or spectral_cube objects
    :param window_hwidth:
        <float>
        the half-width of the spectral window to calculate moments from (useful to isolate hyperfine lines)
    :param v_atpeak:
        <float or ndarraay>
        the velocity or velocity map to center the moment calculation on
    :param signal_mask:
    :return:
    '''

    def moments_pys_spectrum(spectrum, window_hwidth=4.0, v_atpeak=None, iter_refine=False):
        '''
        find moments within a given window (e.g., around the main hyperfine lines)
        # note: iter_refine has not proven to be very effective in our tests

        :param spectrum:
            <pyspeckit.spectrum.classes.Spectrum>
            the spectrum to take the momentw of

        :param window_hwidth: float
            half-width of the window (in km/s) to be used to isolate the main hyperfine lines from the rest of the spectrum

        '''

        if v_atpeak is None:
            moments = spectrum.moments(unit=u.km/u.s)
            v_atpeak = moments[2]

        vmax = v_atpeak + window_hwidth
        vmin = v_atpeak - window_hwidth

        # Extract the spectrum within the window defined around the main hyperfine components and take moments
        slice = spectrum.slice(vmin, vmax, unit=u.km/u.s)
        moments = slice.moments(unit=u.km/u.s)

        if iter_refine:
            # for low snr- this method really doesn't work well
            m0, m1, m2 = moments[1], moments[2], moments[3]
            # make the window smaller by making out channels outside a specific width around moment 1
            # create a window 2 times the second moment
            new_window_hw = m2*3.0
            if new_window_hw > window_hwidth:
                new_window_hw = window_hwidth
            vmax = m1 + new_window_hw
            vmin = m1 - new_window_hw
            slice = spectrum.slice(vmin, vmax, unit=u.km / u.s)
            moments = slice.moments(unit=u.km / u.s)

        return moments[1], moments[2], moments[3]


    def moments_pys_cube(pcube, window_hwidth=4.0, v_atpeak=None, iter_refine=False, multicore=None):
        '''
        find moments within a given window (e.g., around the main hyperfine lines)
        # note: iter_refine has not proven to be very effective in our tests

        :param pcube:
            <pyspeckit.cubes.SpectralCube.Cube>
            the cube to take the moments of

        :param window_hwidth: float
            half-width of the window (in km/s) to be used to isolate the main hyperfine lines from the rest of the spectrum

        :param v_atpeak: float or ndarray
            the velociy or a map of velocities to center the spectral window on
        '''

        multicore = validate_n_cores(multicore)

        def get_win_moms(pcube, v_atpeak):
            # get window moments when v_atpeak is given
            pcube_masked = window_mask_pcube(pcube, v_atpeak, win_hwidth=window_hwidth)
            try:
                pcube_masked.momenteach(unit='km/s', verbose=False, multicore=multicore)
            except AttributeError:
                # if multicore fails, only use one core
                pcube_masked.momenteach(unit='km/s', verbose=False, multicore=1)
            moments = pcube_masked.momentcube
            return moments

        # note: the current implementation may be memory intensitive
        if v_atpeak is None:
            # note the moment 1 estimate of pcube.momenteach seem to be pretty good at ignoring hyperfine structures
            try:
                pcube.momenteach(unit='km/s', vheight=False, verbose=False, multicore=multicore)
            except AttributeError:
                # if multicore fails, only use one core
                pcube.momenteach(unit='km/s', vheight=False, verbose=False, multicore=1)

            moments = pcube.momentcube
            # redo again with the hyperfine lines masked out
            moments = get_win_moms(pcube, v_atpeak=moments[1])
        else:
            moments = get_win_moms(pcube, v_atpeak)

        if iter_refine:
            # use the moment 1 estimated velocity to define new windows
            # for low snr this method may not work well
            moments = get_win_moms(pcube, v_atpeak=moments[1])

        return moments[0], moments[1], moments[2]


    def moments_spectralcube(maskcube, window_hwidth, v_atpeak=None, signal_mask=None):
        # signal_mask is to provide additional masking specifically and only for v_atpeak estimate
        if v_atpeak is None:
            # find the peak of the integrated spectrum if v_atpeak isn't provided
            mask = maskcube.get_mask_array()
            if signal_mask is not None:
                mask = mask*signal_mask
            tot_spec = np.nansum(maskcube._data[:,]*mask, axis=(1,2))
            idx_peak = np.nanargmax(tot_spec)
            logger.debug("Getting window moments of SpectralCube")
            logger.debug("peak T_B: {0}".format(np.nanmax(tot_spec)))
            v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value
            logger.debug("v_atpeak: {0}".format(v_atpeak))

        vmax = v_atpeak + window_hwidth
        vmin = v_atpeak - window_hwidth

        # Extract the spectrum within the window defined around the main hyperfine components and take moments
        slab = maskcube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
        m0 = slab.moment0(axis=0).value
        m1 = slab.moment1(axis=0).to(u.km/u.s).value
        m2 = (np.abs(slab.moment2(axis=0))**0.5).to(u.km/u.s).value

        return m0, m1, m2
    
    # wrapper to find moments for different types of inputs

    if isinstance(spec, pyspeckit.Cube):
        # this method is much slower than using SpectralCube, but also seems more robust at spectral peaks
        return moments_pys_cube(spec, window_hwidth, v_atpeak)

    elif isinstance(spec, pyspeckit.spectrum.classes.Spectrum):
        return moments_pys_spectrum(spec, window_hwidth, v_atpeak)

    elif isinstance(spec, SpectralCube):
        # currently cannot handle v_atpeak as a map
        if hasattr(v_atpeak, 'ndim') and v_atpeak.ndim>0: # numpy floats have ndim=0
            logger.error("the method that handles SpectralCube cannot currently handle v_atpeak as a map, please use single value v_atpeak instead")
        return moments_spectralcube(spec, window_hwidth, v_atpeak, signal_mask)

    else:
        raise Exception("the spec provided is of invalid type")
        return None


def noisemask_moment(sp, m1, m2, mask_sigma=4, noise_rms = None, **kwargs):
    # mask out the 'main' component based on moment map and replace them with fake noise
    # and rerun window_mements to find additional components

    sp_m = copy.copy(sp)

    if 'v_atpeak' not in kwargs:
        kwargs['v_atpeak'] = m1

    if noise_rms is None:
        noise_rms = get_rms_prefit(sp, **kwargs)

    mask = np.logical_and(sp_m.xarr.value < m1 + mask_sigma * m2, sp_m.xarr.value > m1 - mask_sigma * m2)
    sp_m.data[mask] = np.random.randn(np.sum(mask)) * noise_rms

    return window_moments(sp_m, **kwargs)


def moment_guesses(moment1, moment2, ncomp, sigmin=0.07, tex_guess=3.2, tau_guess=0.5, moment0=None, linetype='nh3', mom0_floor=None):
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

    # setup different limits based on the line specie
    line_setup = LineSetup(linetype)
    tex_max = line_setup.tex_max
    tau_max = line_setup.tau_max
    tex_min = line_setup.tex_min
    tau_min = line_setup.tau_min

    if moment0 is not None:
        #print "[WARNING]: moment0 map is provided, thus the user-provided tex and tau will not be used"
        # normalize the moment 0 map with respect to the norm_ref percentile value
        # e.g., 95 percentile value being normalized to have a value of 0.95
        norm_ref = 99.73
        mom0high = np.percentile(moment0[np.isfinite(moment0)], norm_ref)
        # may want to modify this normalization to be something a little simpler or physical (i.e., 99.73/100 ~ 1)
        if mom0_floor is None:
            m0Norm = moment0.copy()*norm_ref/100.0/mom0high
            tex_guess = m0Norm*tex_max
            tau_guess = m0Norm*tau_max
        else:
            m0Norm = moment0.copy() * norm_ref / 100.0 / (mom0high - mom0_floor)
            tex_guess = m0Norm * (tex_max - tex_min) + tex_min
            tau_guess = m0Norm * (tau_max - tau_min) + tau_min

    m1 = moment1
    m2 = moment2

    # Guess linewidth (the current recipe works okay, but potential improvements can be made.
    gs_sig = m2/ncomp
    gs_sig[gs_sig < sigmin] = sigmin
    # note 0.08 k is narrow enough to be purely thermal @ ~10 K

    # there are 4 parameters for each v-component
    gg = np.zeros((ncomp*4,)+m1.shape)

    if ncomp == 1:
        gg[0,:] = m1                 # v0 centriod
        gg[1,:] = gs_sig             # v0 width
        gg[2,:] = tex_guess          # v0 T_ex
        gg[3,:] = tau_guess          # v0 tau

    # using a working recipe (assuming a bright and a faint componet)
    if ncomp == 2:
        sigmaoff = 0.4
        tau2_frac = 0.25                    # the tau weight of the second component relative to the total fraction
        gg[0,:] = m1 - sigmaoff*m2         # v0 centriod
        gg[1,:] = gs_sig                   # v0 width
        gg[2,:] = tex_guess                # v0 T_ex
        gg[3,:] = tau_guess*(1-tau2_frac)  # v0 tau
        gg[4,:] = m1 + sigmaoff*m2         # v1 centriod
        gg[5,:] = gs_sig                   # v1 width
        gg[6,:] = tex_guess*0.8            # v1 T_ex
        gg[7,:] = tau_guess*tau2_frac      # v1 tau

    # using a generalized receipe that have not been tested (lots of room for improvement!)
    if ncomp > 2:
        for i in range (0, ncomp):
            gg[i,  :] = m1+(-1.0+i*1.0/ncomp)*0.5*m2 # v0 centriod (step through a range fo velocities within sigma_v)
            gg[i+1,:] = gs_sig                   # v0 width
            gg[i+2,:] = tex_guess*0.8            # v0 T_ex
            gg[i+3,:] = tau_guess/ncomp*0.25     # v0 tau


    # ensure the tex and tau guesses falls within the guessing limits
    tex_guess[tex_guess < tex_min] = tex_min
    tex_guess[tex_guess > tex_max] = tex_max
    tau_guess[tau_guess < tau_min] = tau_min
    tau_guess[tau_guess > tau_max] = tau_max

    return gg

#=======================================================================================================================
# additional recipe

def moment_guesses_1c(m0, m1, m2):
    # make guesses based on the moment maps
    # note: it does not impose limits on parameters such as tau and tex

    # m0 from pyspeckit is actually amplitube proxy, so to calculate
    gs_sig = m2 ** 0.5
    mom0 = m0 * gs_sig * np.sqrt(2 * np.pi)

    mom0_thres = 3.0 # the regime boundary between fix-tau- or fix-tex-based calculation
    mom0_min = 0.03 # Ta ~ 0.25 at Tex=3, tau=0.1
    tau_fx = 2.5 #1.5 #2.5
    tex_fx = 6.0 #7.0 #6.0 # K

    # divide the tau tex guess into two regimes by integrated flux
    # if flux is greater than mom0_thres, assume a fixed tau value and caculate the corrosponding tex assume Gaussian
    # otherwise, assume a fixed tex value and calculate tau instead
    if isinstance(m0, float):
        # for 0D
        if mom0 < mom0_min:
            mom0 = mom0_min

        if mom0 > mom0_thres:
            tau_guess = tau_fx
            tex_guess = get_tex(mom0, tau_fx)

        else:
            tex_guess = tex_fx
            tau_guess = get_tau(mom0, tex_guess)

    elif m0.ndim == 2:
        # aassume all mom0 lower than mom0 to be mom0_min
        mom0[mom0 < mom0_min] = mom0_min

        # for 2D
        tau_guess = np.zeros(m0.shape)
        tex_guess = np.zeros(m0.shape)
        mask = mom0 > mom0_thres

        tau_guess[mask] = tau_fx
        tex_guess[mask] = get_tex(mom0[mask], tau_fx)

        tau_guess[~mask] = get_tau(mom0[~mask], tex_fx)
        tex_guess[~mask] = tex_fx


    else:
        raise Exception("the moment 0 input has the wrong dimension ({})".format(m0.ndim))
        return None

    return np.asarray([m1, gs_sig, tex_guess, tau_guess])


def mom_guess_wide_sep(spec, vpeak=None, rms=None, planemask=None, multicore=None):
    # for two components

    win_hwidth = 4.0
    # the window for the second component recovery (though the 1st win_hwidth should mask out the hyperfines already)
    win_hwidth2 = 10.0

    # the amount of moment 2 estimated sigma to maskout for "residual moment maps"
    f_sig_mask = 2.0

    rms_thres = 2.0  # the amplitude threshold to count as the signal

    f_tau = 0.5 # weight of the tau for the "equal-weight" guesses
    f_sig = 0.5 # weight of the linewidth for all guesses

    multicore = validate_n_cores(multicore)

    if isinstance(spec, pyspeckit.spectrum.classes.Spectrum):
        spec.xarr.velocity_convention = 'radio'
        spec.xarr = spec.xarr.as_unit('km/s')

        if rms is None:
            # we only need a crude estimate
            rms = mad_std(spec.data, axis=None, ignore_nan=True)

        if vpeak is None:
            # find the v_peak pased on smoothed spectrum
            sp_smooth = spec.copy()
            sp_smooth.smooth(3)
            idx = np.argmax(sp_smooth.data)
            vpeak = sp_smooth.xarr[idx]
            vpeak = vpeak.value
            logger.info("vpeak for widely separated moment guesses: {}".format(vpeak))

        # get the moments
        m0, m1, m2 = window_moments(spec, window_hwidth=win_hwidth, v_atpeak=vpeak)

        # assume the emission is dominated by the brighter component, and use moment maps to
        # create guesses for the first component
        gg1 = moment_guesses_1c(m0, m1, m2)

        # convert moment 2 to sigma
        sig = m2 ** 0.5

        # mask emission found by moment masks, and try to find a secondary peak using a broader window
        sp4 = spec.copy()
        mask = np.logical_and(sp4.xarr.value > m1 - sig * f_sig_mask, sp4.xarr.value < m1 + sig * f_sig_mask)
        #mask2 = np.logical_and(sp4.xarr.value > vpeak - win_hwidth, sp4.xarr.value < vpeak + win_hwidth)
        mask2 = np.logical_and(sp4.xarr.value > m1 - win_hwidth, sp4.xarr.value < m1 + win_hwidth)
        mask = np.logical_or(mask, ~mask2)

        sp4.data[mask] = 0.0
        #sp4.data[~mask2] = 0.0

        m0n, m1n, m2n = window_moments(sp4, window_hwidth=win_hwidth2, v_atpeak=vpeak)

        gg2 = moment_guesses_1c(m0n, m1n, m2n)
        gg = np.concatenate((gg1, gg2))

        if m0n < rms * rms_thres:
            gg[4:8] = gg1[:]
            gg[0] += sig
            gg[4] -= sig
            # set tau of each component to half of what the 1-component moment guess is
            gg[3] *= f_tau
            gg[7] *= f_tau

        gg[1] *= f_sig
        gg[5] *= f_sig


    elif isinstance(spec, SpectralCube):

        if rms is None:
            # we only need a crude estimate
            rms = mad_std(spec._data, axis=0, ignore_nan=True)

        spec = spec.with_spectral_unit("km/s", velocity_convention="radio")
        pcube = pyspeckit.Cube(cube=spec, maskmap=planemask, velocity_convention = 'radio')

        # get the moments using the pyspeckit method
        m0, m1, m2 = window_moments(pcube, window_hwidth=win_hwidth, v_atpeak=vpeak)

        # assume the emission is dominated by the brighter component, and use moment maps to
        # create guesses for the first component
        gg1 = moment_guesses_1c(m0, m1, m2)

        # convert moment 2 to sigma
        sig = m2 ** 0.5

        # maskout emission found by moment masks, and try to find a secondary peak
        spax = spec.spectral_axis.value
        spax_cube = np.broadcast_to(spax[:, np.newaxis, np.newaxis], (len(spax), m1.shape[0], m1.shape[1]))
        smask = np.logical_and(spax_cube > m1 - sig * f_sig_mask, spax_cube < m1 + sig * f_sig_mask)
        smask2 = np.logical_and(spax_cube > m1 - win_hwidth, spax_cube < m1 + win_hwidth)
        smask = np.logical_or(smask, ~smask2)

        maskcube = spec.with_mask(~smask)
        # fill value needs to be zero for moment estimates to work as expected
        maskcube = maskcube.with_fill_value(0.0)

        # convert it to pyspeckit to take advantage of pyspeckit's moment methods
        pcube = pyspeckit.Cube(cube=maskcube, maskmap=planemask)
        pcube.momenteach(verbose=False, multicore=multicore)
        m0n, m1n, m2n = pcube.momentcube

        gg2 = moment_guesses_1c(m0n, m1n, m2n)

        # use "equal weight guesses" if the remaining spectrum is too faint
        mask = m0n < rms * rms_thres

        gg2[:,mask] = gg1[:,mask].copy()
        gg = np.concatenate((gg1, gg2), axis=0)

        #gg[4:8, mask] = gg1[:, mask]
        gg[0, mask] += sig[mask]
        gg[4, mask] -= sig[mask]
        # set tau of each component to half of what the 1-component moment guess is
        gg[3,mask] *= f_tau
        gg[7,mask] *= f_tau
        # set the guessing linewidth to be half of the moment guesses, for both components

        gg[1,:] *= f_sig
        gg[5,:] *= f_sig


    return gg


#=======================================================================================================================
# utility functions

def get_rms_prefit(spectrum, window_hwidth, v_atpeak, linetype='nh3'):

    s = spectrum
    line_setup = LineSetup(linetype)

    vsys = v_atpeak*u.km/u.s
    throw = window_hwidth*u.km/u.s
    voff = line_setup.voff

    mask = np.ones(s.shape[0], dtype=np.bool)

    for deltav in voff:
        mask *= (np.abs(s.xarr - (deltav * u.km / u.s + vsys)) > throw)

    d_rms = s.data.copy()

    return mad_std(d_rms[mask])


def adaptive_moment_maps(maskcube, seeds, window_hwidth, weights=None, signal_mask=None):
    # split the cube up into different regions and make mosaic moment maps from moments of each individual regions

    labels, n_labs = map_divide.dist_divide(seeds, weights=weights, return_nmarkers=True)

    if signal_mask is None:
        signal_mask = seeds

    m0 = np.zeros(labels.shape)
    m0[:] = np.nan
    m1 = m0.copy()
    m2 = m0.copy()

    moments = [m0, m1, m2]

    # make moment map for each region
    for i in range(n_labs):
        mask = labels == i + 1

        new_cube_mask = maskcube.get_mask_array()#maskcube.mask.view()
        new_cube_mask = new_cube_mask * mask

        moms = window_moments(maskcube.with_mask(new_cube_mask), window_hwidth, signal_mask=signal_mask*mask)

        for m, m_p in zip(moments, moms):
            m[mask] = m_p[mask]

    return m0, m1, m2


def window_mask_pcube(pcube, vmid, win_hwidth=4.0):
    # returns a copy of the pucbe (pyspeckit.cubes.SpectralCube.Cube) with spectra outside of the window zeroed
    shape = pcube.cube.shape
    spax = pcube.xarr.value
    spax_cube = np.broadcast_to(spax[:, np.newaxis, np.newaxis], (len(spax), shape[1], shape[2]))

    smask = np.logical_and(spax_cube > vmid - win_hwidth, spax_cube < vmid + win_hwidth)

    pcube = pcube.copy()
    #pcube.cube[~smask] = 0.0
    pcube.cube[~smask] = np.nan
    pcube.maskmap = np.any(np.isfinite(pcube.cube), axis=0)

    return pcube

#=======================================================================================================================
# physics functions

def get_tex(Ta, tau=0.5, nu=23.722634):
    # calculate the excitation temperature given tau

    background_tb = 2.7315
    T0 = (h * nu * 1e9 / kb)

    term1 = Ta / T0 / (1 - np.exp(-tau))
    term2 = 1.0 / (np.exp(T0 / background_tb) - 1)
    term3 = 1.0 / (term1 + term2) + 1
    term4 = T0 / np.log(term3)
    return term4


def get_tau(Ta, tex=6.0, nu=23.722634):
    background_tb = 2.7315
    T0 = (h * nu * 1e9 / kb)

    term1 = 1 / (np.exp(T0 / tex) - 1) - 1 / (np.exp(T0 / background_tb) - 1)
    term2 = -Ta / T0 / term1 + 1
    return -np.log(term2)


def peakT(tex, tau, nu=23.722634):
    background_tb = 2.7315
    tauprof = tau
    T0 = (h * nu * 1e9 / kb)
    return (T0 / (np.exp(T0 / tex) - 1) - T0 / (np.exp(T0 / background_tb) - 1)) * (1 - np.exp(-tauprof))
