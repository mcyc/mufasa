import numpy as np
import copy

from astropy import units as u
from spectral_cube import SpectralCube

import pyspeckit
from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict
from astropy.stats import mad_std

#=======================================================================================================================

# define max and min values of tex and tau to use for the test
# a spectrum with tex and tau values both below the specified minima has an intensity below the expected GAS rms
tex_max = 8.0
tau_max = 1.0
tex_min = 3.1
tau_min = 0.3

#=======================================================================================================================

def master_guess(spectrum, ncomp, sigmin=0.07, v_peak_hwidth=3.0, v_atpeak=None, widewVSep=False, snr_cut=3):

    m0, m1, m2 = window_moments(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=v_atpeak)

    # estimate the rms level, and pass to the spectrum (probably should've been performed in another function)
    rms = get_rms_prefit(spectrum, window_hwidth=v_peak_hwidth, v_atpeak=m1)

    if m0 < snr_cut*rms:
        gg = np.zeros((ncomp * 4,) + np.array([m1]).shape)
        gg[:] = np.nan
        return gg

    if ncomp == 2 and widewVSep:
        # use recipe that recovers two-slab spectra (warning, may not be ideal if more than 3 slabs are present)

        m0_b, m1_b, m2_b = noisemask_moment(spectrum, m1, m2, mask_sigma=4, noise_rms=rms, window_hwidth=v_peak_hwidth)

        if m0_b > snr_cut*rms:
            # if the residual spectrum has m0 that is 3 sigma above the rms noise, treat both moment as individual
            # one component parts
            gg_a = moment_guesses(np.array([m1]), np.array([m2]), ncomp=1, sigmin=sigmin, moment0=np.array([m0]))
            gg_b = moment_guesses(np.array([m1_b]), np.array([m2_b]), ncomp=1, sigmin=sigmin, moment0=np.array([m0_b]))
            gg = np.zeros((ncomp * 4,) + np.array([m1]).shape)
            gg[:4,:] = gg_a[:]
            gg[4:,:] = gg_b[:]

        else:
            gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]))

    else:
        # get the guesses based on moment maps based on "traditional" recipe
        gg = moment_guesses(np.array([m1]), np.array([m2]), ncomp, sigmin=sigmin, moment0=np.array([m0]))

    return gg


def get_window_slab(maskcube, window_hwidth=3.0, v_atpeak=None):
    if v_atpeak is None:
        # find the peak of the integrated spectrum if v_atpeak isn't provided
        tot_spec = np.nansum(maskcube._data[:,]*maskcube.get_mask_array(), axis=(1,2))
        idx_peak = np.nanargmax(tot_spec)
        print "peak T_B: {0}".format(np.nanmax(tot_spec))
        v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value
        print "v_atpeak: {0}".format(v_atpeak)

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



def window_moments(spec, window_hwidth=3.0, v_atpeak=None):
    # wrapper
    if isinstance(spec, pyspeckit.spectrum.classes.Spectrum):
        return window_moments_spc(spec, window_hwidth, v_atpeak)

    elif isinstance(spec, SpectralCube):
        return window_window_moments_spcube(spec, window_hwidth, v_atpeak)

    else:
        print "[ERROR] the input is invalid"
        return None


def window_moments_spc(spectrum, window_hwidth=3.0, v_atpeak=None, iter_refine=False):
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


def window_window_moments_spcube(maskcube, window_hwidth, v_atpeak=None):
    if v_atpeak is None:
        # find the peak of the integrated spectrum if v_atpeak isn't provided
        tot_spec = np.nansum(maskcube._data[:,]*maskcube.get_mask_array(), axis=(1,2))
        idx_peak = np.nanargmax(tot_spec)
        print "peak T_B: {0}".format(np.nanmax(tot_spec))
        v_atpeak = maskcube.spectral_axis[idx_peak].to(u.km/u.s).value
        print "v_atpeak: {0}".format(v_atpeak)

    vmax = v_atpeak + window_hwidth
    vmin = v_atpeak - window_hwidth

    # Extract the spectrum within the window defined around the main hyperfine components and take moments
    slab = maskcube.spectral_slab(vmin*u.km/u.s, vmax*u.km/u.s)
    m0 = slab.moment0(axis=0).value
    m1 = slab.moment1(axis=0).to(u.km/u.s).value
    m2 = (np.abs(slab.moment2(axis=0))**0.5).to(u.km/u.s).value

    return m0, m1, m2




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

    if moment0 is not None:
        #print "[WARNING]: moment0 map is provided, thus the user-provided tex and tau will not be used"
        # normalize the moment 0 map with respect to the norm_ref percentile value
        # e.g., 95 percentile value being normalized to have a value of 0.95
        norm_ref = 99.73
        mom0high = np.percentile(moment0[np.isfinite(moment0)], norm_ref)
        # may want to modify this normalization to be something a little simpler or physical (i.e., 99.73/100 ~ 1)
        m0Norm = moment0.copy()*norm_ref/100.0/mom0high
        tex_guess = m0Norm*tex_max
        tau_guess = m0Norm*tau_max

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
# utility functions

def get_rms_prefit(spectrum, window_hwidth, v_atpeak):

    s = spectrum

    vsys = v_atpeak*u.km/u.s
    throw = window_hwidth*u.km/u.s
    voff11 = voff_lines_dict['oneone']

    mask = np.ones(s.shape[0], dtype=np.bool)

    for deltav in voff11:
        mask *= (np.abs(s.xarr - (deltav * u.km / u.s + vsys)) > throw)

    d_rms = s.data.copy()

    return mad_std(d_rms[mask])