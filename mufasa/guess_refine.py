__author__ = 'mcychen'

#=======================================================================================================================
import numpy as np
from astropy.stats import mad_std
from astropy.wcs import WCS
from skimage.morphology import remove_small_objects, dilation, disk, remove_small_holes
from scipy.ndimage.filters import median_filter
from scipy.interpolate import CloughTocher2DInterpolator as intp
from scipy.interpolate import griddata
from FITS_tools.hcongrid import get_pixel_mapping
from astropy.convolution import Gaussian2DKernel, convolve

from scipy.spatial.qhull import QhullError
#=======================================================================================================================
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#=======================================================================================================================


def quick_2comp_sort(data_cnv, filtsize=2):
    # use median filtered vlsr & sigma maps as a velocity reference to sort the two components

    # arange the maps so the component with the least vlsr errors is the first component
    swapmask = data_cnv[8] > data_cnv[12]
    data_cnv = mask_swap_2comp(data_cnv, swapmask)

    # the use the vlsr error in the first component as the reference and sort the component based on their similarities
    # to this reference (similary bright structures should have similar errors)
    ref = median_filter(data_cnv[8], size=(filtsize, filtsize))
    swapmask = np.abs(data_cnv[8] - ref) > np.abs(data_cnv[12] - ref)
    data_cnv = mask_swap_2comp(data_cnv, swapmask)

    def dist_metric(p1, p2):
        # use the first map (the one that should have the smallest error, hense more reliable) to compute
        #  distance metric based on their similarities to the median filtered quantity
        p_refa = median_filter(p1, size=(filtsize, filtsize))
        #p_refb = median_filter(p2, size=(filtsize, filtsize))

        # distance of the current arangment to the median
        del_pa = np.abs(p1 - p_refa)

        # distance of the swapped arangment to the median
        del_pb = np.abs(p2 - p_refa)
        return del_pa, del_pb

    dist_va, dist_vb = dist_metric(data_cnv[0], data_cnv[4])
    dist_siga, dist_sigb = dist_metric(data_cnv[1], data_cnv[5])

    # use both the vlsr and the sigma as a distance metric
    swapmask = np.hypot(dist_va, dist_siga) > np.hypot(dist_vb, dist_sigb)

    data_cnv= mask_swap_2comp(data_cnv, swapmask)

    return data_cnv


def mask_swap_2comp(data_cnv, swapmask):
    # swap data over the mask
    data_cnv= data_cnv.copy()
    data_cnv[0:4,swapmask], data_cnv[4:8,swapmask] = data_cnv[4:8,swapmask], data_cnv[0:4,swapmask]
    data_cnv[8:12,swapmask], data_cnv[12:16,swapmask] = data_cnv[12:16,swapmask], data_cnv[8:12,swapmask]
    return data_cnv


def guess_from_cnvpara(data_cnv, header_cnv, header_target, mask=None):
    # a wrapper to make guesses based on the parameters fitted to the convolved data
    npara = 4
    ncomp = int(data_cnv.shape[0]/npara/2)

    data_cnv = data_cnv.copy()
    # clean up the maps based on vlsr errors
    data_cnv = simple_para_clean(data_cnv, ncomp, npara=npara)
    hdr_conv = get_celestial_hdr(header_cnv)
    data_cnv[data_cnv == 0] = np.nan
    data_cnv = data_cnv[0:npara*ncomp]


    for i in range (0, ncomp):
        data_cnv[i*npara:i*npara+npara] = refine_each_comp(data_cnv[i*npara:i*npara+npara], mask)

    # regrid the guess back to that of the original data
    hdr_final = get_celestial_hdr(header_target)

    kernel = Gaussian2DKernel(1)

    guesses_final = []

    # regrid the guesses
    for gss in data_cnv:
        newmask = np.isfinite(gss)
        # removal holes with areas that smaller than a 5 by 5 square
        newmask = remove_small_holes(newmask, 25)
        # create a mask to regrid over
        newmask = regrid(newmask, hdr_conv, hdr_final, dmask=None, method='nearest')
        newmask = newmask.astype('bool')

        new_guess = regrid(gss, hdr_conv, hdr_final, dmask=newmask)

        # expand the interpolation a bit, since regridding can often miss some pixels due to aliasing
        newmask_l = dilation(newmask)
        newmask_l = dilation(newmask_l)
        new_guess_cnv = convolve(new_guess, kernel, boundary='extend')

        new_guess_cnv[~newmask_l] = np.nan
        # retrain the originally interpolataed values within the original mask
        mask_finite = np.isfinite(new_guess)
        new_guess_cnv[mask_finite] = new_guess[mask_finite]
        guesses_final.append(new_guess_cnv)

    return np.array(guesses_final)


def tautex_renorm(taumap, texmap, tau_thresh = 0.21, tex_thresh = 15.0):
    from . import moment_guess as mmg

    # attempt to re-normalize the tau & text values at the optically thin regime (where the two are degenerate)
    # note, the latest recipe also works for the optically thick regime in principle
    # only emission with lower amplitude than TA_ltau_thres and tau < tau_thin will have tex > tex_thresh recalculated
    #  (i.e., expected to be optically thin

    isthin = np.logical_and(taumap < tau_thresh, np.isfinite(taumap))
    TA_lowtau = mmg.peakT(texmap[isthin], taumap[isthin])
    TA_ltau_thres = 0.5 # where tau ~1 for Tex = 3.5; tau with Ta above this diverges quickly
    # assume a fixed Tex for low TA
    tex_thin = 3.5      # note: at Tk = 30K, n = 1e3, N = 1e13, & sig = 0.2 km.s --> Tex = 3.49 K, tau = 0.8
    tau_thin = 1.0      # where the main hyperfines of NH3 (1,1) starts to get optically thick

    # for when tau is less than tau_thresh
    #texmap[isthin] = texmap[isthin]*taumap[isthin]/tau_thresh
    #taumap[isthin] = tau_thresh
    texmap[isthin] = mmg.get_tex(TA_lowtau, tau=tau_thresh) #note: tex can be higher than at 40K at Ta~7K
    taumap[isthin] = tau_thresh

    # optically thin gas are also unlikely to have high spatial density and thus high Tex
    hightex = np.logical_and(texmap > tex_thresh, np.isfinite(texmap))
    TA_hightex = mmg.peakT(texmap[hightex], taumap[hightex])
    mask = TA_hightex < TA_ltau_thres # only renormalize high tex when Ta is less than the threshold
    mask = np.logical_and(mask, taumap[hightex] < tau_thin)

    #texmap[hightex] = tex_thin
    #taumap[hightex] = texmap[hightex]*taumap[hightex]/tex_thin
    texmap[hightex][mask] = tex_thin
    taumap[hightex][mask] = mmg.get_tau(TA_hightex[mask], tex=tex_thin, nu=23.722634)

    # note, tau values that are too low will be taken care of by refine_each_comp()
    return taumap, texmap


def refine_each_comp(guess_comp, mask=None, v_range=None, sig_range=None):
    # refine guesses for each component, with values outside ranges specified below removed

    Tex_min = 3.0
    Tex_max = 8.0
    Tau_min = 0.2
    Tau_max = 8.0

    disksize = 1.0

    if mask is None:
        mask = master_mask(guess_comp)

    if v_range is None:
        vmin = None
        vmax = None
    else:
        vmin, vmax =v_range

    if sig_range is None:
        sigmin = None
        sigmax = None
    else:
        sigmin, sigmax = sig_range


    guess_comp[0] = refine_guess(guess_comp[0], min=vmin, max=vmax, mask=mask, disksize=disksize)
    guess_comp[1] = refine_guess(guess_comp[1], min=sigmin, max=sigmax, mask=mask, disksize=disksize)

    # re-normalize the degenerated tau & text for the purpose of estimate guesses
    guess_comp[3], guess_comp[2] = tautex_renorm(guess_comp[3], guess_comp[2], tau_thresh = 0.1)

    # place a more "strict" limits for Tex and Tau guessing than the fitting itself
    guess_comp[2] = refine_guess(guess_comp[2], min=Tex_min, max=Tex_max, mask=mask, disksize=disksize)
    guess_comp[3] = refine_guess(guess_comp[3], min=Tau_min, max=Tau_max, mask=mask, disksize=disksize)
    return guess_comp


def simple_para_clean(pmaps, ncomp, npara=4):
    # clean parameter maps based on their error values

    pmaps=pmaps.copy()

    # remove component with vlsrErr that is number of sigma off from the median as specified below
    std_thres = 2

    pmaps[pmaps == 0] = np.nan

    # loop through each component
    for i in range (0, ncomp):
        # get the STD and Medians of the vlsr errors
        std_vErr = mad_std(pmaps[(i+ncomp)*npara][np.isfinite(pmaps[(i+ncomp)*npara])])
        median_vErr = np.median(pmaps[(i+ncomp)*npara][np.isfinite(pmaps[(i+ncomp)*npara])])

        # remove outlier pixels
        mask = pmaps[(i+ncomp)*npara] > median_vErr + std_vErr*std_thres

        pmaps[i*npara:(i+1)*npara, mask] = np.nan
        pmaps[(i+ncomp)*npara:(i+ncomp+1)*npara, mask] = np.nan

    return pmaps


def get_celestial_hdr(header):
    # make a new header that only contains celestial (i.e., on-sky) information
    new_hdr = WCS(header).celestial.to_header()
    new_hdr['NAXIS1'] = header['NAXIS1']
    new_hdr['NAXIS2'] = header['NAXIS2']
    return new_hdr


def master_mask(pcube):
    # create a 2D mask over where any of the paramater map has finite values
    mask = np.any(np.isfinite(pcube), axis=0)
    mask = mask_cleaning(mask)
    return mask


def mask_cleaning(mask):
    # designed to clean a noisy map, with a footprint that is likely slightly larger
    mask = remove_small_objects(mask, min_size=9)
    mask = dilation(mask, disk(1))
    mask = remove_small_holes(mask, 9)
    return mask


def regrid(image, header1, header2, dmask=None, method='cubic'):
    # similar to hcongrid from FITS_tools, but uses scipy.interpolate.griddata to interpolate over nan values
    grid1 = get_pixel_mapping(header1, header2)

    xline = np.arange(image.shape[1])
    yline = np.arange(image.shape[0])
    X,Y = np.meshgrid(xline, yline)

    mask = np.isfinite(image)

    if dmask is None:
        dmask = np.ones(grid1[0].shape, dtype=bool)

    return griddata((X[mask],Y[mask]), image[mask], (grid1[1]*dmask, grid1[0]*dmask), method=method, fill_value=np.nan)


def refine_guess(map, min=None, max=None, mask=None, disksize=1):
    # refine parameter maps by outlier-fitering, masking, and interpolating
    map = map.copy()

    if min is not None:
        map[map<min] = np.nan
    if max is not None:
        map[map>max] = np.nan

    if np.sum(np.isfinite(map)) == 0:
        # if there are no valid pixel in the guesses, set it to one of the limits or zero
        if min is not None:
            map[:] = min
        elif max is not None:
            map[:] = max
        else:
            map[:] = 0.0
        return map

    #map_med = median_filter(map, footprint=disk(disksize))
    #if np.sum(np.isfinite(map_med)) == 0:

    kernel = Gaussian2DKernel(disksize)
    map = convolve(map, kernel, boundary='extend')

    if mask is None:
        mask = np.isfinite(map)
        mask = mask_cleaning(mask)

    def interpolate(map, mask):
        # interpolate over the dmask footprint
        xline = np.arange(map.shape[1])
        yline = np.arange(map.shape[0])
        X,Y = np.meshgrid(xline, yline)
        itpmask = np.isfinite(map)
        C = intp((X[itpmask],Y[itpmask]), map[itpmask])

        # interpolate over the dmask footprint
        zi = C(X*mask,Y*mask)
        return zi

    def interpolate_via_cnv(map):
        kernel = Gaussian2DKernel(3)
        zi = convolve(map, kernel, boundary='extend')
        # retrain the original median_filtered map over its original positions
        map_finite = np.isfinite(map)
        zi[map_finite] = map[map_finite]
        return zi

    if np.sum(mask) >= 2:
        try:
            # interpolate the mask
            zi = interpolate(map, mask)
        except QhullError as e:
            logger.warning("qhull input error found; astropy convolve will be used instead")
            zi = interpolate_via_cnv(map) # use astropy convolve as a proxy for interpolation
            
        except ValueError as e:
            logger.error("ValueError found (no points given); astropy convolve will be used instead")
            zi = interpolate_via_cnv(map)
    else:
        zi = interpolate_via_cnv(map)

    return zi


def save_guesses(paracube, header, savename, ncomp=2):
    # a method to save the fitted parameter cube with relavent header information
    import copy
    from astropy.io import fits

    npara = 4

    hdr_new = copy.deepcopy(header)

    # write the header information for each plane (i.e., map)
    for i in range (0, ncomp):
        hdr_new['PLANE{0}'.format(i*npara+0)] = 'VELOCITY_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+1)] = 'SIGMA_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+2)] = 'TEX_{0}'.format(i+1)
        hdr_new['PLANE{0}'.format(i*npara+3)] = 'TAU_{0}'.format(i+1)

    hdr_new['CDELT3']= 1
    hdr_new['CTYPE3']= 'FITPAR'
    hdr_new['CRVAL3']= 0
    hdr_new['CRPIX3']= 1

    fitcubefile = fits.PrimaryHDU(data=paracube, header=hdr_new)
    fitcubefile.writeto(savename ,overwrite=True)