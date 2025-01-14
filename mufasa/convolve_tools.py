"""
The `mufasa.convolve_tools` module provides tools for processing spectral cubes with spatial
convolution, signal-to-noise masking, and edge trimming.
"""

from __future__ import print_function
__author__ = 'mcychen'

import numpy as np
import astropy.io.fits as fits
import FITS_tools
from astropy import units as u
from skimage.morphology import remove_small_objects, disk, opening, binary_erosion, dilation, remove_small_holes
from spectral_cube import SpectralCube
from radio_beam import Beam
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.convolution import Gaussian2DKernel, convolve
from FITS_tools.hcongrid import get_pixel_mapping
from scipy.interpolate import griddata
import scipy.ndimage as nd
from spectral_cube.utils import NoBeamError # imoprt NoBeamError, since cube most likely wasn't able to read the beam either
import gc

# for Astropy 6.1.4 forward compatibility
try:
    from astropy.units import UnitScaleError
except ImportError:
    from astropy.units.core import UnitScaleError

#=======================================================================================================================
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#=======================================================================================================================
# utility tools for convolve cubes

def convolve_sky_byfactor(cube, factor, savename=None, edgetrim_width=5, downsample=True, **kwargs):
    # factor = factor * 1.0 # probably unecessary, better option is factor = float(factor)

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube, use_dask=True)

    if edgetrim_width is not None:
        cube = edge_trim(cube, trim_width=edgetrim_width)

    hdr = cube.header

    # sanity check
    if hdr['CUNIT1'] != hdr['CUNIT2']:
        raise Exception("the spatial axis units for the cube do not match each other!")
        return None

    beamunit = getattr(u, hdr['CUNIT1'])
    bmaj = hdr['BMAJ'] * beamunit * factor
    bmin = hdr['BMIN'] * beamunit * factor
    pa = hdr['BPA']

    try:
        beam = Beam(major=bmaj, minor=bmin, pa=pa)
    except UnitScaleError:
        beam = Beam(major=bmaj, minor=bmin, pa=None)

    # convolve
    try:
        # for Astropy 6.1.4 forward compatibility
        cnv_cube = convolve_sky(cube, beam, **kwargs)
    except NoBeamError:
        cube = cube.with_beam(beam)
        cnv_cube = convolve_sky(cube, beam, **kwargs)

    if not np.isnan(cnv_cube.fill_value):
        cnv_cube = cnv_cube.with_fill_value(np.nan)

    if downsample:
        # regrid the convolved cube
        nhdr = FITS_tools.downsample.downsample_header(hdr, factor=factor, axis=1)
        nhdr = FITS_tools.downsample.downsample_header(nhdr, factor=factor, axis=2)
        nhdr['NAXIS1'] = int(np.rint(hdr['NAXIS1']/factor))
        nhdr['NAXIS2'] = int(np.rint(hdr['NAXIS2']/factor))
        newcube = cnv_cube.reproject(nhdr, order='bilinear')
    else:
        newcube = cnv_cube
    gc.collect()

    if savename != None:
        newcube.write(savename, overwrite=True)

    return newcube


def convolve_sky(cube, beam, snrmasked=False, iterrefine=False, snr_min=3.0):
    # return the convolved cube in the same gridding as the input
    # note: iterrefine masks data as well

    if not isinstance(cube, SpectralCube):
        cube = SpectralCube.read(cube, use_dask=True)

    if not np.isnan(cube.fill_value):
        cube = cube.with_fill_value(np.nan)

    mask = np.any(cube.mask.include(), axis=0)

    if snrmasked:
        planemask = snr_mask(cube, snr_min)
        plane_mask_size = np.sum(planemask)
        if plane_mask_size > 25:
            mask = mask & planemask
            logger.info("snr plane mask size = {}".format(plane_mask_size))
        else:
            logger.warning("snr plane mask too small (size = {}), no snr mask is applied".format(plane_mask_size))

    maskcube = cube.with_mask(mask)

    # enable huge operations (https://spectral-cube.readthedocs.io/en/latest/big_data.html for details)
    if maskcube.size > 1e8:
        logger.warning("maskcube is large ({} pixels)".format(maskcube.size))
    maskcube.allow_huge_operations = True
    cnv_cube = maskcube.convolve_to(beam)
    maskcube.allow_huge_operations = False
    gc.collect()

    if snrmasked and iterrefine:
        # use the convolved cube for new masking
        logger.debug("--- second iteration refinement ---")
        mask = cube.mask.include()
        planemask = snr_mask(cnv_cube, snr_min)
        plane_mask_size = np.sum(planemask)
        if np.sum(planemask) > 25:
             mask = mask*planemask
             logger.info("snr plane mask size = {}".format(plane_mask_size))
        else:
            logger.warning("snr plane mask too small (size = {}), no snr mask is applied".format(plane_mask_size))
        maskcube = cube.with_mask(mask)
        maskcube.allow_huge_operations = True
        cnv_cube = maskcube.convolve_to(beam)
        maskcube.allow_huge_operations = False
        gc.collect()

    return cnv_cube


def snr_mask(cube, snr_min=1.0, errmappath=None):
    # create a mask around the cube with a snr cut

    if errmappath is not None:
        errmap = fits.getdata(errmappath)

    else:
        # make a quick RMS estimate using median absolute deviation (MAD)
        #mask_gg = np.isfinite(cube._data)
        #errmap = mad_std(cube._data[mask_gg], axis=0)
        errmap = cube.mad_std(axis=0)#, how='ray')
        logger.info("median rms: {0}".format(np.nanmedian(errmap)))

    snr = cube.filled_data[:].value / errmap
    peaksnr = np.nanmax(snr, axis=0)

    #the snr map will inetiabley be noisy, so a little smoothing
    kernel = Gaussian2DKernel(1)
    peaksnr = convolve(peaksnr, kernel)

    def default_masking(snr, snr_min):
        planemask = (snr > snr_min)

        if planemask.size > 100:
            # attempt to remove noisy features
            planemask = binary_erosion(planemask, disk(1))
            planemask_im = remove_small_objects(planemask, min_size=9)
            if np.sum(planemask_im) > 9:
                # only adopt the erroded mask if there are objects left in it
                planemask = planemask_im
            # note, dialation is larger than erosion so the foot print is a bit more extended
            planemask = dilation(planemask, disk(3))

        return (planemask)

    planemask = default_masking(peaksnr, snr_min)
    del peaksnr  # Free memory
    gc.collect()

    return planemask


def edge_trim(cube, trim_width=3):
        # trim the edges by N pixels to guess the location of the peak emission
        mask = np.any(cube.mask.include(), axis=0)
        #mask = np.any(np.isfinite(cube._data), axis=0)
        if mask.size > 100:
            mask = binary_erosion(mask, disk(trim_width))
        mask = cube.mask.include() & mask

        return cube.with_mask(mask)


def regrid_mask(mask, header, header_targ, tightBin=True):
    # calculate scaling ratio between the two images
    yratio = np.abs(header['CDELT2']/header_targ['CDELT2'])
    xratio = np.abs(header['CDELT2']/header_targ['CDELT2'])
    maxratio = np.max([yratio, xratio])

    if (maxratio <= 0.5) & tightBin:
        # erode the mask a bit to avoid binning artifacts when downsampling
        s = 2
        kern = np.ones((s, s), dtype=bool)
        mask = binary_erosion(mask, structure=kern)

    # using the fits convention of x and y
    shape = (header_targ['NAXIS2'], header_targ['NAXIS1'])

    # regrid a boolean mask
    grid = get_pixel_mapping(header_targ, header)

    if (maxratio <= 0.5):
        # the mapping seems a little off for the y-axis when downsampling
        # works for factor of 2 grid, but may want to check and see if this is an issue with any relative pixel size grid
        grid[0] = grid[0] + 1.0
        outbd = grid[0]> shape[0]
        # make sure the coordinates are not out of bound
        grid[0][outbd] = grid[0][outbd] - 1.0

    grid = grid.astype(int)

    newmask = np.zeros(shape, dtype=bool)
    newmask[grid[0, mask], grid[1, mask]] = True

    if maxratio > 1:
        # dilate the mask to preserve the footprint
        s = int(maxratio - np.finfo(np.float32).eps) + 1
        kern = np.ones((s+1,s+1), dtype=bool)
        kern[-1,:] = False
        kern[:,0] = False
        newmask = nd.binary_dilation(newmask, structure=kern)

    return newmask


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


def get_celestial_hdr(header):
    # make a new header that only contains celestial (i.e., on-sky) information
    new_hdr = WCS(header).celestial.to_header()
    new_hdr['NAXIS1'] = header['NAXIS1']
    new_hdr['NAXIS2'] = header['NAXIS2']
    return new_hdr


