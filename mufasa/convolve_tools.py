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
import dask.array as da
from dask.diagnostics import ProgressBar

# for Astropy 6.1.4 forward compatibility
try:
    from astropy.units import UnitScaleError
except ImportError:
    from astropy.units.core import UnitScaleError

from .utils.memory import monitor_peak_memory
from .utils import dask_utils

#=======================================================================================================================
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#=======================================================================================================================
# utility tools for convolve cubes

@monitor_peak_memory()
def convolve_sky_byfactor(cube, factor, savename=None, edgetrim_width=5, downsample=True,
                          rechunk=True, scheduler='synchronous', **kwargs):

    kwargs['rechunk'] = rechunk
    kwargs['scheduler'] = scheduler

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
        cnv_cube = convolve_sky(cube, beam,  **kwargs)

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

@monitor_peak_memory()
def convolve_sky(cube, beam, snrmasked=False, iterrefine=False, snr_min=3.0, rechunk=True, scheduler='synchronous'):
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

    if isinstance(maskcube._data, da.Array):
        # persist to clean up the graph and make subsequent calculation
        # more memory effecient
        maskcube._data = dask_utils.persist_and_clean(maskcube._data)
        maskcube._data = dask_utils.reset_graph(maskcube._data)
        gc.collect()

    # convolve the cube
    cnv_cube = _convolve_to(maskcube, beam, scheduler, rechunk=rechunk)
    gc.collect()

    if snrmasked and iterrefine:
        # use the convolved cube for new masking
        logger.debug("--- second iteration refinement ---")
        mask = cube.mask.include()

        if rechunk and isinstance(cnv_cube._data, da.Array):
            # chunk the convoved cube back for more effecient operation
            cnv_cube = _rechunk(cnv_cube, rechunk=cube._data.chunks)

        planemask = snr_mask(cnv_cube, snr_min)
        plane_mask_size = np.sum(planemask)
        if np.sum(planemask) > 25:
             mask = mask*planemask
             logger.info("snr plane mask size = {}".format(plane_mask_size))
        else:
            logger.warning("snr plane mask too small (size = {}), no snr mask is applied".format(plane_mask_size))
        maskcube = cube.with_mask(mask)

        cnv_cube = _convolve_to(maskcube, beam, scheduler, rechunk=rechunk)
        gc.collect()

    return cnv_cube


@monitor_peak_memory()
def _convolve_to(cube, beam, scheduler='synchronous', rechunk=False, save_to_tmp_dir=False):
    # base function to handle convolution
    if isinstance(cube._data, da.Array):

        if rechunk:
            cube = _rechunk(cube, rechunk)

        print(f"convolving cube with chunks: {cube._data.chunks}")
        print(f"=============== convolving cube with \'{scheduler}\' scheduler =====================")

        with cube.use_dask_scheduler('threads'):
            pbar = ProgressBar()
            pbar.register()
            cnv_cube = cube.convolve_to(beam, save_to_tmp_dir=save_to_tmp_dir)
            if not save_to_tmp_dir:
                # compute the convolution now
                cnv_cube._data = dask_utils.persist_and_clean(cnv_cube._data)
    else:
        # enable huge operations (https://spectral-cube.readthedocs.io/en/latest/big_data.html for details)
        if cube.size > 1e8:
            logger.warning("cube is large ({} pixels)".format(cube.size))
        cube.allow_huge_operations = True
        cnv_cube = cube.convolve_to(beam)
        cube.allow_huge_operations = False

    return cnv_cube


def _rechunk(cube, rechunk):
    # rechunk the cube as needed
    # rechunk to not overwhelm the convlution process with sub-optimal chunks
    # rechunk by planes (not breaking up the spatial image) is recommended

    chunks_og = cube._data.chunks  # record the original chunking
    if isinstance(rechunk, tuple):
        logger.info("using user provided tuple for rechunking")
        cube = cube.rechunk(rechunk)
    elif isinstance(rechunk, str):
        cube = cube.rechunk(str)
    else:
        cube = cube.rechunk('auto')

    return cube


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


