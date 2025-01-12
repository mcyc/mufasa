"""
The `mufasa.UltraCube` module provides tools for processing, analyzing, and visualizing spectral
cubes, particularly for fitting multi-component spectral models.
"""

from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

#======================================================================================================================#
import os
import warnings
from functools import wraps
import numpy as np

from spectral_cube import SpectralCube
# prevent any spectral-cube related warnings from being displayed.
from spectral_cube.utils import SpectralCubeWarning
warnings.filterwarnings(action='ignore', category=SpectralCubeWarning, append=True)
from copy import copy

import pyspeckit
import gc
from astropy import units as u
from astropy.units import UnitConversionError
import scipy.ndimage as nd
from astropy.stats import mad_std

import dask.array as da

from . import aic
from . import multi_v_fit as mvf
from . import convolve_tools as cnvtool
from .spec_models import meta_model
from .utils.multicore import validate_n_cores
from .utils import dask_utils, dask_ops
from .visualization.spec_viz import Plotter
from . import PCube

#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

class UltraCube(object):
    """
    A framework for multi-component spectral cube analysis and model fitting.

    The `UltraCube` class manages spectral cubes, supports multi-component model
    fitting, and provides tools for statistical evaluation, visualization, and
    residual analysis. It is designed to handle large-scale spectral datasets
    efficiently.

    Parameters
    ----------
    cubefile : str, optional
        Path to the FITS cube file. If not provided, a `SpectralCube` instance must be
        supplied via the `cube` parameter.
    cube : SpectralCube, optional
        A `SpectralCube` instance representing the input spectral data. This parameter
        is used only if `cubefile` is not provided.
    fittype : str, optional
        Type of spectral model to fit. Must be compatible with the `MetaModel` framework.
    snr_min : float, optional
        Minimum signal-to-noise ratio for considering a voxel during fitting.
    rmsfile : str, optional
        Path to the file containing RMS values for the cube.
    cnv_factor : int, optional, default=2
        Factor by which to spatially convolve the cube for pre-processing.
    n_cores : int or bool, optional, default=True
        Number of CPU cores to use for parallel processing. If `True`, uses all available
        cores minus one.

    Notes
    -----
    - The `UltraCube` class is designed to handle cubes with varying spatial
      and spectral resolutions. It uses the `pyspeckit` library for fitting
      and the `MetaModel` framework for defining custom spectral models.
    - Convolution, masking, and statistical evaluation tools are provided for
      pre- and post-processing of data.
    - If a file path is provided via `cubefile`, the cube is loaded using
      `SpectralCube.read`.
    - The convolution factor (`cnv_factor`) is applied to the spatial axes to
      reduce resolution before processing.
    - The number of CPU cores (`n_cores`) defaults to all but one available core
      for parallel computations, but can be explicitly set.

    """

    def __init__(self, cubefile=None, cube=None, fittype=None, snr_min=None, rmsfile=None, cnv_factor=2, n_cores=True):
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
        self.plotter = None
        self.meta_model = None

        if cubefile is not None:
            self.cubefile = cubefile
            self.load_cube(cubefile)
        else:
            if hasattr(cube, 'spectral_axis'):
                # Load from a SpectralCube instance
                self.cube = cube

        if fittype is not None:
            # for the current usage, setting ncomp=1 is fine. Changes to MetalModel in the future will be needed
            # to elimiate needing ncomp during its intialization
            self.meta_model = meta_model.MetaModel(fittype=fittype, ncomp=1)

        if not snr_min is None:
            self.snr_min = snr_min

        if not rmsfile is None:
            self.rmsfile = rmsfile


    def make_header2D(self):
        """
        Create a 2D header by removing the specified axis.

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header to modify.
        axis : int
            Axis to remove from the header.

        Returns
        -------
        astropy.io.fits.Header
            Modified 2D header.
        """
        return mvf.make_header(ndim=2, ref_header=self.cube.header)

    def load_cube(self, fitsfile):
        """
        Load a spectral cube from a FITS file and convert its units to K and km/s.

        The function ensures the cube has intensity units in Kelvin and a spectral axis
        in velocity units (km/s). If the cube lacks a rest frequency, it assigns one
        based on the spectral model.

        Parameters
        ----------
        fitsfile : str
            Path to the FITS cube file.

        Returns
        -------
        None
        """
        cube = SpectralCube.read(fitsfile, use_dask=True)
        cube = to_K(cube)

        if cube.spectral_axis.unit.is_equivalent(u.Hz):
            # assign rest frequency from the model before spectral axis velocity conversion
            if not hasattr(cube.wcs.wcs, 'restfrq') or np.isnan(cube.wcs.wcs.restfrq):
                logger.warning("The cube has no reference rest frequency. "
                               "The rest frequency of the spectral model will be used instead")
                cube = cube.with_spectral_unit(u.Hz, rest_value=mod_info.rest_value)

        self.cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')


    def load_pcube(self, pcube_ref=None, pyspeckit=False):
        """
        Load a cube into a pyspeckit.SpectralCube object from a .fits file.

        Parameters
        ----------
        pcube_ref : pyspeckit.SpectralCube
            A pyspeckit.SpectralCube object that works with the same data cube. If provided, the new pcube's cube attribute
            will be pointed towards this reference cube to avoid reduandance and save memory

        Returns
        -------
        pcube : pyspeckit.SpectralCube
            The pyspeckit.SpectralCube object needed work with the fitting
        """
        # read the cube first as a SpectralCube object to performe unit conversion
        # Note: dask is set to False to ensure it's compitable with some of pyspeckit's functions
        # this loading mehod is not memory efficient, but needed workaround at the moment
        cube_temp = SpectralCube.read(self.cubefile, dask=False)
        cube_temp = to_K(cube_temp) # convert the unit to K;

        if pyspeckit:
            pcube = pyspeckit.Cube(cube=cube_temp)
        else:
            pcube = PCube.PCube(cube=cube_temp)

        # premptively release memory
        del cube_temp
        gc.collect()

        if pcube_ref is not None:
            if is_K(pcube_ref.unit):
                # set pointer of the new pcube's cube to the reference one to remove redundancy and conserve memory
                pcube.cube = pcube_ref.cube

        if pcube.xarr.refX is None or np.isnan(pcube.wcs.wcs.restfrq):
            # Specify the reference rest frequency if not present
            logger.warning("The cube has no reference rest frequency."
                           " The rest frequency of the spectral model will be used instead")
            pcube.xarr.refX = mod_info.freq_dict[linename] * u.Hz

        if pcube.xarr.velocity_convention is None:
            pcube.xarr.velocity_convention = 'radio'

        return pcube



    def convolve_cube(self, savename=None, factor=None, edgetrim_width=5):
        """
        Convolve the SpectralCube to a lower spatial resolution by a specified factor.

        Parameters
        ----------
        savename : str, optional
            Path to save the convolved cube. If None, the convolved cube is not saved.
        factor : int, optional
            Factor by which to spatially convolve the cube. If None, the default `self.cnv_factor` is used.
        edgetrim_width : int, optional
            Number of pixels to trim at the edges after convolution. Default is 5.

        Returns
        -------
        None

        Notes
        -----
        - Convolution reduces the spatial resolution by the specified factor.
        - The resulting convolved cube is stored in `self.cube_cnv`.
        - This method uses the `convolve_sky_byfactor` function to perform spatial convolution.

        Raises
        ------
        ValueError
            If the cube cannot be convolved due to incompatible dimensions or data types.
        """
        if factor is None:
            factor = self.cnv_factor
        self.cube_cnv = convolve_sky_byfactor(self.cube, factor, savename, edgetrim_width=edgetrim_width)


    def get_cnv_cube(self, filename=None):
        """
        Load the convolved cube if the file exists, or create one if it does not.

        Parameters
        ----------
        filename : str, optional
            Path to the convolved cube file. If None, a new convolved cube is created using the default `cnv_factor`.

        Returns
        -------
        None
        """
        if filename is None:
            self.convolve_cube(factor=self.cnv_factor)
        elif os.path.exists(filename):
            self.cube_cnv = SpectralCube.read(filename, use_dask=True)
        else:
            logger.warning("The specified convolved cube file does not exist.")


    def fit_cube(self, ncomp, simpfit=False, **kwargs):
        """
        Fit the spectral cube with the specified number of components.

        Parameters
        ----------
        ncomp : int or list of int
            Number of components for the model. If a list is provided, fits are performed for each component.
        simpfit : bool, optional
            Whether to use a simplified fitting method (`cubefit_simp`) instead of the general fitting method (`cubefit_gen`).
        **kwargs : dict, optional
            Additional keyword arguments passed to `pyspeckit.Cube.fiteach`. Includes options like `multicore` and `snr_min`.

        Returns
        -------
        None

        Notes
        -----
        - The `multicore` parameter in kwargs controls parallel processing. By default, the number of cores set in `self.n_cores` is used.
        - Fit results are stored in `self.pcubes` for each component.

        Raises
        ------
        TypeError
            If `ncomp` is not an integer or a list of integers.
        """
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
            # initiate pcube objects for fitting and model handling
            if self.pcubes:
                # use a pre-existing pcube as reference
                _, pcube_ref = next(iter(self.pcubes.items()))
                self.pcubes[str(nc)] = self.load_pcube(pcube_ref=pcube_ref)
            else:
                self.pcubes[str(nc)] = self.load_pcube()
            self.pcubes[str(nc)] = fit_cube(self.cube, self.pcubes[str(nc)], fittype=self.fittype, simpfit=simpfit, ncomp=nc, **kwargs)

            if self.pcubes[str(nc)].has_fit.sum() > 0 and hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)].get_modelcube(multicore=kwargs['multicore']) > 0
                if isinstance(mod_mask, da.Array):
                    mod_mask.persist()
                self.include_model_mask(mod_mask)
            gc.collect()

    def has_fit(self, ncomp):
        """
        Return a mask indicating which pixels have been fitted with a model.

        Parameters
        ----------
        ncomp : int
            The number of components for the model being checked. This value is used to identify the modelled
            parameter cube (`parcube`) associated with the number of components.

        Returns
        -------
        np.ndarray
            A 2D boolean array with the same spatial dimensions as the data, where `True` indicates pixels
            that have been fitted (i.e., contain non-zero, finite values in the parameter cube for the specified
            model component) and `False` indicates pixels that were not fitted.
        """
        parcube = self.pcubes[str(ncomp)].parcube
        mask = np.any(parcube != 0, axis=0)
        mask = mask & np.any(np.isfinite(parcube), axis=0)
        return mask


    def include_model_mask(self, mask):
        # update the mask that shows were all the models are non-zero

        if self.master_model_mask is None:
            self.master_model_mask = mask
        else:
            logger.debug("Using logical_or() to join masks")
            self.master_model_mask = np.logical_or(self.master_model_mask, mask)

        if isinstance(self.master_model_mask, da.Array):
            self.master_model_mask = dask_utils.persist_and_clean(self.master_model_mask, debug=True, visualize_filename="include_mod_mask.svg")

    def reset_model_mask(self, ncomps, multicore=True):
        #reset and re-generate master_model_mask for all the components in ncomps
        self.master_model_mask = None

        for nc in ncomps:
            if nc > 0 and hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)]._modelcube > 0
                mod_mask = dask_utils.persist_and_clean(mod_mask, debug=True, visualize_filename="reset_mod_mask.svg")
                self.include_model_mask(mod_mask)
            gc.collect()


    def save_fit(self, savename, ncomp, header_note=None):
        # note, this implementation currently relies on
        if hasattr(self.pcubes[str(ncomp)], 'parcube'):
            save_fit(self.pcubes[str(ncomp)], savename, ncomp, header_note=header_note)
        else:
            logger.warning("no fit was performed and thus no file will be saved")


    def load_model_fit(self, filename, ncomp, calc_model=True, multicore=None, pyspeckit=False):
        self.pcubes[str(ncomp)] = load_model_fit(self.cubefile, filename, ncomp, self.fittype, pyspeckit)
        if calc_model:
            if multicore is None: multicore = self.n_cores
            # update model mask
            mod_mask = self.pcubes[str(ncomp)].get_modelcube(multicore=multicore) > 0
            if isinstance(mod_mask, da.Array):
                mod_mask = dask_utils.persist_and_clean(mod_mask, debug=True, visualize_filename="mod_mask_at_load.svg")
            gc.collect()
            self.include_model_mask(mod_mask)

    def get_residual(self, ncomp, multicore=None):
        """
        Calculate the residual cube by subtracting the fitted model from the data.

        Parameters
        ----------
        ncomp : int
            Number of components used in the fitted model.
        multicore : int or bool, optional
            Number of cores to use for computation. Defaults to `self.n_cores`.

        Returns
        -------
        np.ndarray or dask.array.Array
            Residual array representing the difference between the data and the model fit.

        Notes
        -----
        - Residual cubes are stored in `self.residual_cubes` for reuse.
        - Residuals are computed for pixels with valid model fits.

        Raises
        ------
        ValueError
            If no model fit is available for the specified number of components.
        """
        if multicore is None: multicore = self.n_cores
        compID = str(ncomp)
        model = self.pcubes[compID].get_modelcube(multicore=multicore)
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
        rrs, nsamp = calc_rss(self, ncomp, usemask=True, mask=mask, return_size=True, update_cube=False,
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

    def get_plotter(self, update=False, spec_unit='km/s', **kwargs):
        """
        Initialize or update the Plotter instance for visualizing fitted spectra.

        Parameters
        ----------
        update : bool, optional
            If True, update the existing plotter instance (default is False).
        spec_unit : str, optional
            The spectral unit to use for plotting the spectral axis (default is 'km/s').
        **kwargs : dict, optional
            Additional keyword arguments passed to the `Plotter` class for initialization.

        Returns
        -------
        None
        """
        if self.plotter is None or update:
            self.plotter = Plotter(self, fittype=self.fittype, spec_unit=spec_unit, **kwargs)

    def plot_spec(self, x, y, ax=None, xlab=None, ylab=None, **kwargs):
        """
        Plot a single spectrum at (x, y).

        Parameters
        ----------
        x : int
            X-coordinate of the pixel.
        y : int
            Y-coordinate of the pixel.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created.
        xlab : str, optional
            X-axis label. Default is the LSR velocity label.
        ylab : str, optional
            Y-axis label. Default is the main beam temperature label.
        **kwargs : dict
            Additional keyword arguments passed to `plot_spec`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        self.get_plotter()
        return self.plotter.plot_spec(x, y, ax=ax, xlab=xlab, ylab=ylab, **kwargs)

    def plot_spec_grid(self, x, y, size=3, xsize=None, ysize=None, xlim=None, ylim=None, figsize=None, **kwargs):
        """
        Plot a grid of spectra centered at (x, y).

        Parameters
        ----------
        x : int
            X-coordinate of the central pixel.
        y : int
            Y-coordinate of the central pixel.
        size : int, optional
            Size of the grid (must be odd). Default is 3.
        xsize : int, optional
            Number of columns in the grid. Default is size.
        ysize : int, optional
            Number of rows in the grid. Default is size.
        xlim : tuple, optional
            X-axis limits for the plot, in their native units.
        ylim : tuple, optional
            Y-axis limits for the plot, in their native units.
        figsize : tuple, optional
            Size of the figure.
        **kwargs : dict
            Additional keyword arguments passed to `plot_spec_grid`.
        """
        self.get_plotter()
        self.plotter.plot_spec_grid(x, y, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim, figsize=figsize,
                                    **kwargs)

    def plot_fit(self, x, y, ncomp, ax=None, **kwargs):
        """
        Plot a model fit for a spectrum at (x, y).

        Parameters
        ----------
        x : int
            X-coordinate of the pixel.
        y : int
            Y-coordinate of the pixel.
        ncomp : int
            The component number to plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments passed to `plot_fit`.
        """
        self.get_plotter()
        if ax is None:
            fig, ax = self.plot_spec(x, y)
        self.plotter.plot_fit(x, y, ax, ncomp, **kwargs)

    def plot_fits_grid(self, x, y, ncomp, size=3, xsize=None, ysize=None, xlim=None, ylim=None,
                       figsize=None, origin='lower', mod_all=True, savename=None, **kwargs):
        """
        Plot a grid of model fits centered at (x, y).

        Parameters
        ----------
        x : int
            X-coordinate of the central pixel.
        y : int
            Y-coordinate of the central pixel.
        ncomp : int
            The component number to plot.
        size : int, optional
            Size of the grid (must be odd). Default is 3.
        xsize : int, optional
            Number of columns in the grid. Default is size.
        ysize : int, optional
            Number of rows in the grid. Default is size.
        xlim : tuple, optional
            X-axis limits for the plot, in their native units.
        ylim : tuple, optional
            Y-axis limits for the plot, in their native units.
        figsize : tuple, optional
            Size of the figure.
        origin : {'lower', 'upper'}, optional
            Origin of the grid. Default is 'lower'.
        mod_all : bool, optional
            Whether to plot all model components. Default is True.
        savename : str, optional
            If provided, save the figure to the given filename.
        **kwargs : dict
            Additional keyword arguments passed to `plot_fits_grid`.
        """
        self.get_plotter()
        self.plotter.plot_fits_grid(x, y, ncomp, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim,
                                    figsize=figsize, origin=origin, mod_all=mod_all, savename=savename, **kwargs)


class UCubePlus(UltraCube):
    """
    A subclass of UltraCube that includes directory management for parameter maps and model fits.
    """
    __module__ = "mufasa.UltraCube"  # Explicitly set the module

    def __init__(self, cubefile, cube=None, fittype=None, paraNameRoot=None, paraDir=None, **kwargs):
        """
        Initialize the UCubePlus object.

        Parameters
        ----------
        cubefile : str
            Path to the .fits cube file.
        cube : SpectralCube, optional
            A spectral cube object. Used if `cubefile` is not provided.
        paraNameRoot : str, optional
            Root name for the parameter map files. If None, the cube file name is used as the basis.
        paraDir : str, optional
            Directory to store the parameter map files. If None, a default directory is created.
        fittype : str, optional
            Keyword for the spectral model to be fitted.
        **kwargs
            Additional keyword arguments passed to the UltraCube initializer.

        Returns
        -------
        None
        """
        super().__init__(cubefile, cube, fittype=fittype, **kwargs)

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

    def read_model_fit(self, ncomps, read_conv=False, **kwargs):
        """
        Load model fits if they exist; otherwise, perform the fitting.

        .. :noindex:

        Parameters
        ----------
        ncomps : list of int
            List of the number of components to load or fit.
        read_conv : bool, optional
            If True, attempts to read fits for convolved cubes as well. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments passed to the fitting methods.

        Returns
        -------
        None

        Notes
        -----
        - If fits files are not found, the fitting process is triggered, and results are saved to disk.
        - File paths for each component are stored in `self.paraPaths`.

        Raises
        ------
        FileNotFoundError
            If the specified fit files are not found and fitting cannot proceed.
        """
        for nc in ncomps:
            if str(nc) not in self.paraPaths:
                self.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(self.paraDir, self.paraNameRoot, nc)

            super().load_model_fit(self.paraPaths[str(nc)], ncomp=nc, calc_model=False)

            if 'conv' in self.paraPaths[str(nc)]:
                logger.info(f'Reading convolved cube fits for {nc} component(s)')


    def get_model_fit(self, ncomp, update=True, **kwargs):
        """
        Load the model fits if they exist, or perform fitting if they don't.

        Parameters
        ----------
        ncomp : int or list of int
            Number of components for the model fit. If a list is provided, a fits will be performed for each component.
        update : bool, optional
            Whether to update (i.e., re-fit) the cube even if model fits already exist (default is True).
        **kwargs
            Additional keyword arguments passed to `pyspeckit.Cube.fiteach` if the fitting needs to be updated.

        Returns
        -------
        None
        """
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

def fit_cube(cube, pcube, fittype, simpfit=False, **kwargs):
    """
    Fit the spectral cube using the specified fitting type.

    Parameters
    ----------
    cube :
        The cube to be fitted.
    fittype : str
        The type of spectral model to be used for fitting.
    simpfit : bool, optional
        If True, use a simplified fitting method (`cubefit_simp`) without pre-processing (default is False).
    **kwargs
        Additional keyword arguments passed to `pyspeckit.Cube.fiteach`.

    Returns
    -------
    pyspeckit.Cube
        The fitted pyspeckit cube.
    """
    if simpfit:
        # fit the cube with the provided guesses and masks with no pre-processing
        return mvf.cubefit_simp(cube, pcube, fittype=fittype, **kwargs)
    else:
        return mvf.cubefit_gen(cube, pcube, fittype=fittype, **kwargs)



def save_fit(pcube, savename, ncomp, header_note=None):
    """
    Save the fitted parameter cube to a .fits file with the appropriate header.

    Parameters
    ----------
    pcube : pyspeckit.Cube
        The fitted parameter cube to be saved.
    savename : str
        The path where the .fits file will be saved.
    ncomp : int
        The number of components in the model.
    header_note : str, optional
        A single-line comment to include in the FITS header metadata.

    Returns
    -------
    None
    """
    # specifically save ammonia multi-component model with the right fits header
    mvf.save_pcube(pcube, savename, ncomp, header_note=header_note)



def load_model_fit(cube, filename, ncomp, fittype, pyspeckit=False):
    """
    Load the spectral fit results from a .fits file.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube object to which the fit results will be loaded.
    filename : str
        Path to the .fits file containing the fitted parameters.
    ncomp : int
        Number of components in the model.
    fittype : str
        The keyword that designates the model. Currently available options are 'nh3_multi_v' and 'n2hp_multi_v'.

    Returns
    -------
    pyspeckit.Cube
        The fitted pyspeckit cube with the loaded model.
    """
    if pyspeckit:
        pcube = pyspeckit.Cube(cube)
    else:
        pcube = PCube.PCube(cube)

    # register fitter
    if fittype == 'nh3_multi_v':
        linename = 'oneone'
        from .spec_models import ammonia_multiv as ammv
        fitter = ammv.nh3_multi_v_model_generator(n_comp=ncomp, linenames=[linename])

    elif fittype == 'n2hp_multi_v':
        linename = 'onezero'
        from .spec_models import n2hp_multiv as n2hpmv
        fitter = n2hpmv.n2hp_multi_v_model_generator(n_comp=ncomp, linenames=[linename])

    pcube.specfit.Registry.add_fitter(fittype, fitter, fitter.npars)
    pcube.xarr.velocity_convention = 'radio'
    pcube.load_model_fit(filename, npars=fitter.npars, fittype=fittype)
    gc.collect()
    return pcube



def convolve_sky_byfactor(cube, factor, savename=None, **kwargs):
    """
    Convolve the spatial dimensions of a spectral cube by a specified factor.

    This function reduces the spatial resolution of the input cube by convolving
    it with a Gaussian kernel scaled by the given factor. The resulting cube can
    optionally be saved to disk.

    Parameters
    ----------
    cube : SpectralCube
        The input spectral cube to be convolved. Must be an instance of
        `spectral_cube.SpectralCube`.
    factor : int
        The factor by which to reduce the spatial resolution. A factor of 2
        doubles the beam size, effectively halving the spatial resolution.
    savename : str, optional
        The file path to save the convolved cube. If None, the convolved cube
        is not saved. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments passed to the convolution function
        (e.g., to customize the kernel or handle edge cases).

    Returns
    -------
    convolved_cube : SpectralCube
        The convolved spectral cube with reduced spatial resolution.

    Notes
    -----
    - Convolution is performed only on the spatial dimensions of the cube.
    - The kernel size is determined automatically based on the specified `factor`.
    - This function can be computationally intensive for large cubes. Consider
      using memory-efficient techniques if applicable.

    Raises
    ------
    ValueError
        If the input `cube` is not compatible with the convolution operation
        or if the specified `factor` is invalid.
    IOError
        If an error occurs while saving the convolved cube to the specified path.

    """
    return cnvtool.convolve_sky_byfactor(cube, factor, savename, **kwargs)

#======================================================================================================================#
# UltraCube based methods

def calc_rss(ucube, compID, usemask=True, mask=None, return_size=True, update_cube=False, planemask=None, expand=20):
    """
    Calculate the residual sum of squares (RSS) for a spectral cube model fit.

    The RSS is computed as the sum of squared differences between the observed data
    and the model values for each voxel in the spectral cube. The calculation can be
    restricted using masks and expanded spectral regions.

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing the spectral data and fitted models.
    compID : int or str
        The component ID or number of components in the model to evaluate. If `compID` is 0,
        the model is assumed to be a flat baseline (`y = 0`).
    usemask : bool, optional
        Whether to apply a mask during the RSS calculation. If True, regions where the
        model is zero or invalid are excluded. Default is True.
    mask : numpy.ndarray, optional
        A 3D boolean array specifying which voxels to include in the calculation.
        If None and `usemask` is True, a default mask derived from the model is used.
    return_size : bool, optional
        Whether to return the effective sample size (number of valid voxels) for each spatial pixel.
        Default is True.
    update_cube : bool, optional
        Whether to update the `UltraCube` instance with the calculated RSS values. Default is False.
    planemask : numpy.ndarray, optional
        A 2D spatial boolean array specifying which spatial pixels to calculate RSS for.
        If None, all spatial pixels are considered. Default is None.
    expand : int, optional
        Number of spectral channels to expand the mask region for RSS calculation. Default is 20.

    Returns
    -------
    rss : numpy.ndarray
        A 2D array containing the RSS values for each spatial pixel.
    nsamp : numpy.ndarray, optional
        A 2D array containing the effective sample size (number of valid spectral samples)
        for each spatial pixel. Returned only if `return_size` is True.
    final_mask : numpy.ndarray, optional
        The 3D boolean mask used in the RSS calculation. Returned only if explicitly required
        by the caller.

    Notes
    -----
    The RSS is calculated as:

        RSS = Σ (data - model)²

    where the sum is taken over spectral channels for each spatial pixel.

    If `expand > 0`, the mask region is extended by the specified number of spectral channels.

    Raises
    ------
    ValueError
        If the dimensions of the data cube and model cube do not match, or if an invalid mask is provided.
    """
    if isinstance(compID, int):
        compID = str(compID)

    cube = ucube.cube

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(update=False, multicore=ucube.n_cores)

    gc.collect()
    return get_rss(cube, modcube, expand=expand, usemask=usemask, mask=mask, return_size=return_size,
                   planemask=planemask)


def calc_chisq(ucube, compID, reduced=False, usemask=False, mask=None, expand=20):
    """
    Calculate the chi-squared (χ²) or reduced chi-squared value for a spectral cube model fit.

    This function computes the goodness-of-fit by comparing the data in the spectral cube
    to the model values. Optionally, the calculation can include masking and spectral
    region expansion.

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing the spectral data and fitted models.
    compID : int or str
        The component ID or number of components in the model to evaluate. If `compID` is 0,
        the model is assumed to be a flat baseline (`y = 0`).
    reduced : bool, optional
        Whether to compute the reduced chi-squared value by normalizing with the degrees of freedom.
        If False, computes the standard chi-squared value. Default is False.
    usemask : bool, optional
        Whether to apply a mask to exclude invalid or unwanted regions from the calculation. Default is False.
    mask : numpy.ndarray, optional
        A 3D boolean array specifying which voxels to include in the chi-squared calculation.
        If None and `usemask` is True, a default mask derived from the model is used.
    expand : int, optional
        Number of spectral channels to expand the mask region. Default is 20.

    Returns
    -------
    chisq : numpy.ndarray
        A 2D array representing the chi-squared (or reduced chi-squared) values
        for each spatial pixel in the cube.

    Notes
    -----
    The chi-squared statistic is calculated as:

        χ² = Σ [(data - model)² / rms²]

    where the sum is taken over spectral channels for each spatial pixel.

    If `reduced=True`, the reduced chi-squared is computed by dividing χ² by
    the degrees of freedom, which is the number of samples minus the number of model parameters.

    Raises
    ------
    ValueError
        If the dimensions of the data cube and model cube do not match.
    """
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


def calc_AICc(ucube, compID, mask, planemask=None, return_NSamp=True, expand=20):
    """
    Calculate the corrected Akaike Information Criterion (AICc) for a spectral cube model.

    This function computes the AICc for a given model component by evaluating
    the residual sum of squares (RSS) and the effective sample size. The AICc
    values help compare the goodness-of-fit for models with varying complexities.

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing the spectral data, fitted models,
        and associated parameters.
    compID : int or str
        The component ID or number of components in the model. If an integer,
        it is used to calculate the number of parameters as `n_parameters = compID * 4`.
    mask : numpy.ndarray
        A 3D boolean array specifying which voxels (volume pixels) to include in the AICc calculation.
    planemask : numpy.ndarray, optional
        A 2D spatial boolean array specifying which pixels to calculate AICc for. If provided,
        calculations are restricted to these pixels. Default is None.
    return_NSamp : bool, optional
        If True, return the effective sample size along with the AICc values. Default is True.
    expand : int, optional
        Number of spectral channels to expand the region where the RSS is calculated.
        Default is 20.

    Returns
    -------
    AICc_map : numpy.ndarray
        A 2D array containing the computed AICc values for the specified regions.
    NSamp_map : numpy.ndarray, optional
        A 2D array containing the effective sample size for each spatial pixel.
        Only returned if `return_NSamp` is True.

    Notes
    -----
    The corrected Akaike Information Criterion (AICc) is a modification of the AIC
    that accounts for finite sample sizes. It is given by:

        AICc = AIC + (2k(k + 1)) / (N - k - 1)

    where `k` is the number of parameters and `N` is the sample size.

    Raises
    ------
    ValueError
        If the input `ucube` does not contain valid models or residuals for the specified `compID`.
    """
    if isinstance(compID, int):
        p = compID * 4
        compID = str(compID)
    elif isinstance(compID, str):
        p = int(compID) * 4

    if compID == '0':
        # the zero component model is just a y = 0 baseline
        modcube = np.zeros(cube.shape)
    else:
        modcube = ucube.pcubes[compID].get_modelcube(update=False, multicore=ucube.n_cores)

    # get the rss value and sample size
    rss_map, NSamp_map = get_rss(ucube.cube, modcube, expand=expand, usemask=True, mask=mask,
                                 return_size=True, return_mask=False, planemask=planemask)
    AICc_map = aic.AICc(rss=rss_map, p=p, N=NSamp_map)

    if return_NSamp:
        return AICc_map, NSamp_map
    else:
        return AICc_map


def calc_AICc_likelihood(ucube, ncomp_A, ncomp_B, ucube_B=None, multicore=True, expand=0, planemask=None):
    """
    Calculate the relative likelihood of two models based on their AICc values.

    This function computes the logarithmic relative likelihood of model A
    compared to model B, given their corrected Akaike Information Criterion (AICc)
    values. It can optionally use a second `UltraCube` instance for comparisons.

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing the spectral data,
        fitted models, and associated parameters for model A.
    ncomp_A : int
        Number of components in model A.
    ncomp_B : int
        Number of components in model B.
    ucube_B : UltraCube, optional
        An optional second `UltraCube` instance for model B. If provided,
        AICc values for both models are calculated independently, and a
        common mask is used. Default is None.
    multicore : bool, optional
        Whether to enable parallel processing. Default is True.
    expand : int, optional
        Number of spectral channels to expand the region where AICc values
        are calculated. Default is 0.
    planemask : numpy.ndarray, optional
        A 2D spatial boolean array specifying which pixels to calculate the
        relative likelihood for. If None, all pixels are considered. Default is None.

    Returns
    -------
    lnk : numpy.ndarray
        A 2D array containing the logarithmic relative likelihood of model A
        compared to model B.

    Notes
    -----
    - The likelihood is derived from the difference in AICc values:

        .. math::

            \ln(\mathcal{L}_A / \mathcal{L}_B) = (AICc_B - AICc_A) / 2

      where :math:`\mathcal{L}_A` and :math:`\mathcal{L}_B` are the likelihoods of models A and B.
    - If `ucube_B` is provided, the models are compared over their common mask, and the AICc values
      are recalculated fresh.
    - This function handles mismatches in sample size (`NSamp`) by resetting and updating model masks.
    - Currently, the expand argument is only used if ucube_B is provided

    Raises
    ------
    ValueError
        If the required AICc values for the models cannot be calculated due to missing data or invalid masks.

    """
    if not ucube_B is None:
        # if a second UCube is provide for model comparison, use their common mask and calculate AICc values
        # without storing/updating them in the UCubes
        # reset model masks first
        ucube.reset_model_mask(ncomps=[ncomp_A], multicore=multicore)
        ucube_B.reset_model_mask(ncomps=[ncomp_B], multicore=multicore)

        mask = np.logical_or(ucube.master_model_mask, ucube_B.master_model_mask)
        AICc_A = calc_AICc(ucube, compID=ncomp_A, mask=mask, planemask=planemask, return_NSamp=False, expand=expand)
        AICc_B = calc_AICc(ucube_B, compID=ncomp_B, mask=mask, planemask=planemask, return_NSamp=False, expand=expand)

        return aic.likelihood(AICc_A, AICc_B)

    if not str(ncomp_A) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_A)

    if not str(ncomp_B) in ucube.NSamp_maps:
        ucube.get_AICc(ncomp_B)

    NSamp_mapA = ucube.NSamp_maps[str(ncomp_A)]
    NSamp_mapB = ucube.NSamp_maps[str(ncomp_B)]

    if not np.array_equal(NSamp_mapA, NSamp_mapB, equal_nan=True):
        logger.warning("Number of samples do not match. Recalculating AICc values")
        #reset the master component mask first
        ucube.reset_model_mask(ncomps=[ncomp_A, ncomp_B], multicore=multicore)

        if planemask is None:
            pmask = NSamp_mapA != NSamp_mapB
        else:
            pmask = np.logical_and(planemask, NSamp_mapA != NSamp_mapB)
        ucube.get_AICc(ncomp_A, update=True, planemask=pmask)
        ucube.get_AICc(ncomp_B, update=True, planemask=pmask)

    gc.collect()

    if planemask is None:
        lnk = aic.likelihood(ucube.AICc_maps[str(ncomp_A)], ucube.AICc_maps[str(ncomp_B)])
    else:
        lnk = aic.likelihood(ucube.AICc_maps[str(ncomp_A)][planemask], ucube.AICc_maps[str(ncomp_B)][planemask])
    return lnk

def get_all_lnk_maps(ucube, ncomp_max=2, rest_model_mask=True, multicore=True):
    """
    Compute log-likelihood ratio maps for model comparisons up to a specified number of components.

    This function calculates log-likelihood ratio (lnk) maps for comparing spectral
    models with different numbers of components, based on the Akaike Information
    Criterion corrected for finite sample sizes (AICc).

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing the spectral data and fitted models.
    ncomp_max : int, optional
        The maximum number of components to include in the model comparison.
        Default is 2.
    rest_model_mask : bool, optional
        If True, resets and updates the master model mask in `ucube` for components
        being compared. Default is True.
    multicore : bool, optional
        Whether to enable parallel processing for calculations. Default is True.

    Returns
    -------
    lnk_maps : tuple of numpy.ndarray
        Log-likelihood ratio maps for the model comparisons. The returned maps include:
        - `lnk10`: Comparison between 1-component and 0-component models.
        - `lnk20` (if `ncomp_max >= 2`): Comparison between 2-component and 0-component models.
        - `lnk21` (if `ncomp_max >= 2`): Comparison between 2-component and 1-component models.

    Notes
    -----
    - Log-likelihood ratios are calculated using AICc values for each model as:

        .. math::

            \ln(\mathcal{L}_A / \mathcal{L}_B) = (AICc_B - AICc_A) / 2

      where :math:`\mathcal{L}_A` and :math:`\mathcal{L}_B` are the likelihoods of
      models A and B, respectively.
    - If `ncomp_max` is greater than 2, the function does not compute higher-order
      comparisons and simply returns the log-likelihood maps for up to 2 components.

    Raises
    ------
    ValueError
        If `ncomp_max` is less than 1 or if model data for the required number of
        components is missing in `ucube`.
    """
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
    """
    Select the best 2-component parameter cube based on AICc likelihood thresholds.

    This function identifies regions in the parameter cube where the 2-component model
    is justified over simpler models (0- or 1-component) based on log-likelihood ratio
    thresholds calculated from the AICc. It can also combine results with a 1-component
    model for regions where 2-component fits are not justified.

    Parameters
    ----------
    ucube : UltraCube
        An instance of the `UltraCube` class containing spectral data and model fits.
    multicore : bool, optional
        Whether to enable parallel processing for calculations. Default is True.
    lnk21_thres : float, optional
        Threshold for the log-likelihood ratio between 2-component and 1-component models.
        Default is 5.
    lnk20_thres : float, optional
        Threshold for the log-likelihood ratio between 2-component and 0-component models.
        Default is 5.
    lnk10_thres : float, optional
        Threshold for the log-likelihood ratio between 1-component and 0-component models.
        Default is 5.
    return_lnks : bool, optional
        If True, return the log-likelihood ratio maps (lnk10, lnk20, lnk21) along with
        the parameter and error cubes. Default is True.
    include_1c : bool, optional
        If True, include 1-component model results in regions where the 2-component model
        is not justified. Default is True.

    Returns
    -------
    tuple
        If `return_lnks` is True:
            - parcube : numpy.ndarray
                A 3D array representing the best-fit parameter cube for the justified
                model in each spatial region.
            - errcube : numpy.ndarray
                A 3D array representing the associated errors for the parameters.
            - lnk10 : numpy.ndarray
                Log-likelihood ratio map comparing 1-component and 0-component models.
            - lnk20 : numpy.ndarray
                Log-likelihood ratio map comparing 2-component and 0-component models.
            - lnk21 : numpy.ndarray
                Log-likelihood ratio map comparing 2-component and 1-component models.
        If `return_lnks` is False:
            - parcube : numpy.ndarray
            - errcube : numpy.ndarray

    Notes
    -----
    - The function determines the best model for each spatial region by comparing
      log-likelihood ratios with the specified thresholds.
    - Regions where the 2-component model is not justified (based on `lnk21_thres`
      and `lnk20_thres`) can revert to the 1-component model if `include_1c` is True.
    - Any region failing to meet the `lnk10_thres` threshold for the 1-component model
      is set to NaN in the parameter and error cubes.

    Raises
    ------
    ValueError
        If model fits for 1-component or 2-component models are missing in `ucube`.

    """
    lnk10, lnk20, lnk21 = get_all_lnk_maps(ucube, ncomp_max=2, multicore=multicore)

    parcube = copy(ucube.pcubes['2'].parcube)
    errcube = copy(ucube.pcubes['2'].errcube)

    mask = np.logical_and(lnk21 > lnk21_thres, lnk20 > lnk20_thres)

    val_fill = np.nan #the value to fill outside of mask

    if include_1c:
        parcube[:4, ~mask] = copy(ucube.pcubes['1'].parcube[:4, ~mask])
        errcube[:4, ~mask] = copy(ucube.pcubes['1'].errcube[:4, ~mask])
        parcube[4:8, ~mask] = val_fill
        errcube[4:8, ~mask] = val_fill

    else:
        parcube[:, ~mask] = val_fill
        errcube[:, ~mask] = val_fill

    mask = lnk10 > lnk10_thres
    parcube[:, ~mask] = val_fill
    errcube[:, ~mask] = val_fill

    if return_lnks:
        # set lnk pixels with no fit to nan
        mask = ucube.has_fit(ncomp=1)
        lnk10[~mask] = val_fill
        mask = ucube.has_fit(ncomp=2)
        lnk20[~mask] = val_fill
        lnk21[~mask] = val_fill

        return parcube, errcube, lnk10, lnk20, lnk21
    else:
        return parcube, errcube

#======================================================================================================================#
# statistics tools

def get_rss(cube, model, expand=20, usemask=True, mask=None, return_size=True, return_mask=False,
            include_nosamp=True, planemask=None):
    """
    Calculate the residual sum of squares (RSS) for a spectral cube model fit.

    Parameters
    ----------
    cube : SpectralCube
        The spectral data cube to analyze.
    model : numpy.ndarray
        The model array to compare against the data cube.
    expand : int, optional
        Number of channels to expand the region where the RSS is calculated along the spectral dimension. Default is 20.
    usemask : bool, optional
        Whether to apply a mask during RSS calculation. Default is True.
    mask : numpy.ndarray, optional
        A boolean array specifying which elements to include in the calculation. If None, a default mask based on the model is used.
    return_size : bool, optional
        If True, return the size of the valid sample region used in the RSS calculation. Default is True.
    return_mask : bool, optional
        If True, return the final mask used during RSS computation. Default is False.
    include_nosamp : bool, optional
        Whether to include spectral regions with no sample data by filling gaps with a default mask. Default is True.
    planemask : numpy.ndarray, optional
        A 2D mask specifying where to calculate RSS for optimized computation. Default is None.

    Returns
    -------
    tuple
        The calculated RSS values. If `return_size` or `return_mask` is True, additional elements include:
        - Sample size per pixel (if `return_size` is True).
        - Final mask used (if `return_mask` is True).
    """
    if usemask:
        if mask is None:
            # may want to change this for future models that includes absorptions
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    if isinstance(mask, da.Array):
        mask = mask.persist()

    if include_nosamp:
        # if there no mask in a given pixel, fill it in with combined spectral mask
        if isinstance(mask, da.Array):
            nsamp_map = np.nansum(mask, axis=0).compute()
        else:
            nsamp_map = np.nansum(mask, axis=0)
        mm = nsamp_map <= 0
        try:
            max_y, max_x = np.where(nsamp_map == np.nanmax(nsamp_map))
            spec_mask_fill = copy(mask[:, max_y[0], max_x[0]])
        except:
            spec_mask_fill = np.any(mask, axis=(1, 2))

        # Handle dask compatibility by computing the mask if necessary
        if isinstance(mask, da.Array):
            mask = mask.compute()

        # Apply the mask update with numpy-compatible indexing
        mask[:, mm] = spec_mask_fill[:, np.newaxis]

        # Convert back to dask if needed
        if isinstance(cube._data, da.Array):
            mask = da.from_array(mask, chunks=cube._data.chunksize)

    # assume flat-baseline model even if no model exists
    if isinstance(model, da.Array):
        model = da.where(da.isnan(model), 0, model)
    else:
        model[np.isnan(model)] = 0

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    if expand > 0:
        mask = expand_mask(mask, expand)

    if planemask is None:
        residual = get_residual(cube, model)
        residual = da.from_array(residual) if not isinstance(cube._data, da.Array) else residual
    else:
        # Get residual with planemask applied
        residual = get_residual(cube, model, planemask=planemask)

        # Handle mask based on its type
        if isinstance(mask, da.Array):
            mask = dask_ops.apply_planemask(mask, planemask, persist=True)
        else:
            mask = mask[:, planemask]

        # Check for shape mismatch
        if mask.shape != residual.shape:
            msg = (
                f"mask with shape {mask.shape} does not match the residual shape {residual.shape}. "
                f"This shouldn't happen; there may be a bug in the code."
            )
            logger.error(msg)  # Use logger.error for unexpected critical issues
            raise ValueError(msg)

    # Use da.where for masking instead of residual * mask
    masked_residual = da.where(mask, residual, 0) if isinstance(residual, da.Array) else np.where(mask, residual, 0)

    # Calculate RSS
    rss = da.nansum(masked_residual ** 2, axis=0) if isinstance(masked_residual, da.Array) else np.nansum(
        masked_residual ** 2, axis=0)
    rss = da.where(rss == 0, np.nan, rss) if isinstance(rss, da.Array) else np.where(rss == 0, np.nan, rss)

    returns = (rss.compute() if isinstance(rss, da.Array) else rss,)

    if return_size:
        nsamp = da.nansum(mask, axis=0) if isinstance(mask, da.Array) else np.nansum(mask, axis=0)
        nsamp = nsamp.astype(np.float64)
        nsamp[np.isnan(rss)] = np.nan
        returns += (nsamp.compute() if isinstance(nsamp, da.Array) else nsamp,)
    if return_mask:
        returns += (mask.compute() if isinstance(mask, da.Array) else mask,)

    return returns


def get_chisq(cube, model, expand=20, reduced=True, usemask=True, mask=None):
    """
    Calculate the chi-squared or reduced chi-squared value for a spectral cube.

    This function computes the chi-squared goodness-of-fit statistic by comparing
    the observed data in the cube with a provided model. Optionally, the calculation
    can be restricted to masked regions and expanded spectral regions.

    Parameters
    ----------
    cube : SpectralCube
        The observed spectral cube containing the data to compare against the model.
    model : numpy.ndarray
        A 3D array representing the model cube, which must have the same shape as the input cube.
    expand : int, optional
        Number of spectral channels to expand the mask region. Default is 20.
    reduced : bool, optional
        If True, compute the reduced chi-squared value by normalizing with the degrees of freedom.
        If False, compute the standard chi-squared value. Default is True.
    usemask : bool, optional
        If True, apply a mask to exclude invalid regions from the calculation. If no mask is provided,
        regions where the model is zero are excluded. Default is True.
    mask : numpy.ndarray, optional
        A 3D boolean array specifying regions to include in the calculation. If None,
        a mask is automatically derived from the model.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the chi-squared (or reduced chi-squared) value
        at each spatial pixel in the cube.

    Notes
    -----
    - The `expand` parameter allows users to extend the region of interest by a
      specified number of spectral channels around the model.
    - The reduced chi-squared value is normalized by the number of degrees of freedom.
    - The residuals between the cube and model are weighted using the mask.

    Raises
    ------
    ValueError
        If the cube and model dimensions do not match.

    Examples
    --------
    >>> from spectral_cube import SpectralCube
    >>> import numpy as np
    >>> cube = SpectralCube.read("example_cube.fits")
    >>> model = np.random.random(cube.shape)
    >>> chisq = get_chisq(cube, model, expand=10, reduced=True)
    >>> print(chisq)
    """
    if usemask:
        if mask is None:
            mask = model > 0
    else:
        mask = ~np.isnan(model)

    if isinstance(mask, da.Array):
        mask = mask.persist()

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

    if isinstance(chisq, da.Array):
        chisq = chisq.compute()

    gc.collect()

    if reduced:
        # return the reduce chi-squares values
        return chisq
    else:
        # return the ch-squared values and the number of data points used
        if isinstance(mask, da.Array):
            return chisq, np.nansum(mask, axis=0).compute()
        else:
            return chisq, np.nansum(mask, axis=0)


def get_masked_moment(cube, model, order=0, expand=10, mask=None):
    """
    Calculate a masked moment of a spectral cube.

    This function generates a moment map (e.g., integrated intensity, centroid)
    for the input spectral cube, applying a mask derived from the provided model.

    Parameters
    ----------
    cube : spectral_cube.SpectralCube
        The spectral cube containing the observed data, from which the moment
        is calculated. This must be an instance of the `SpectralCube` class
        from the `spectral_cube` package.
    model : dask.array.Array
        A 3D Dask array representing the model cube, used to derive the mask.
        The dimensions of the array must match those of the cube.
    order : int, optional
        The order of the moment to compute:
        - `0` for integrated intensity (default),
        - `1` for centroid,
        - `2` for line width.
    expand : int, optional
        Currently not supported. May be deprecated in future versions. Default is 10.
    mask : numpy.ndarray, optional
        A 3D boolean array specifying which elements to include in the moment
        calculation. If provided, it is combined with the mask derived from
        the model. Default is None.

    Returns
    -------
    astropy.units.Quantity
        The computed moment map, with the unit determined by the input cube
        and the moment order.

    Notes
    -----
    - The mask is generated by identifying non-zero elements in the model and
      excluding regions with low signal-to-noise ratios using a threshold of
      0.1 times the residual RMS.
    - The `expand` parameter is currently not supported and may be deprecated
      in a future release.
    - Pixels with low signal-to-noise ratios are excluded from the moment map
      based on the mask.
    - The function uses the spectral axis of the cube for moment calculations.

    Raises
    ------
    ValueError
        If the cube and model dimensions do not match.

    Examples
    --------
    >>> from spectral_cube import SpectralCube
    >>> cube = SpectralCube.read("example_cube.fits")
    >>> model = np.random.random(cube.shape)
    >>> moment_map = get_masked_moment(cube, model, order=0)
    >>> moment_map.write("masked_moment.fits")
    """
    # If no mask is provided, create one based on the model
    if mask is None:
        residual = get_residual(cube, model)
        rms = mad_std(residual, axis=0)
        mask = model > 0.1*rms
    else:
        # Ensure mask is updated with finite model values
        mask = np.logical_and(mask, np.isfinite(model))

    # Apply the mask to the cube and return the moment
    maskcube = cube.with_mask(mask.compute().astype(bool))
    maskcube = maskcube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    mom = maskcube.moment(order=order)
    return mom


def expand_mask(mask, expand):
    """
    Expand a 3D mask along the spectral axis by a specified buffer size.

    This function applies a binary dilation to the input mask, increasing its coverage
    along the spectral axis by the specified `expand` value. The expansion is applied
    independently for each spatial pixel.

    Parameters
    ----------
    mask : numpy.ndarray
        A 3D boolean array representing the original mask. The first dimension corresponds
        to the spectral axis, while the remaining dimensions represent spatial axes.
    expand : int
        The number of spectral channels to extend the mask along the spectral axis.
        Must be a non-negative integer.

    Returns
    -------
    expanded_mask : numpy.ndarray
        A 3D boolean array of the same shape as the input mask, with the spectral
        regions expanded by the specified buffer size.

    Notes
    -----
    The binary dilation process enlarges the `True` regions in the mask along the
    spectral axis. The dilation does not affect spatial axes, but it creates a buffer
    around the original mask in the spectral dimension.

    Raises
    ------
    ValueError
        If `expand` is not a non-negative integer.
    TypeError
        If the input `mask` is not a boolean numpy array.

    """
    selem = np.ones(expand,dtype=bool)
    selem.shape += (1,1,)
    if isinstance(mask, da.Array):
        mask = dask_ops.dask_binary_dilation(mask, selem)
    else:
        mask = nd.binary_dilation(mask, selem)
    return mask


def get_rms(residual):
    """
    Compute a robust estimate of the root mean square (RMS) from the fit residuals.

    This function calculates the RMS of the residuals using a robust method based on the
    median absolute deviation (MAD). It is less sensitive to outliers compared to a
    standard RMS calculation.

    Parameters
    ----------
    residual : numpy.ndarray
        A 3D array representing the residuals between the observed data and the fitted model.
        The first dimension corresponds to the spectral axis, while the remaining dimensions
        represent the spatial axes.

    Returns
    -------
    rms : numpy.ndarray
        A 2D array representing the RMS for each spatial pixel in the residual cube.

    Notes
    -----
    The RMS is calculated using the following robust formula:

    RMS = 1.4826 × median(abs(x - median(x))) / sqrt(2)

    Here, `x` represents the residuals for each spectral channel at a spatial pixel. The
    subtraction and absolute value are applied element-wise along the spectral axis. The robust
    method reduces sensitivity to noise spikes or outliers in the residuals.

    Raises
    ------
    ValueError
        If the input `residual` is not a 3D array or if it contains invalid (e.g., NaN) values.

    """
    diff = residual - np.roll(residual, 2, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(diff), axis=0) / 2**0.5
    gc.collect()
    return rms


def get_residual(cube, model, planemask=None):
    """
    Calculate the residual between the data cube and the model cube.

    The residual is calculated as the difference between the spectral data
    in the cube and the corresponding model values. Optionally, a 2D
    spatial mask (`planemask`) can be applied to restrict the calculation
    to specific spatial regions.

    Parameters
    ----------
    cube : SpectralCube
        A spectral cube object containing the observed data. It can be a
        dask-enabled cube or a standard numpy-based cube.
    model : numpy.ndarray
        A 3D array representing the model cube, where the dimensions match
        those of the data cube (spectral axis as the first dimension).
    planemask : numpy.ndarray, optional
        A 2D boolean array specifying the spatial pixels for which residuals
        should be calculated. If provided, only these regions will be used
        in the calculation. Default is None.

    Returns
    -------
    numpy.ndarray or dask.array.Array
        The residual array, calculated as the difference between the cube's
        data and the model values. The returned type matches the cube's data
        type (e.g., dask array if the cube uses dask, or numpy array otherwise).

    Notes
    -----
    - If `planemask` is provided, the residuals are calculated only for the
      specified spatial pixels, which can reduce computation time.
    - Handles memory-efficient computation when dask is enabled.
    """
    # Get the cube data as a Dask or NumPy array
    data = cube._data

    if planemask is None:
        residual = data - model
    else:
        # only operate on the masked pixels
        logger.debug(f"==== computing 2D-masked residual ====")
        logger.debug(f"planemask type:{type(planemask)}; planemask shape:{planemask.shape}")
        logger.debug(f"model type:{type(model)}; data type:{type(data)}")
        logger.debug(f"model shape:{model.shape}; data shape:{data.shape}")
        # mask model
        if isinstance(model, da.Array):
            model_masked = dask_ops.apply_planemask(model, planemask, persist=False)
        elif isinstance(model, np.ndarray):
            model_masked = model[:, planemask]
        else:
            msg = (f"The model type is {type(model)}. This shouldn't happen; there may be a bug in the code.")
            logger.error(msg)  # Use logger.error for unexpected critical issues
            raise TypeError(msg)

        # mask data
        if isinstance(data, da.Array):
            data_masked = dask_ops.apply_planemask(data, planemask, persist=False)
        elif isinstance(data, np.ndarray):
            data_masked = data[:, planemask]
        else:
            msg = (f"The data type is {type(data)}. This shouldn't happen; there may be a bug in the code.")
            logger.error(msg)  # Use logger.error for unexpected critical issues
            raise TypeError(msg)

        residual = data_masked - model_masked

        '''
        if isinstance(model, da.Array) and isinstance(data, da.Array):
            data_masked = dask_ops.apply_planemask(data, planemask, persist=False)
            model_masked = dask_ops.apply_planemask(model, planemask, persist=False)
            residual = data_masked - model_masked

        elif isinstance(model, np.ndarray) and isinstance(data, np.ndarray):
            residual = data[:, planemask] - model[:, model]

        else:
            # Handle NumPy or mixed array cases
            logger.debug(f"==== computing mixed masked residual ====")
            logger.debug(f"model type:{type(model)}; data type:{type(data)}")
            logger.debug(f"model shape:{type(model.shape)}; data shape:{type(data.shape)}")

            if isinstance(model, da.Array):
                model = dask_ops.apply_planemask(model, planemask, persist=False)

            if isinstance(data, da.Array):
                data = dask_ops.apply_planemask(data, planemask, persist=False)
                
            residual = (data - model)
        
            
            #planemask_broadcast = da.broadcast_to(planemask, data.shape[1:])
            #data_masked = da.where(planemask_broadcast, data, 0)
            #model_masked = da.where(planemask_broadcast, model, 0)
            #residual = data_masked - model_masked

            # Flatten the residual spatial dimensions and the planemask
            #residual_flat = residual.reshape(residual.shape[0], -1)
            #planemask_flat = planemask.ravel()  # Flatten planemask

            # Apply the mask using da.compress
            #residual = da.compress(planemask_flat, residual_flat, axis=1)
        '''

    if isinstance(residual, da.Array): # may be redundant, but better safe than sorry
        residual = dask_utils.persist_and_clean(residual, debug=False, visualize_filename=None)
        residual.compute()

    # Run garbage collection for memory management
    gc.collect()
    return residual


def get_Tpeak(model):
    """
    Calculate the peak value of a model cube at each spatial pixel.

    Parameters
    ----------
    model : numpy.ndarray
        The input model cube with spectral data. The first dimension is assumed
        to represent the spectral axis, and subsequent dimensions represent spatial pixels.

    Returns
    -------
    numpy.ndarray
        A 2D array where each element corresponds to the peak value of the model
        cube along the spectral axis for each spatial pixel.
    """
    if isinstance(model, da.Array):
        return np.nanmax(model, axis=0).compute()
    else:
        return np.nanmax(model, axis=0)


def is_K(data_unit):
    """
    Check if a given unit is equivalent to Kelvin (K).

    Parameters
    ----------
    data_unit : astropy.units.Unit or str
        The unit to be checked. It can be an `astropy.units.Unit` instance or a string
        representation of the unit.

    Returns
    -------
    bool
        True if the provided unit is equivalent to Kelvin (K), otherwise False.

    Examples
    --------
    >>> from astropy import units as u
    >>> is_K(u.K)
    True
    >>> is_K('K')
    True
    >>> is_K(u.Jy)
    False
    """
    return data_unit == 'K' or data_unit == u.K


def to_K(cube):
    """
    Convert the unit of a spectral cube to Kelvin (K).

    This function attempts to convert the unit of a `spectral_cube.SpectralCube` object
    to brightness temperature (K) using the Rayleigh-Jeans approximation. If the cube's
    current unit is not convertible, it will handle the error by either warning the user
    or raising a `UnitConversionError`.

    Parameters
    ----------
    cube : spectral_cube.SpectralCube
        A `SpectralCube` object whose unit needs to be converted to Kelvin (K).

    Returns
    -------
    spectral_cube.SpectralCube
        A new `SpectralCube` object with its unit set to Kelvin (K).

    Raises
    ------
    UnitConversionError
        If the cube's unit cannot be converted to Kelvin (K) and it is not possible to
        forcefully assign a unit.

    Notes
    -----
    If the cube does not have an assigned unit, a unit of Kelvin (K) will be assumed and
    a warning will be issued.

    Examples
    --------
    >>> from spectral_cube import SpectralCube
    >>> import astropy.units as u
    >>> cube = SpectralCube.read('example_cube.fits')
    >>> cube_in_K = to_K(cube)
    """
    # Decorator to handle UnitConversionError
    def handle_unit_conversion_error(func):
        @wraps(func)
        def wrapper(cube, *args, **kwargs):
            try:
                return func(cube, *args, **kwargs)
            except UnitConversionError:
                if hasattr(cube, 'unit'):
                    if cube.unit is None or cube.unit == '':
                        logger.warning("The cube does not have a unit. A unit of K will be assumed.")
                    else:
                        raise UnitConversionError(f"Cube's unit ({cube.unit}) is not convertible to K")
                else:
                    logger.warning("The cube does not have a unit. A unit of K will be assumed.")
                cube._unit = u.K
                return cube

        return wrapper

    # Decorator to handle ValueError and retry the conversion
    def handle_value_error(func):
        @wraps(func)
        def wrapper(cube, *args, **kwargs):
            try:
                return func(cube, *args, **kwargs)
            except ValueError:
                # Allow the entire cube to be loaded temporarily if the cube is too large
                logger.warning("ValueError encountered, temporarily allowing huge operations.")
                cube.allow_huge_operations = True
                result = func(cube, *args, **kwargs)
                gc.collect()  # Garbage collection to free up memory
                cube.allow_huge_operations = False
                return result

        return wrapper

    # Function to convert the cube to units of K
    @handle_value_error
    @handle_unit_conversion_error
    def convert_to_kelvin(cube):
        # Only convert if the cube does not already have a unit of K
        if cube.unit != u.K:
            cube = cube.to(u.K)
        return cube

    return convert_to_kelvin(cube)

