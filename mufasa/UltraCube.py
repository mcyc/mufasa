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

import dask.array as da

from . import aic
from . import multi_v_fit as mvf
from . import convolve_tools as cnvtool
from .spec_models import meta_model
from .utils.multicore import validate_n_cores
from .visualization.spec_viz import Plotter
#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

class UltraCube(object):
    """
    A framework to manage and fit multi-component spectral models for spectral cubes.

    This class provides tools to handle:
    - Loading and managing spectral cubes,
    - Fitting spectral models with multiple components,
    - Generating diagnostic and visualization tools for fits,
    - Calculating residuals, chi-squared, and AICc metrics.

    Attributes
    ----------
    pcubes : dict
        Holds pyspeckit cubes for each model fit.
    residual_cubes : dict
        Stores residual cubes for each component fitted.
    rms_maps : dict
        Root-mean-square (RMS) maps of residuals.
    Tpeak_maps : dict
        Peak temperature maps from fitted models.
    chisq_maps : dict
        Chi-squared maps for fitted models.
    rchisq_maps : dict
        Reduced chi-squared maps.
    rss_maps : dict
        Residual sum of squares maps.
    NSamp_maps : dict
        Number of samples per pixel.
    AICc_maps : dict
        Corrected Akaike Information Criterion maps for model evaluation.
    master_model_mask : np.ndarray, optional
        Boolean mask indicating regions with fitted models.
    snr_min : float
        Minimum signal-to-noise ratio for attempting fits.
    cnv_factor : int
        Convolution factor for spatial resolution adjustments.
    n_cores : int
        Number of CPU cores used for parallel processing.
    fittype : str
        Spectral model type used for fitting.
    plotter : Plotter
        Visualization tool for fits.
    meta_model : MetaModel
        Metadata handler for spectral models.
    """

    def __init__(self, cubefile=None, cube=None, fittype=None, snr_min=None, rmsfile=None, cnv_factor=2, n_cores=True):
        """
        Initialize the UltraCube object.

        Parameters
        ----------
        cubefile : str, optional
            Path to the .fits cube file. If None, the cube parameter is used.
        cube : SpectralCube, optional
            A spectral cube object to initialize the UltraCube.
        fittype : str, optional
            Type of spectral model to be fitted.
        snr_min : float, optional
            Minimum peak signal-to-noise ratio for attempting fits.
        rmsfile : str, optional
            Path to the file containing RMS values for the cube.
        cnv_factor : int, optional
            Factor by which to spatially convolve the cube. Default is 2.
        n_cores : int, optional
            Number of CPU cores to use for computation. Default is all available cores minus one.

        Raises
        ------
        ValueError
            If both `cubefile` and `cube` are None.
        """

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
        return mvf.make_header(ndim=2, ref_header=self.cube.header)

    def load_pcube(self, pcube_ref=None):
        """
        Load a SpectralCube as a pyspeckit.SpectralCube object.

        Parameters
        ----------
        pcube_ref : pyspeckit.SpectralCube, optional
            An existing pyspeckit cube object. If provided, the loaded cube's
            data pointer will be linked to this reference cube to conserve memory.

        Returns
        -------
        pyspeckit.SpectralCube
            The loaded pyspeckit cube for spectral model fitting.

        Notes
        -----
        - The cube is first loaded as a SpectralCube to ensure proper unit
          conversions.
        - Memory is conserved when `pcube_ref` is provided by linking data pointers.
        - The method handles rest frequency assignment and velocity convention
          updates.

        Raises
        ------
        ValueError
            If the cube file cannot be loaded or its units are incompatible.
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


    def load_pcube(self, pcube_ref=None):
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

        pcube = pyspeckit.Cube(cube=cube_temp)

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
            self.master_model_mask = np.logical_or(self.master_model_mask, mask)

    def reset_model_mask(self, ncomps, multicore=True):
        #reset and re-generate master_model_mask for all the components in ncomps
        self.master_model_mask = None

        for nc in ncomps:
            if nc > 0 and hasattr(self.pcubes[str(nc)],'parcube'):
                # update model mask if any fit has been performed
                mod_mask = self.pcubes[str(nc)]._modelcube > 0
                self.include_model_mask(mod_mask)
            gc.collect()


    def save_fit(self, savename, ncomp, header_note=None):
        # note, this implementation currently relies on
        if hasattr(self.pcubes[str(ncomp)], 'parcube'):
            save_fit(self.pcubes[str(ncomp)], savename, ncomp, header_note=header_note)
        else:
            logger.warning("no fit was performed and thus no file will be saved")


    def load_model_fit(self, filename, ncomp, calc_model=True, multicore=None):
        self.pcubes[str(ncomp)] = load_model_fit(self.cubefile, filename, ncomp, self.fittype)
        if calc_model:
            if multicore is None: multicore = self.n_cores
            # update model mask
            mod_mask = self.pcubes[str(ncomp)].get_modelcube(multicore=multicore) > 0
            logger.debug("{}comp model mask size: {}".format(ncomp, np.sum(mod_mask)) )
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

    Attributes
    ----------
    paraNameRoot : str
        Root name for parameter map files. Derived from the cube filename if not provided.
    paraDir : str
        Directory where parameter maps are stored. Created if it does not exist.
    paraPaths : dict
        Dictionary mapping component numbers to file paths for saved parameter maps.

    Methods
    -------
    read_model_fit(ncomps, read_conv=False, **kwargs)
        Load model fits from disk or perform fitting if files do not exist.
    get_model_fit(ncomp, update=True, **kwargs)
        Load or perform model fitting for the specified number of components.
    """

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



def load_model_fit(cube, filename, ncomp, fittype):
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

    #pcube = pyspeckit.Cube(cube=cube)
    pcube = pyspeckit.Cube(cube)

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
        modcube = ucube.pcubes[compID].get_modelcube(update=False, multicore=ucube.n_cores)

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


def calc_AICc(ucube, compID, mask, planemask=None, return_NSamp=True, expand=20):
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
    # return the log likelihood of the A model relative to the B model
    # currently, expand is only used if ucube_B is provided

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
        # set lnk pixels with no fit to nan
        mask = ucube.has_fit(ncomp=1)
        lnk10[~mask] = np.nan
        mask = ucube.has_fit(ncomp=2)
        lnk20[~mask] = np.nan
        lnk21[~mask] = np.nan

        return parcube, errcube, lnk10, lnk20, lnk21
    else:
        return parcube, errcube

#======================================================================================================================#
# statistics tools

def get_rss(cube, model, expand=20, usemask=True, mask=None, return_size=True, return_mask=False,
            include_nosamp=True, planemask=None):
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
    model[np.isnan(model)] = 0

    # creating mask over region where the model is non-zero,
    # plus a buffer of size set by the expand keyword.
    if expand > 0:
        mask = expand_mask(mask, expand)
    mask = mask.astype(float)

    if planemask is None:
        residual = get_residual(cube, model)
        residual = da.from_array(residual) if isinstance(cube._data, da.Array) else residual
    else:
        residual = get_residual(cube, model, planemask=planemask)
        mask_temp = mask
        mask = mask[:, planemask]

    # note: using nan-sum may walk over some potential bad pixel cases
    rss = da.nansum((residual * mask) ** 2, axis=0) if isinstance(residual, da.Array) else np.nansum(
        (residual * mask) ** 2, axis=0)
    rss[rss == 0] = np.nan

    returns = (rss.compute() if isinstance(rss, da.Array) else rss,)

    if return_size:
        nsamp = da.nansum(mask, axis=0) if isinstance(mask, da.Array) else np.nansum(mask, axis=0)
        nsamp[np.isnan(rss)] = np.nan
        returns += (nsamp.compute() if isinstance(nsamp, da.Array) else nsamp,)
    if return_mask:
        returns += (mask.compute() if isinstance(mask, da.Array) else mask,)

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
    '''
    Calculate the residual of the fit to the cube.
    :param cube: a SpectralCube object, potentially with dask enabled
    :param model: ndarray, a model of the cube
    :param planemask: a 2D boolean mask specifying where to calculate. Using this can save computing time
    :return: residual, either as a dask array (if dask is enabled) or a numpy array
    '''
    # Get the cube data as a dask array or numpy array
    data = cube.filled_data[:].value  # dask array if dask is enabled, numpy array otherwise

    # Calculate residual with or without a planemask
    if planemask is None:
        residual = data - model
    else:
        # If dask, apply the mask in a memory-efficient way
        if isinstance(data, da.Array):
            planemask_expanded = da.from_array(planemask, chunks=data.chunksize[1:])
            residual = data[:, planemask_expanded] - model[:, planemask]
        else:
            # Non-dask (numpy array), use direct masking
            residual = data[:, planemask] - model[:, planemask]

    # If residual is a dask array, compute only if needed (e.g., for direct use in numpy context)
    if isinstance(residual, da.Array):
        residual = residual.compute()

    # Run garbage collection for memory management
    gc.collect()

    return residual


def get_Tpeak(model):
    '''
    Return the peak value of a model cube at each spatial pixel (i.e. along the spectral axis)
    :param model:
    :return:
    '''
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

