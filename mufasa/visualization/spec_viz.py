import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from pyspeckit.spectrum.units import SpectroscopicAxis
import warnings

from ..spec_models.meta_model import MetaModel

# =======================================================================================================================

class Plotter(object):
    def __init__(self, ucube, fittype, ncomp_list=None, spec_unit='km/s', bunit=None):
        """
        Initialize the Plotter class.

        Parameters
        ----------
        ucube : UltraCube
            The UltraCube object containing the spectral cube and parameter cubes.
        fittype : str
            The type of model fit to use ('nh3_multi_v' or 'n2hp_multi_v').
        ncomp_list : list of int, optional
            List of component numbers to plot. If None, all components are used.
        spec_unit : str, optional
            The spectral unit for the cube. Default is 'km/s'.
        bunit : str or Quantity, optional
            The desired unit for the cube's data (e.g., 'K' for brightness temperature or 'Jy' for flux).
        """
        self.ucube = ucube
        self.cube = self.ucube.cube.with_spectral_unit(spec_unit, velocity_convention='radio')
        self.xarr = SpectroscopicAxis(self.cube.spectral_axis.value,
                                      unit=spec_unit,
                                      refX=self.cube._header['RESTFRQ'],
                                      velocity_convention='radio')

        # Convert bunit to a Unit if provided as a string
        if isinstance(bunit, str):
            bunit = u.Unit(bunit)

        # Set cube's unit and y-axis label
        if bunit is not None:
            if hasattr(self.cube, 'unit') and (self.cube.unit is not None and self.cube.unit != ''):
                try:
                    # Attempt to convert the cube's unit to bunit
                    self.cube = self.cube.to(bunit)
                    self.ylab = f"Intensity ({bunit.to_string()})"
                except u.UnitConversionError:
                    warnings.warn(f"Incompatible units: cube's unit ({self.cube.unit}) cannot be converted to {bunit}.")
                    self.ylab = f"Intensity ({self.cube.unit.to_string()})"
            else:
                # Cube has no unit; assume bunit as the unit and set the label accordingly
                warnings.warn("Cube has no unit attribute; setting y-axis label to specified bunit.")
                self.ylab = f"Intensity ({bunit.to_string()})"
        else:
            # No bunit specified; set y label based on cube's unit if available
            if hasattr(self.cube, 'unit') and (self.cube.unit is not None and self.cube.unit != ''):
                if self.cube.unit.is_equivalent(u.K):
                    self.ylab = r"$T_{\mathrm{MB}}$ (K)"
                elif self.cube.unit.is_equivalent(u.Jy):
                    self.ylab = r"Flux Density (Jy)"
                elif self.cube.unit.is_equivalent(u.mJy):
                    self.ylab = r"Flux Density (mJy)"
                else:
                    # Generic label if unit is unknown or unsupported
                    self.ylab = f"Intensity ({self.cube.unit.to_string()})"
            else:
                # No unit in cube and no bunit specified, set to generic label
                warnings.warn("Cube data has no unit attribute and no bunit specified; setting y-axis label to 'Intensity'.")
                self.ylab = "Intensity"

        # Set the x-axis label
        self.xlab = r"$v_{\mathrm{LSR}}$ (km s$^{-1}$)"

        meta_model = MetaModel(fittype=fittype, ncomp=1) #ncomp is just a placehoder here
        self.model_func = meta_model.model_func

        # Process ncomp_list for parcubes
        self.parcubes = {}
        if ncomp_list is None:
            for key in self.ucube.pcubes:
                self.parcubes[key] = self.ucube.pcubes[key].parcube
        else:
            for n in ncomp_list:
                self.parcubes[str(n)] = self.ucube.pcubes[str(n)].parcube



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
        # add defaults that can be superseded
        kwargs = {'xlab': self.xlab, 'ylab': self.ylab, **kwargs}

        self.fig, self.axs = \
            plot_spec_grid(self.cube, x, y, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim, figsize=figsize, **kwargs)


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
        spc = self.cube[:, y, x]
        if xlab is None:
            xlab = self.xlab
        if ylab is None:
            ylab = self.ylab
        self.fig, self.axs = plot_spec(spc, xarr=self.xarr, ax=ax, xlab=xlab, ylab=ylab, **kwargs)
        return self.fig, self.axs


    def plot_fit(self, x, y, ax=None, ncomp=None, **kwargs):
        """
        Plot a model fit for a spectrum at (x, y).

        Parameters
        ----------
        x : int
            X-coordinate of the pixel.
        y : int
            Y-coordinate of the pixel.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created.
        ncomp : int
            The component number to plot.
        **kwargs : dict
            Additional keyword arguments passed to `plot_model`.
        """
        if ax is None:
            self.fig, ax = plt.subplots()
            self.axs = ax
        plot_model(self.parcubes[str(ncomp)][:, y, x], self.model_func, self.xarr, ax, ncomp, **kwargs)


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
            X-axis limits for the plot.
        ylim : tuple, optional
            Y-axis limits for the plot.
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
        # add defaults that can be superseded
        kwargs = {'xlab': self.xlab, 'ylab': self.ylab, **kwargs}

        self.fig, self.axs = \
            plot_fits_grid(self.cube, self.parcubes[str(ncomp)], self.model_func, x, y, self.xarr,
                           ncomp=ncomp, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim,
                           figsize=figsize, origin=origin, mod_all=mod_all, savename=savename,
                           **kwargs)

# =======================================================================================================================
# None-plotting functions

def get_cube_slab(cube, vZoomLims=(-5, 20)):
    """
    Extract a spectral slab from the cube over the specified velocity range.

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube to extract the slab from.
    vZoomLims : tuple of float, optional
        The velocity limits for the slab, in km/s. Default is (-5, 20).

    Returns
    -------
    cube_s : SpectralCube
        The extracted spectral slab.
    xarr : SpectroscopicAxis
        The corresponding spectroscopic axis for the slab.
    """
    # currently incomplete. it will be used to save some computation time when calculating the model and plotting
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    cube_s = cube.spectral_slab(vZoomLims[0] * u.km / u.s, vZoomLims[1] * u.km / u.s)

    # SpectroscopicAxis has the advantage of being able to performed unit conversion automatically
    xarr = SpectroscopicAxis(cube_s.spectral_axis.value, unit=cube_s.spectral_axis.unit,
                             refX=cube_s._header['RESTFRQ'], velocity_convention='radio')
    return cube_s, xarr

# =======================================================================================================================
# None-class functions

def plot_spec(spc, xarr=None, ax=None, fill=False, xlab=None, ylab=None, **kwargs):
    """
    Plot a spectrum.

    Parameters
    ----------
    spc : Spectrum
        The spectrum to plot.
    xarr : SpectroscopicAxis, optional
        The spectroscopic axis to use for the x-axis. If None, the spectrum's spectral axis is used.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    fill : bool, optional
        Whether to fill the area under the curve. Default is False.
    xlab : str, optional
        X-axis label.
    ylab : str, optional
        Y-axis label.
    **kwargs : dict
        Additional keyword arguments passed to `ax.plot` or `ax.fill_between`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object, if `ax` is None.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if 'c' in kwargs:
        if 'color' in kwargs:
            raise TypeError("Got both 'color' and 'c', which are aliases of one another")
        else:
            kwargs['color'] = kwargs['c']
            del kwargs['c']
    elif 'color' not in kwargs:
        kwargs['color'] = '0.65'

    kwargs_df = dict(lw=1)  # default kwargs
    kwargs = {**kwargs_df, **kwargs}

    return_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        return_fig = True

    if xarr is None:
        xarr = spc.spectral_axis

    if fill:
        ax.fill_between(xarr.value, 0, spc, **kwargs)
    else:
        ax.plot(xarr.value, spc, **kwargs)

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)

    if return_fig:
        return fig, ax


def get_spec_grid(size=3, xsize=None, ysize=None, figsize=None):
    """
    Create a grid of subplots for spectra.

    Parameters
    ----------
    size : int, optional
        Size of the grid (must be odd). Default is 3.
    xsize : int, optional
        Number of columns in the grid. Default is size.
    ysize : int, optional
        Number of rows in the grid. Default is size.
    figsize : tuple, optional
        Size of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        The array of axes objects.
    """
    if size % 2 == 0 and size > 0:
        raise ValueError("Size provided must be an odd, positive integer that is not zero.")

    if xsize is None:
        xsize = size

    if ysize is None:
        ysize = size

    # initiate the plot grid
    fig, axs = plt.subplots(ysize, xsize, sharex='all', sharey='all', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=figsize)

    return fig, axs


def plot_spec_grid(cube, x, y, size=3, xsize=None, ysize=None, xlim=None, ylim=None, figsize=None,
                   origin='lower', **kwargs):
    """
    Plot a grid of spectra from the cube centered at (x, y).

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube to plot.
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
        X-axis limits for the plot.
    ylim : tuple, optional
        Y-axis limits for the plot.
    figsize : tuple, optional
        Size of the figure.
    origin : {'lower', 'upper'}, optional
        Origin of the grid. Default is 'lower'.
    **kwargs : dict
        Additional keyword arguments passed to `plot_spec`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        The array of axes objects.
    """
    # grab the x y labels and ensure it wasn't passed downstream
    if 'xlab' in kwargs:
        xlab = kwargs['xlab']
        del kwargs['xlab']

    if 'ylab' in kwargs:
        ylab = kwargs['ylab']
        del kwargs['ylab']

    fig, axs = get_spec_grid(size=size, xsize=xsize, ysize=ysize, figsize=figsize, **kwargs)
    ysize, xsize = axs.shape

    xpad = int(xsize / 2)
    ypad = int(ysize / 2)

    # Get a subcube
    scube = cube[:, y - ypad: y + ypad + 1, x - xpad: x + xpad + 1]

    if ylim is None:
        ymax = scube.max()
        ylim = (None, ymax * 1.1)

    # Ensure user-provided xlim and ylim have compatible units with data
    xlim = ensure_units_compatible(xlim, scube.spectral_axis.unit)
    ylim = ensure_units_compatible(ylim, scube.unit)

    # Strip units from xlim and ylim if they are Quantities
    xlim = strip_units(xlim)
    ylim = strip_units(ylim)

    # Existing code for plotting the spectra over the grid
    for index, ax in np.ndenumerate(axs):
        yi, xi = index
        if origin == 'lower':
            yi = ysize - 1 - yi
        elif origin != 'upper':
            raise KeyError(f"The keyword '{origin}' is unsupported.")

        spc = scube[:, yi, xi]
        plot_spec(spc, ax=ax, **kwargs)

    if origin == 'lower':
        axs = np.flip(axs, axis=0)

    for ax in axs.flat:
        ax.label_outer()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Provide a common labeling ax
    fig.add_subplot(111, frameon=False, zorder=-100)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    return fig, axs


def plot_model(para, model_func, xarr, ax, ncomp, **kwargs):
    """
    Plot a model fit for a spectrum.

    Parameters
    ----------
    para : numpy.ndarray
        The parameter array for the model.
    model_func : function
        The model function to use for the fit.
    xarr : SpectroscopicAxis
        The spectroscopic axis for the x-axis.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    ncomp : int
        The number of components in the model.
    **kwargs : dict
        Additional keyword arguments passed to `plot_spec`.
    """
    for i in range(ncomp):
        pp = para[i * 4:(i + 1) * 4]
        mod = model_func(xarr, *pp)
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.6
        plot_spec(mod, xarr, ax, fill=True, color=f'C{i}', **kwargs)

    mod_tot = model_func(xarr, *para)
    plot_spec(mod_tot, xarr, ax, c='0.1', zorder=30, **kwargs)


def plot_fits_grid(cube, para, model_func, x, y, xarr, ncomp, size=3, xsize=None, ysize=None, xlim=None, ylim=None,
                   figsize=None, origin='lower', mod_all=True, savename=None, **kwargs):
    """
    Plot a grid of model fits from the cube centered at (x, y).

    Parameters
    ----------
    cube : SpectralCube
        The spectral cube to plot.
    para : numpy.ndarray
        The parameter array for the model.
    model_func : function
        The model function to use for the fit.
    x : int
        X-coordinate of the central pixel.
    y : int
        Y-coordinate of the central pixel.
    xarr : SpectroscopicAxis
        The spectroscopic axis for the x-axis.
    ncomp : int
        The number of components in the model.
    size : int, optional
        Size of the grid (must be odd). Default is 3.
    xsize : int, optional
        Number of columns in the grid. Default is size.
    ysize : int, optional
        Number of rows in the grid. Default is size.
    xlim : tuple, optional
        X-axis limits for the plot.
    ylim : tuple, optional
        Y-axis limits for the plot.
    figsize : tuple, optional
        Size of the figure.
    origin : {'lower', 'upper'}, optional
        Origin of the grid. Default is 'lower'.
    mod_all : bool, optional
        Whether to plot all model components. Default is True.
    savename : str, optional
        If provided, save the figure to the given filename.
    **kwargs : dict
        Additional keyword arguments passed to `plot_model`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        The array of axes objects.
    """
    fig, axs = plot_spec_grid(cube, x, y, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim,
                              figsize=figsize, origin=origin, **kwargs)

    ysize, xsize = axs.shape
    xpad = int(xsize / 2)
    ypad = int(ysize / 2)

    if mod_all:
        for j in range(y - ypad, y + ypad + 1):
            for i in range(x - xpad, x + xpad + 1):
                plot_model(para[:, j, i], model_func, xarr, ax=axs[j - y + ypad, i - x + xpad], ncomp=ncomp, lw=1)
    else:
        # Plot the central pixel only
        plot_model(para[:, y, x], model_func, xarr, ax=axs[ypad, xpad], ncomp=ncomp, lw=1)

    if savename is not None:
        fig.savefig(savename, bbox_inches='tight')

    return fig, axs


#=======================

def strip_units(lim):
    """
    Helper function to strip units from a limit tuple if it contains Quantity.

    Parameters
    ----------
    lim : tuple or None
        A tuple of limits (min, max) which may contain Quantity instances with units.

    Returns
    -------
    tuple
        A tuple with units stripped, containing only numeric values.
    """
    if lim is not None:
        lim = tuple(val.value if isinstance(val, u.Quantity) else val for val in lim)
    return lim


def ensure_units_compatible(lim, data_unit, suppress_warning=False):
    """
    Ensure the limits have compatible units with the data, converting if needed.
    If the data has no units, strip units from the limits directly and issue a warning if desired.

    Parameters
    ----------
    lim : tuple or None
        The limit tuple (min, max) to be checked and potentially converted.
    data_unit : astropy.units.Unit or None
        The unit of the data to which the limits should be compatible.
    suppress_warning : bool, optional
        If True, suppresses warnings when stripping units from limits.

    Returns
    -------
    tuple
        Limits in compatible units with data, or stripped of units if data has no units.
    """
    if lim is not None:
        # Strip units if data has no units, and issue a warning if not suppressed
        if data_unit is None:
            for val in lim:
                if isinstance(val, u.Quantity) and not suppress_warning:
                    warnings.warn("Data has no units; stripping units from provided limit.")
            lim = strip_units(lim)
        else:
            # Convert each limit to match data units if it's a Quantity with a different unit
            lim = tuple(
                val.to(data_unit) if (val is not None and isinstance(val, u.Quantity) and val.unit != data_unit) else val
                for val in lim
            )
    return lim
