import numpy as np
from astropy.io import fits
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set notebook mode to work in offline
pyo.init_notebook_mode()

from ..utils import dataframe as dframe
from ..moment_guess import peakT


class ScatterPPV(object):
    """
    A class to plot the fitted parameters in 3D scatter plots. Most of the data is stored in a pandas DataFrame.

    Parameters
    ----------
    parafile : str
        Path to the .fits file containing the MUFASA generated parameter maps.
    fittype : str
        The name of the fit model, e.g., "nh3_multi_v" or "n2hp_multi_v".
    vrange : tuple of float, optional
        Velocity range to clip the data (in km/s). Data outside this range is excluded. Default is None.
    verr_thres : float, optional
        Velocity error threshold (in km/s) to filter out data with higher errors. Data with a velocity error greater than this threshold is excluded. Default is 5.

    Examples
    --------
    Initialize the ScatterPPV object and plot the position-position-velocity (PPV) scatter plot:

    >>> sc = scatter_3D.ScatterPPV("path/to/fname.fits", fittype="nh3_multi_v")
    >>> sc.plot_ppv(savename='monR2.html', vel_scale=0.5)
    """

    def __init__(self, parafile, fittype, vrange=None, verr_thres=5):
        """
        Initialize the ScatterPPV object by loading data from a .fits file and setting up parameters.

        For a detailed description of parameters, refer to the class docstring.

        Parameters
        ----------
        parafile : str
            Path to the .fits file of the modeled parameter maps.
        fittype : str
            Name of the fit model.
        vrange : tuple of float, optional
            Velocity range to clip the data (in km/s).
        verr_thres : float, optional
            The velocity error threshold (in km/s) to filter the data. Data with errors above this threshold is excluded.
        """

    def __init__(self, parafile, fittype, vrange=None, verr_thres=5):
        """
        Parameters
        ----------
        parafile : str
            Path to the .fits file containing the modeled parameter maps.
        fittype : str
            The name of the fit model, e.g., "nh3_multi_v" or "n2hp_multi_v".
        vrange : tuple of float, optional
            Velocity range to clip the data (in km/s). Data outside this range is excluded. Default is None.
        verr_thres : float, optional
            Velocity error threshold (in km/s) to filter out data with higher errors.
             Data with a velocity error greater than this threshold is excluded. Default is 5.
        """

        self.paracube, self.header = fits.getdata(parafile, header=True)
        self.fittype = fittype

        # get the rest frequency
        if self.fittype == "nh3_multi_v":
            from pyspeckit.spectrum.models.ammonia_constants import freq_dict
            self.rest_freq = freq_dict['oneone']*1e-9 # in GHz

        elif self.fittype == "n2hp_multi_v":
            from ..spec_models.n2hp_constants import freq_dict
            self.rest_freq = freq_dict['onezero']*1e-9 # in GHz

        # structure the data in the data frame
        self.dataframe = dframe.make_dataframe(self.paracube, vrange=vrange, verr_thres=verr_thres)

        # estimate the peak intensity of each spectral model
        self.add_peakI()

        # get the relative wcs coordinates
        self.add_wcs_del()


    def add_peakI(self, nu=None):
        """
        Calculate and add a peak intensity value for each model point in the DataFrame.

        Parameters
        ----------
        nu : float, optional
            Reference frequency (in GHz) to estimate the peak intensity. If not provided, defaults to the `rest_freq` attribute.

        Returns
        -------
        None
        """

        if nu is None:
            nu = self.rest_freq

        self.dataframe['peakT'] = peakT(tex=self.dataframe['tex'], tau=self.dataframe['tau'], nu=nu)

    def add_wcs_del(self, ra_ref=None, dec_ref=None, unit='arcmin'):
        """
        Calculate relative RA & Dec coordinates and add them to the DataFrame as columns.

        Parameters
        ----------
        ra_ref : float, optional
            Reference RA value to calculate relative RA. If not provided, uses the minimum RA in the data.
        dec_ref : float, optional
            Reference Dec value to calculate relative Dec. If not provided, uses the minimum Dec in the data.
        unit : {'arcmin', 'arcsec'}, optional
            Units for delta RA & Dec. Use 'arcmin' to plot in arcminutes or 'arcsec' for arcseconds. Default is 'arcmin'.

        Returns
        -------
        None
        """

        if unit == 'arcmin':
            f = 60
        elif unit == 'arcsec':
            f = 3600

        df = self.dataframe
        df['delt RA'] = df['x_crd'] * self.header['CDELT1'] * f * -1.0
        df['delt Dec'] = df['y_crd'] * self.header['CDELT2'] * f
        ''
        if ra_ref is None:
            ra_ref = df['delt RA'].min()
        if dec_ref is None:
            dec_ref = df['delt Dec'].min()

        df['delt RA'] = df['delt RA'] - ra_ref
        df['delt Dec'] = df['delt Dec'] - dec_ref


    def plot_ppv(self, label_key='peakT', vel_scale=0.8, xyunit='arcmin', **kwargs):
        """
        Plot the fitted model in position-position-velocity (PPV) space, with points colored by a specified key.

        Parameters
        ----------
        label_key : str, optional
            DataFrame column to color each data point by, e.g., 'peakT' for peak intensity or clustering labels. Default is 'peakT'.
        vel_scale : float, optional
            Scale factor for the velocity axis relative to x & y axes, where x is normalized to 1. Default is 0.8.
        xyunit : {'arcmin', 'pix'}, optional
            Units for x & y coordinates. If 'arcmin', plots relative RA and Dec in arcminutes. If 'pix', plots coordinates in pixels. Default is 'arcmin'.
        kwargs : dict
            Additional keyword arguments for customization in plotting.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The created 3D scatter plot figure.
        """

        if label_key == 'peakT':
            kwdf = dict(cmap='magma_r', opacity_ranges=5)
            kwargs = {**kwdf, **kwargs}
            kwargs['label_key'] = label_key

        # get the velocity range to plot
        vmin, vmax = np.nanpercentile(self.dataframe['vlsr'], [1, 99])
        vpad = (vmax - vmin)/10
        if vpad > 0.5:
            vmax += vpad
        else:
            vmax += 0.5

        vmask = np.logical_and(self.dataframe['vlsr']>vmin, self.dataframe['vlsr']<vmax)

        # Calculate the 1st and 99th percentiles for color and opacity scaling
        cmin, cmax = np.nanpercentile(self.dataframe[label_key][vmask], [1, 99])

        if xyunit == 'arcmin':
            # plot delta RA & Dec in arcmin
            kwargs['x_key'] = 'delt RA'
            kwargs['y_key'] = 'delt Dec'
            kwargs['xlab'] = u'\u0394' + 'RA (arcmin)'
            kwargs['ylab'] = u"\u0394" + 'Dec (arcmin)'

        elif xyunit == 'pix':
            # plot x & y in pixel coordinates
            kwargs['x_key'] = 'y_crd'
            kwargs['y_key'] = 'delt Dec'
            kwargs['xlab'] = 'RA (pix)'
            kwargs['ylab'] = 'Dec (pix)'

        self.fig = scatter_3D_df(self.dataframe[vmask], z_key='vlsr',
                                 zlab='<i>v</i><sub>LSR</sub> (km s<sup>-1</sup>)',
                                 nx=self.header['NAXIS1'], ny=self.header['NAXIS2'],
                                 vmin=cmin, vmax=cmax, z_scale=vel_scale,
                                 **kwargs)

def scatter_3D_df(dataframe, x_key, y_key, z_key, label_key=None, mask_df=None,
                  auto_open_html=True, **kwargs):
    """
    A wrapper for scatter_3D to quickly plot a pandas DataFrame in 3D.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing the data to plot.
    x_key : str
        Column name for the x-axis values.
    y_key : str
        Column name for the y-axis values.
    z_key : str
        Column name for the z-axis values.
    label_key : str, optional
        Column name to color the data points by. If None, data points are plotted without color scaling. Default is None.
    mask_df : pandas.Series or None, optional
        Boolean mask to filter the DataFrame before plotting. Default is None.
    auto_open_html : bool, optional
        Whether to automatically open the HTML plot file upon saving. Default is True.
    kwargs : dict
        Additional keyword arguments for the 3D scatter plot.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The 3D scatter plot figure.
    """

    if mask_df is not None:
        dataframe = dataframe[mask_df]

    x, y, z = dataframe[x_key], dataframe[y_key], dataframe[z_key]

    if label_key is None:
        labels = None
    elif label_key in dataframe.keys():
        labels = dataframe[label_key]
    else:
        labels = label_key

    return scatter_3D(x, y, z, labels=labels, auto_open_html=auto_open_html, **kwargs)


def scatter_3D(x, y, z, labels=None, nx=None, ny=None, z_scale=0.8, shadow=True, fig=None, savename=None,
               scene=None, xlab=None, ylab=None, zlab=None, showfig=True, kw_line=None,
               cmap='Spectral_r', auto_open_html=True, vmin=None, vmax=None,
               opacity_ranges=1, **kwargs):
    """
    Plot a 3D scatter plot with optional opacity scaling for point ranges.

    Parameters
    ----------
    x : array-like
        X coordinates of the data points.
    y : array-like
        Y coordinates of the data points.
    z : array-like
        Z coordinates of the data points.
    labels : array-like, optional
        Label values for color scaling. If 'peakT' (default), opacity is split into ranges; otherwise, a single trace with full opacity is used.
    nx : int, optional
        Number of pixels in x to set the aspect ratio. Default is None.
    ny : int, optional
        Number of pixels in y to set the aspect ratio. Default is None.
    z_scale : float, optional
        Aspect ratio for z-axis relative to x and y axes. Default is 0.8.
    shadow : bool or float, optional
        Adds a shadow projection on the z-plane. Default is True.
    fig : plotly.graph_objs.Figure, optional
        Figure to plot on. If None, creates a new figure.
    savename : str, optional
        Path to save the plot as an HTML file. Default is None.
    scene : dict, optional
        Scene configuration for the 3D plot. Default is None.
    xlab : str, optional
        X-axis label. Default is None.
    ylab : str, optional
        Y-axis label. Default is None.
    zlab : str, optional
        Z-axis label. Default is None.
    showfig : bool, optional
        Whether to display the figure after plotting. Default is True.
    kw_line : dict, optional
        Dictionary of line properties for connecting points, if desired. Default is None.
    cmap : str, optional
        Colormap for data points. Default is 'Spectral_r'.
    auto_open_html : bool, optional
        If True, auto-opens saved HTML. Default is True.
    vmin : float, optional
        Minimum value for color scaling. Default is None.
    vmax : float, optional
        Maximum value for color scaling. Default is None.
    opacity_ranges : int, optional
        Number of opacity ranges (1 to 5). For 'peakT' labels, splits opacity into equal intervals over 1-99 percentile. Default is 1.
    kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Generated 3D scatter plot figure.
    """

    if fig is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    if nx is None:
        nx = x.max()
    if ny is None:
        ny = y.max()

    # Define opacity levels and ranges if more than one range is specified
    opacity_levels = np.linspace(0.1, 1, opacity_ranges)
    if labels is not None and opacity_ranges > 1:
        # Calculate percentiles for the given number of opacity ranges
        percentiles = np.linspace(0, 100, opacity_ranges + 1)
        cutoffs = np.nanpercentile(labels, percentiles)

        # Plot each range with respective opacity
        for i, (opacity, cutoff_min, cutoff_max) in enumerate(zip(opacity_levels, cutoffs[:-1], cutoffs[1:])):
            mask = (labels >= cutoff_min) & (labels < cutoff_max)
            sub_x, sub_y, sub_z = x[mask], y[mask], z[mask]
            sub_labels = labels[mask]

            marker = dict(size=1, color=sub_labels, colorscale=cmap, opacity=opacity)
            if vmin is not None and vmax is not None:
                marker.update(cmin=vmin, cmax=vmax)

            kw_scatter3d = dict(mode='markers', marker=marker)

            if kw_line is not None:
                kw_line_default = dict(color=labels, width=2)
                line = {**kw_line_default, **kw_line}
                kw_scatter3d['line'] = line
                del kw_scatter3d['mode']

            fig.add_scatter3d(x=sub_x, y=sub_y, z=sub_z, mode='markers', marker=marker)
    else:
        # Single trace for non-peakT labels or opacity_ranges=1
        marker = dict(size=1, color=labels, colorscale=cmap, opacity=1.0)
        if vmin is not None and vmax is not None:
            marker.update(cmin=vmin, cmax=vmax)

        kw_scatter3d = dict(mode='markers', marker=marker)

        fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker=marker)

    # Configure the scene and other plot settings
    if scene is None:
        scene = dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
                     xaxis=dict(),
                     yaxis=dict(),
                     zaxis=dict(),
                     aspectmode='manual',
                     aspectratio=dict(x=1, y=1*ny/nx, z=z_scale)) #fixed aspect ratio

    fig.update_layout(scene=scene, showlegend=False)

    if xlab is not None:
        fig.update_layout(scene=dict(xaxis_title=xlab))
    if ylab is not None:
        fig.update_layout(scene=dict(yaxis_title=ylab))
    if zlab is not None:
        fig.update_layout(scene=dict(zaxis_title=zlab))

    if shadow:
        marker = dict(size=1, color=labels, colorscale=cmap, opacity=1.0)
        if vmin is not None and vmax is not None:
            marker.update(cmin=vmin, cmax=vmax)

        # display the shadow
        z_shadow = z.copy()
        if isinstance(shadow, bool):
            z_shadow[:] = np.nanmin(z) - 0.5

        elif isinstance(shadow, int) or isinstance(shadow, float):
            z_shadow[:] = shadow

        mk = {**marker, 'opacity':0.03}
        kw_scatter3d_mod = {**kw_scatter3d, "marker":mk}
        fig.add_scatter3d(x=x, y=y, z=z_shadow, **kw_scatter3d_mod)

        # add a low transparent layer of grey to make it look more like a "shadow"
        mk = {**marker, 'opacity': 0.02, 'color':'grey'}
        kw_scatter3d_mod = {**kw_scatter3d, "marker": mk}
        fig.add_scatter3d(x=x, y=y, z=z_shadow, **kw_scatter3d_mod)

    if showfig:
        fig.show()

    if savename is not None:
        fig.write_html(savename, auto_open=auto_open_html)

    return fig