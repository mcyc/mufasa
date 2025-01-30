import numpy as np
from astropy.io import fits
from plotly.subplots import make_subplots
import plotly.offline as pyo

from ..utils import dataframe as dframe
from ..moment_guess import peakT as quickPeakT

#======================================================================================================================#
from ..utils.mufasa_log import get_logger
logger = get_logger(__name__)


class ScatterPPV(object):
    """A class to plot the fitted parameters in 3D scatter plots. Most of the data is stored in a pandas DataFrame.

    Parameters
    ----------
    parafile : str
        Path to the .fits file containing the MUFASA generated parameter maps.
    fittype : str
        The name of the fit model, e.g., "nh3_multi_v" or "n2hp_multi_v".
    vrange : tuple of float, optional
        Velocity range to clip the data (in km/s). Data outside this range is excluded.
        Default is None.
    verr_thres : float, optional
        Velocity error threshold (in km/s) to filter out data with higher errors.
        Data with a velocity error greater than this threshold is excluded. Default is 5.

    Examples
    --------

    Initialize the ScatterPPV object and plot the position-position-velocity (PPV) scatter plot:

    >>> sc = scatter_3D.ScatterPPV("path/to/fname.fits", fittype="nh3_multi_v")
    >>> sc.plot_ppv(savename='monR2.html', vel_scale=0.5)
    """

    def __init__(self, parafile, fittype, vrange=None, verr_thres=5, meta_model=None):
        """Initialize the ScatterPPV object by loading data from a .fits file and setting up parameters.

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
        self.paracube, self.header = fits.getdata(parafile, header=True)
        self.fittype = fittype
        self.meta_model = meta_model

        if meta_model is None:
            # get the rest frequency
            if self.fittype == "nh3_multi_v":
                #from pyspeckit.spectrum.models.ammonia_constants import freq_dict
                from ..spec_models.m_constants import nh3_constants
                freq_dict = nh3_constants['nh3_constants']
                self.rest_freq = freq_dict['oneone']*1e-9 # in GHz

            elif self.fittype == "n2hp_multi_v":
                #from ..spec_models.n2hp_constants import freq_dict
                from ..spec_models.m_constants import n2hp_constants
                freq_dict = nh3_constants['n2hp_constants']
                self.rest_freq = freq_dict['onezero']*1e-9 # in GHz

        else:
            if fittype != meta_model.fittype:
                msg = f"The provided fittype ({fittype}) does not match that from the MetaModel ({meta_model.fittype})." \
                      f"MetaModel's fittype will be adopted over the provided on."
                logger.warning(msg)

            self.fittype = meta_model.fittype
            self.rest_freq = self.meta_model.rest_value.value

        # structure the data in the data frame
        self.dataframe = dframe.make_dataframe(self.paracube, vrange=vrange, verr_thres=verr_thres)

        # estimate the peak intensity of each spectral model
        self.add_peakI()

        # get the relative wcs coordinates
        self.add_wcs_del()

    def set_meta_model(self, meta_model):
        self.meta_model = meta_model


    def add_peakI(self):
        """Calculate and add a peak intensity value for each model point in the DataFrame.

        Parameters
        ----------
        nu : float, optional
            Reference frequency (in GHz) to estimate the peak intensity. If not provided, defaults to the `rest_freq` attribute.

        Returns
        -------
        None
        """
        if self.meta_model is not None:
            para = np.array(
                [self.dataframe['vlsr'].values,
                 self.dataframe['sigv'].values,
                 self.dataframe['tex'].values,
                 self.dataframe['tau'].values]
            )
            self.dataframe['peakT'] = self.meta_model.peakT(para)

        else:
            self.dataframe['peakT'] = quickPeakT(tex=self.dataframe['tex'], tau=self.dataframe['tau'], nu=self.rest_freq)


    def add_wcs_del(self, ra_ref=None, dec_ref=None, unit='arcmin'):
        """Calculate relative RA & Dec coordinates and add them to the DataFrame as columns.

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

    def plot_ppv(self, label_key='peakT', vel_scale=0.8, xyunit='arcmin', savename=None, **kwargs):
        """
        Plot the fitted model in position-position-velocity (PPV) space as a 3D scatter plot.

        Points in the PPV plot are colored based on a specified key (e.g., intensity values)
        from the DataFrame. The velocity axis can be scaled relative to the spatial axes,
        and units for the x and y axes can be customized.

        Parameters
        ----------
        label_key : str, default='peakT'
            Column name in the DataFrame used for coloring data points.
            For example, 'peakT' for peak intensity or clustering labels.

        vel_scale : float, default=0.8
            Scaling factor for the velocity axis (z-axis) relative to the spatial axes.
            The x-axis is normalized to 1.

        xyunit : {'arcmin', 'pix'}, default='arcmin'
            Units for the x and y coordinates:
            - 'arcmin': Plot relative RA and Dec in arcminutes.
            - 'pix': Plot coordinates in pixels.

        savename : str, default=None
            File path to save the generated plot as an HTML file. If None, the plot is not saved.

        **kwargs : dict
            Additional keyword arguments passed to the `scatter_3D_df` function.
            Useful options include:

            auto_open_html : bool, default=True
                If True, automatically open the saved HTML file in a browser.

            mask_df : pandas.Series or None, default=None
                Boolean mask to filter the DataFrame before plotting.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            A 3D scatter plot figure representing the PPV data.

        Other Parameters
        ----------------
        kwargs : dict
            Additional options for customizing the scatter plot, such as color mapping
            and opacity scaling.

        Notes
        -----
        - If `label_key` is "peakT", default color mapping ("magma_r") and opacity ranges
          are applied automatically.
        - The velocity axis range is derived from the 1st and 99th percentiles of the
          "vlsr" column, with additional padding.
        - Color scaling uses the 1st and 99th percentiles of the column specified by `label_key`.

        Examples
        --------
        >>> model.plot_ppv(label_key='peakT', vel_scale=0.9, xyunit='pix', savename='ppv_plot.html')
        >>> model.plot_ppv(label_key='cluster_label', auto_open_html=False, mask_df=mask)

        See Also
        --------
        scatter_3D_df : Function used internally for generating the scatter plot.
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

        if label_key is not None:
            # Calculate the 1st and 99th percentiles for color and opacity scaling
            cmin, cmax = np.nanpercentile(self.dataframe[label_key][vmask], [1, 99])
            kwargs['vmin'] = cmin
            kwargs['vmax'] = cmax

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
                                 z_scale=vel_scale, savename=savename, **kwargs)

def scatter_3D_df(dataframe, x_key, y_key, z_key, label_key=None, mask_df=None,
                  auto_open_html=True, **kwargs):
    """A wrapper for scatter_3D to quickly plot a pandas DataFrame in 3D.

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
    """Plot a 3D scatter plot with optional opacity scaling for point ranges.

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
        if labels is None:
            labels = '#1f77b4' #muted blue
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
        # Set notebook mode to work in offline
        pyo.init_notebook_mode()
        fig.show()

    if savename is not None:
        fig.write_html(savename, auto_open=auto_open_html)

    return fig