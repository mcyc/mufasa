import numpy as np
from astropy.io import fits
from plotly.subplots import make_subplots
import plotly.offline as pyo
import pandas as pd

# Set notebook mode to work in offline
pyo.init_notebook_mode()

from ..utils import dataframe as dframe
from ..moment_guess import peakT


class ScatterPPV(object):
    def __init__(self, parafile, vrange=None, verr_min=5):
        self.paracube, self.header = fits.getdata(parafile, header=True)
        self.dataframe = dframe.make_dataframe(self.paracube, vrange=vrange, verr_min=verr_min)
        self.add_peakI()

    def add_peakI(self, nu=23.722634):
        self.dataframe['peakT'] = peakT(tex=self.dataframe['tex'], tau=self.dataframe['tau'], nu=nu)

    def plot_ppv(self, **kwargs):
        kwdf = dict(label_key='peakT', cmap='magma_r')
        kwargs = {**kwdf, **kwargs}

        label_key = kwargs['label_key']

        # Calculate the 1st and 99th percentiles for color and opacity scaling
        vmin, vmax = np.percentile(self.dataframe[label_key], [1, 99])

        # Pass vmin and vmax to the scatter_3D function
        self.fig = scatter_3D_df(self.dataframe, x_key='x_crd', y_key='y_crd', z_key='vlsr',
                                 vmin=vmin, vmax=vmax, opacity_scale=(vmin, vmax), **kwargs)


def scatter_3D_df(dataframe, x_key, y_key, z_key, label_key=None, mask_df=None,
                  auto_open_html=True, **kwargs):
    # Wrapper around scatter_3D to use pandas dataframe quickly
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


def scatter_3D(x, y, z, labels=None, nx=None, ny=None, shadow=True, fig=None, savename=None,
               scene=None, xlab=None, ylab=None, zlab=None, showfig=True, kw_line=None,
               cmap='Spectral_r', auto_open_html=True, vmin=None, vmax=None, **kwargs):
    '''
    Other parameters...
    :param vmin: Minimum value for color scaling
    :param vmax: Maximum value for color scaling
    '''

    if labels is None:
        labels = 'darkslateblue'

    # Set up color scaling
    marker = dict(size=1, color=labels, colorscale=cmap)

    # Check if vmin and vmax are provided for color scaling
    if vmin is not None and vmax is not None:
        marker['cmin'] = vmin
        marker['cmax'] = vmax
        marker['color'] = labels

    kw_scatter3d = dict(mode='markers', marker=marker)

    if kw_line is not None:
        kw_line_default = dict(color=labels, width=2)
        line = {**kw_line_default, **kw_line}
        kw_scatter3d['line'] = line
        del kw_scatter3d['mode']

    if nx is None:
        nx = x.max()
    if ny is None:
        ny = y.max()

    if isinstance(labels, int) or isinstance(labels, float):
        labels = labels.astype(float)
        labels[labels < 0] = np.nan

    if scene is None:
        scene = dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
                     xaxis=dict(),
                     yaxis=dict(),
                     zaxis=dict(),
                     aspectmode='manual',
                     aspectratio=dict(x=1, y=1 * ny / nx, z=0.8))

    if xlab is not None:
        scene['xaxis_title'] = xlab

    if ylab is not None:
        scene['yaxis_title'] = ylab

    if zlab is not None:
        scene['zaxis_title'] = zlab

    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    fig.add_scatter3d(x=x, y=y, z=z, **kw_scatter3d)

    if shadow:
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

    fig.update_layout(scene=scene, showlegend=False)

    if showfig:
        fig.show()

    if savename is not None:
        fig.write_html(savename, auto_open=auto_open_html)

    return fig

