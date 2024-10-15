import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
from pyspeckit.spectrum.units import SpectroscopicAxis


# =======================================================================================================================

class Plotter(object):
    def __init__(self, ucube, fittype, ncomp_list=None, spec_unit='km/s'):
        self.ucube = ucube
        self.cube = self.ucube.cube.with_spectral_unit(spec_unit, velocity_convention='radio')
        # need to check and see if cube has the right unit
        self.xarr = SpectroscopicAxis(self.cube.spectral_axis.value,
                                      unit=spec_unit,
                                      refX=self.cube._header['RESTFRQ'],
                                      velocity_convention='radio')

        # the following need to be consistient with the data provided
        self.xlab = r"$v_{\mathrm{LSR}}$ (km s$^{-1}$)"
        self.ylab = r"$T_{\mathrm{MB}}$ (K)"

        if fittype == "nh3_multi_v":
            from ..spec_models.ammonia_multiv import ammonia_multi_v
            self.model_func = ammonia_multi_v
        elif fittype == "n2hp_multi_v":
            from ..spec_models.n2hp_multiv import n2hp_multi_v
            self.model_func = n2hp_multi_v
        else:
            raise ValueError("{} is not one of the currently accepted fittypes.")

        self.parcubes = {}

        if ncomp_list is None:
            for key in self.ucube.pcubes:
                self.parcubes[key] = self.ucube.pcubes[key].parcube
        else:
            for n in ncomp_list:
                self.parcubes[str(n)] = self.ucube.pcubes[str(n)].parcube

    def plot_spec_grid(self, x, y, size=3, xsize=None, ysize=None, xlim=None, ylim=None, figsize=None, **kwargs):

        # add defaults that can be superceeded
        kwargs = {'xlab':self.xlab, 'ylab':self.ylab, **kwargs}

        self.fig, self.axs = \
            plot_spec_grid(self.cube, x, y, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim, figsize=figsize, **kwargs)

    def plot_spec(self, x, y, ax=None, xlab=None, ylab=None, **kwargs):
        spc = self.cube[:, y, x]
        if xlab is None:
            xlab = self.xlab
        if ylab is None:
            ylab = self.ylab
        return plot_spec(spc, xarr=self.xarr, ax=ax, xlab=xlab, ylab=ylab, **kwargs)


    def plot_fit(self, x, y, ax, ncomp, **kwargs):
        plot_model(self.parcubes[str(ncomp)][:,y,x], self.model_func, self.xarr, ax, ncomp, **kwargs)

    def plot_fits_grid(self, x, y, ncomp, size=3, xsize=None, ysize=None, xlim=None, ylim=None,
                       figsize=None, origin='lower', mod_all=True, savename=None, **kwargs):

        # add defaults that can be superceeded
        kwargs = {'xlab':self.xlab, 'ylab':self.ylab, **kwargs}

        plot_fits_grid(self.cube, self.parcubes[str(ncomp)], self.model_func, x, y, self.xarr,
                       ncomp=ncomp, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim,
                       figsize=figsize, origin=origin, mod_all=mod_all, savename=savename,
                       **kwargs)

# =======================================================================================================================


def get_cube_slab(cube, vZoomLims=(-5, 20)):
    # currently incomplete. it will be used to save some computation time when calculating the model and plotting
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    cube_s = cube.spectral_slab(vZoomLims[0] * u.km / u.s, vZoomLims[1] * u.km / u.s)

    # SpectroscopicAxis has the advantage of being able to performed unit conversion automatically
    xarr = SpectroscopicAxis(cube_s.spectral_axis.value, unit=cube_s.spectral_axis.unit,
                             refX=cube_s._header['RESTFRQ'], velocity_convention='radio')
    return cube_s, xarr


# =======================================================================================================================


def plot_spec(spc, xarr=None, ax=None, fill=False, xlab=None, ylab=None, **kwargs):

    if 'c' in kwargs:
        if 'color' in kwargs:
            raise TypeError("Got both 'color' and 'c', which are aliases of one another")
        else:
            kwargs['color'] = kwargs['c']
            del kwargs['c']
    elif 'color' not in kwargs:
        kwargs['color'] = '0.5'

    kwargs_df = dict(lw=1) # default kwargs
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

    if size%2 ==0 and size > 0:
        raise ValueError("Size provided must be an odd, positive interget that is not zero.")

    if xsize is None:
        xsize = size

    if ysize is None:
        ysize = size

    if size%2 ==0 and size > 0:
        raise ValueError("Size provided must be an odd, positive interget that is not zero.")

    if xsize is None:
        xsize = size

    if ysize is None:
        ysize = size

    # initiate the plot grid
    fig, axs = plt.subplots(ysize, xsize, sharex=all, sharey=all, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=figsize)

    return fig, axs


def plot_spec_grid(cube, x, y, size=3, xsize=None, ysize=None, xlim=None, ylim=None, figsize=None,
                   origin='lower', **kwargs):

    # grab the x y labels and ensure it wasn't passed downstream
    if 'xlab' in kwargs:
        xlab = kwargs['xlab']
        del kwargs['xlab']

    if 'ylab' in kwargs:
        ylab = kwargs['ylab']
        del kwargs['ylab']

    fig, axs = get_spec_grid(size=size, xsize=xsize, ysize=ysize, figsize=figsize, **kwargs)
    ysize, xsize = axs.shape

    xpad = int(xsize/2)
    ypad = int(ysize/2)

    # get a subcube
    scube = cube[:, y-ypad : y+ypad+1, x-xpad: x+xpad+ 1]

    if ylim is None:
        ymax = scube.max()
        ylim = (None, ymax*1.1)

    # plot spectra over the grid
    for index, ax in np.ndenumerate(axs):
        yi, xi = index

        if origin == 'lower':
            yi = ysize - 1 - yi
        elif origin != 'upper':
            raise KeyError("The keyword \'{}\' is unsupported.".format(origin))

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

    # provide a common labeling ax
    fig.add_subplot(111, frameon=False, zorder=-100)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    return fig, axs


def plot_model(para, model_func, xarr, ax, ncomp, **kwargs):

    for i in range(ncomp):
        pp = para[i*4:(i+1)*4]
        mod = model_func(xarr, *pp)
        if not 'alpha' in kwargs:
            kwargs['alpha'] = 0.6
        plot_spec(mod, xarr, ax, fill=True, color='C{}'.format(i), **kwargs)

    mod_tot = model_func(xarr, *para)
    plot_spec(mod_tot, xarr, ax, c='0.1', zorder=30, **kwargs)


def plot_fits_grid(cube, para, model_func, x, y, xarr, ncomp, size=3, xsize=None, ysize=None, xlim=None, ylim=None,
                   figsize=None, origin='lower', mod_all=True, savename=None, **kwargs):

    fig, axs = plot_spec_grid(cube, x, y, size=size, xsize=xsize, ysize=ysize, xlim=xlim, ylim=ylim,
                              figsize=figsize, origin=origin, **kwargs)

    ysize, xsize = axs.shape
    xpad = int(xsize/2)
    ypad = int(ysize/2)

    if mod_all:
        for j in range(y - ypad, y + ypad + 1):
            for i in range(x - xpad, x + xpad + 1):
                plot_model(para[:,j,i], model_func, xarr, ax=axs[j - y + ypad, i - x + xpad], ncomp=ncomp, lw=1)

    else:
        # plot the central pixel only
        plot_model(para[:,y,x], model_func, xarr, ax=axs[ypad, xpad], ncomp=ncomp, lw=1)

    if not savename is None:
        fig.savefig(savename, bbox_inches='tight')
    return fig