"""
The `mufasa.deblend_cube` module provides functionality for deblending hyperfine structures in spectral cubes,
allowing for the reconstruction of fitted models with Gaussian lines accounting for
optical depths.
"""

from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

#=======================================================================================================================

# import external library
import numpy as np
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
import astropy.units as u
import gc

from pyspeckit.parallel_map import parallel_map

from .utils.multicore import validate_n_cores
from .spec_models.meta_model import MetaModel

#=======================================================================================================================

def deblend(para, specCubeRef, vmin=4.0, vmax=11.0, f_spcsamp=None, tau_wgt=0.1, n_cpu=None, linetype='nh3', fittype=None):
    """
    Deblend hyperfine structures in a cube based on fitted models.

    This function reconstructs the fitted model with Gaussian lines accounting
    for optical depths (e.g., similar to CO rotational transitions).

    Parameters
    ----------
    para : numpy.ndarray
        The fitted parameters in the order of velocity, width, Tex, and tau for
        each velocity slab. The size of the z-axis for `para` must be a multiple of 4.
    specCubeRef : spectral_cube.SpectralCube
        The reference cube from which the deblended cube is constructed.
    vmin : float, optional
        The lower velocity limit on the deblended cube in km/s. Default is 4.0.
    vmax : float, optional
        The upper velocity limit on the deblended cube in km/s. Default is 11.0.
    f_spcsamp : int, optional
        The scaling factor for spectral sampling relative to the reference cube.
        For example, `f_spcsamp=2` doubles the spectral resolution.
    tau_wgt : float, optional
        A scaling factor for the input tau. For example, `tau_wgt=0.1` better
        represents the true optical depth of an NH3 (1,1) hyperfine group than
        the "fitted tau". Default is 0.1.
    n_cpu : int, optional
        The number of CPUs to use. If None, defaults to all CPUs available minus one.
    linetype : str, optional
        The line type to use for deblending. Options are 'nh3' or 'n2hp'. Default is 'nh3'.

    Returns
    -------
    mcube : spectral_cube.SpectralCube
        The deblended cube.
    """

    # get different types of deblending models

    if fittype is None:
        if linetype == 'nh3':
            #from .spec_models import nh3_deblended
            #deblend_mod = nh3_deblended.nh3_vtau_singlemodel_deblended
            fittype = 'nh3_multi_v'

        elif linetype == 'n2hp':
            #from .spec_models import n2hp_deblended
            #deblend_mod = n2hp_deblended.n2hp_vtau_singlemodel_deblended
            fittype = 'n2hp_multi_v'
        else:
            raise TypeError("{} is an invalid linetype".format(linetype))

    meta_model = MetaModel(fittype=fittype)
    deblend_mod = meta_model.model_func

    # open the reference cube file
    cube = specCubeRef
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')

    # trim the cube to the specified velocity range
    cube = cube.spectral_slab(vmin*u.km/u.s,vmax*u.km/u.s)

    # generate an empty SpectralCube to house the deblended cube
    if f_spcsamp is None:
        deblend = np.zeros(cube.shape)
        hdr = cube.wcs.to_header()
        wcs_new = cube.wcs
    else:
        deblend = np.zeros((cube.shape[0]*int(f_spcsamp), cube.shape[1], cube.shape[2]))
        wcs_new = cube.wcs.deepcopy()
        # adjust the spectral reference value
        wcs_new.wcs.crpix[2] = wcs_new.wcs.crpix[2]*int(f_spcsamp)
        # adjust the spaxel size
        wcs_new.wcs.cdelt[2] = wcs_new.wcs.cdelt[2]/int(f_spcsamp)
        hdr = wcs_new.to_header()

    # retain the beam information
    hdr['BMAJ'] = cube.header['BMAJ']
    hdr['BMIN'] = cube.header['BMIN']
    hdr['BPA'] = cube.header['BPA']

    mcube = SpectralCube(deblend, wcs_new, header=hdr)

    # convert back to an unit that the ammonia hf model can handle (i.e. Hz) without having to create a
    # pyspeckit.spectrum.units.SpectroscopicAxis object (which runs rather slow for model building in comparison)
    mcube = mcube.with_spectral_unit(u.Hz, velocity_convention='radio')
    xarr = mcube.spectral_axis

    yy,xx = np.indices(para.shape[1:])
    # a pixel is valid as long as it has a single finite value
    isvalid = np.any(np.isfinite(para),axis=0)
    valid_pixels = zip(xx[isvalid], yy[isvalid])

    def model_a_pixel(xy):
        x,y = int(xy[0]), int(xy[1])
        # nh3_vtau_singlemodel_deblended takes Hz as the spectral unit
        models = [deblend_mod(xarr, Tex=tex, tau=tau*tau_wgt, xoff_v=vel, width=width)
                  for vel, width, tex, tau in zip(para[::4, y,x], para[1::4, y,x], para[2::4, y,x], para[3::4, y,x])]

        mcube._data[:,y,x] = np.nansum(np.array(models), axis=0)
        return ((x, y), mcube._data[:, y, x])

    n_cpu = validate_n_cores(n_cpu)

    if n_cpu > 1:
        print("------------------ deblending cube -----------------")
        print("number of cpu used: {}".format(n_cpu))
        sequence = [(x, y) for x, y in valid_pixels]
        result = parallel_map(model_a_pixel, sequence, numcores=n_cpu)
        merged_result = [core_result for core_result in result
                         if core_result is not None]
        for mr in merged_result:
            ((x, y), model) = mr
            x = int(x)
            y = int(y)
            mcube._data[:, y, x] = model
    else:
        for xy in ProgressBar(list(valid_pixels)):
            model_a_pixel(xy)

    # convert back to km/s in units before saving
    mcube = mcube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    gc.collect()
    print("--------------- deblending completed ---------------")

    return mcube
