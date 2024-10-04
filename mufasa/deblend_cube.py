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

#=======================================================================================================================

def deblend(para, specCubeRef, vmin=4.0, vmax=11.0, f_spcsamp = None, tau_wgt = 0.1, n_cpu=None, linetype='nh3'):
    '''
    Deblend hyperfine structures in a cube based on fitted models, i.e., reconstruct the fitted model with Gaussian
    lines with optical depths accounted for (e.g., similar to CO (J = 0-1))

    :param para: <ndarray>
        The fitted parameters in the order of vel, width, tex, and tau for each velocity slab.
        (Note: the size of the z axis for para must thus be in the multiple of 4)

    :param specCubeRef: <SpectralCube.Cube>
        The reference cube from which the deblended cube is constructed from

    :param vmin: <float>
        The lower veloicty limit on the deblended cube in the unit of km/s

    :param vmax: <float>
        The upper veloicty limit on the deblended cube in the unit of km/s

    :param f_spcsamp: <int>
        The scaling factor for the spectral sampling relative the reference cube
        (e.g., f_spcsamp = 2 give you twice the spectral resolution)

    :param tau_wgt:
        The scaling factor for the input tau
        (e.g., tau_wgt = 0.1 better represents the true optical depth of a NH3 (1,1) hyperfine group than the
         "fitted tau")

    :param n_cpu: <int>
        The number of cpus to use. If None, defaults to all the cpus available minus one.

    :return mcube: <SpectralCube.Cube>
        The deblended cube
    '''

    # get different types of deblending models
    if linetype == 'nh3':
        from .spec_models import nh3_deblended
        deblend_mod = nh3_deblended.nh3_vtau_singlemodel_deblended

    elif linetype == 'n2hp':
        from .spec_models import n2hp_deblended
        deblend_mod = n2hp_deblended.n2hp_vtau_singlemodel_deblended
    else:
        raise Exception("{} is an invalid linetype".format(linetype))

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
