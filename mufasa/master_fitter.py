from __future__ import print_function
from __future__ import absolute_import
__author__ = 'mcychen'

#=======================================================================================================================
import os
import numpy as np
from spectral_cube import SpectralCube
from astropy import units as u
from skimage.morphology import dilation
import astropy.io.fits as fits
import gc
#from scipy.ndimage.filters import median_filter
from scipy.signal import medfilt2d
from skimage.morphology import dilation, square

from . import UltraCube as UCube
from . import moment_guess as mmg
from . import convolve_tools as cnvtool
from . import guess_refine as gss_rf


#=======================================================================================================================

class Region(object):

    def __init__(self, cubePath, paraNameRoot, paraDir=None, cnv_factor=2):

        self.cubePath = cubePath
        self.paraNameRoot = paraNameRoot
        self.paraDir = paraDir

        self.ucube = UCube.UCubePlus(cubePath, paraNameRoot=paraNameRoot, paraDir=paraDir, cnv_factor=cnv_factor)

        # for convolving cube
        self.cnv_factor = cnv_factor


    def get_convolved_cube(self, update=True, cnv_cubePath=None, edgetrim_width=5, paraNameRoot=None, paraDir=None):
        if paraDir is None:
            paraDir = self.paraDir
        get_convolved_cube(self, update=update, cnv_cubePath=cnv_cubePath, edgetrim_width=edgetrim_width,
                           paraNameRoot=paraNameRoot, paraDir=paraDir)


    def get_convolved_fits(self, ncomp, **kwargs):
        get_convolved_fits(self, ncomp, **kwargs)


    def get_fits(self, ncomp, **kwargs):
        get_fits(self, ncomp, **kwargs)


    def load_fits(self, ncomp):
        # basically the same as get_fits(), but with update set to False to ensure the fits aren't refitted
        get_fits(self, ncomp, update=False)


    def master_2comp_fit(self, snr_min=3, **kwargs):
        master_2comp_fit(self, snr_min=snr_min, **kwargs)

    def standard_2comp_fit(self, planemask=None):
        standard_2comp_fit(self, planemask=planemask)



#=======================================================================================================================


def get_convolved_cube(reg, update=True, cnv_cubePath=None, edgetrim_width=5, paraNameRoot=None, paraDir=None):

    if cnv_cubePath is None:
        root = "conv{0}Xbeam".format(int(np.rint(reg.cnv_factor)))
        reg.cnv_cubePath = "{0}_{1}.fits".format(os.path.splitext(reg.cubePath)[0], root)
    else:
        reg.cnv_cubePath = cnv_cubePath

    reg.cnv_para_paths ={}

    if update or (not os.path.isfile(reg.cnv_cubePath)):
        reg.ucube.convolve_cube(factor=reg.cnv_factor, savename=reg.cnv_cubePath, edgetrim_width=edgetrim_width)

    if paraNameRoot is None:
        paraNameRoot = "{}_conv{}Xbeam".format(reg.paraNameRoot, int(np.rint(reg.cnv_factor)))

    if paraDir is None:
        paraDir = reg.paraDir

    reg.ucube_cnv = UCube.UCubePlus(cubefile=reg.cnv_cubePath, paraNameRoot=paraNameRoot,
                                     paraDir=paraDir, cnv_factor=reg.cnv_factor)

    # MC: a mechanism is needed to make sure the convolved cube has the same resolution has the cnv_factor


def get_convolved_fits(reg, ncomp, **kwargs):
    if not hasattr(reg, 'ucube_cnv'):
        reg.get_convolved_cube(update=True)
    elif 'update' in kwargs:
        reg.get_convolved_cube(update=kwargs['update'])
    reg.ucube_cnv.get_model_fit(ncomp, **kwargs)


def get_fits(reg, ncomp, **kwargs):
    reg.ucube.get_model_fit(ncomp, **kwargs)


#=======================================================================================================================
# functions specific to 2-comonent fits


def master_2comp_fit(reg, snr_min=0.0, recover_wide=True, planemask=None, updateCnvFits=True, refit_bad_pix=True):
    # note, planemask superseeds snr-based maask
    iter_2comp_fit(reg, snr_min=snr_min, updateCnvFits=updateCnvFits, planemask=planemask)

    if refit_bad_pix:
        refit_bad_2comp(reg.ucube, snr_min=snr_min, lnk_thresh=-20)

    if recover_wide:
        refit_2comp_wide(reg, snr_min=snr_min)
    save_best_2comp_fit(reg)

    return reg


def iter_2comp_fit(reg, snr_min=3, updateCnvFits=True, planemask=None):
    # ensure this is a two compnent fitting method
    ncomp = [1,2]

    # convolve the cube and fit it
    # get_convolved_cube(reg, update=updateCnvFits)
    get_convolved_fits(reg, ncomp, update=updateCnvFits, snr_min=snr_min)

    # use the result from the convolved cube as guesses for the full resolution fits
    for nc in ncomp:
        pcube_cnv = reg.ucube_cnv.pcubes[str(nc)]
        para_cnv = np.append(pcube_cnv.parcube, pcube_cnv.errcube, axis=0)
        if nc == 2:
            para_cnv = gss_rf.quick_2comp_sort(para_cnv, filtsize=3)

        guesses = gss_rf.guess_from_cnvpara(para_cnv, reg.ucube_cnv.cube.header, reg.ucube.cube.header)
        # update is set to True to save the fits
        kwargs = {'update':True, 'guesses':guesses, 'snr_min':snr_min}
        if planemask is not None:
            kwargs['maskmap'] = planemask
        reg.ucube.get_model_fit([nc], **kwargs)


def refit_bad_2comp(ucube, snr_min=3, lnk_thresh=-20):
    # refit pixels where 2 component fits are substentially worse than good one components
    # defualt threshold of -20 should be able to pickup where 2 compoent fits are exceptionally poor

    print("begin re-fitting bad 2-comp pixels")

    lnk21 = ucube.get_AICc_likelihood(2, 1)
    lnk10 = ucube.get_AICc_likelihood(1, 0)

    # where the fits are poor
    mask = np.logical_and(lnk10 > 5, lnk21 < lnk_thresh)
    mask = np.logical_and(mask, np.isfinite(lnk10))

    mask_size = np.sum(mask)
    print("refit mask size for bad_2comp: {}".format(mask_size))

    guesses = ucube.pcubes['2'].parcube.copy()

    for i, gmap in enumerate(guesses):
        guesses[i] = medfilt2d(gmap, kernel_size=3)

    gc.collect()
    # re-fit and save the updated model
    replace_bad_pix(ucube, mask, snr_min, guesses, lnk21)



def refit_swap_2comp(reg, snr_min=3):

    ncomp = [1, 2]

    # load the fitted parameters
    for nc in ncomp:
        if not str(nc) in reg.ucube.pcubes:
            # no need to worry about wide seperation as they likley don't overlap in velocity space
            reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc)

    # refit only over where two component models are already determined to be better
    # note: this may miss a few fits where the swapped two-comp may return a better result?
    lnk21 = reg.ucube.get_AICc_likelihood(2, 1)
    mask = lnk21 > 5

    gc.collect()

    # swap the parameter of the two slabs and use it as the initial guesses
    guesses = reg.ucube.pcubes['2'].parcube.copy()

    for i in range(4):
        guesses[i], guesses[i+4] = guesses[i+4], guesses[i]

    ucube_new = UCube.UltraCube(reg.ucube.cubefile)
    ucube_new.fit_cube(ncomp=[2], maskmap=mask, snr_min=snr_min, guesses=guesses)

    gc.collect()

    # do a model comparison between the new two component fit verses the original one
    lnk_NvsO = UCube.calc_AICc_likelihood(ucube_new, 2, 2, ucube_B=reg.ucube)

    gc.collect()

    # adopt the better fit parameters into the final map
    good_mask = lnk_NvsO > 0
    # replace the values
    replace_para(reg.ucube.pcubes['2'], ucube_new.pcubes['2'], good_mask)

    # re-fit and save the updated model
    save_updated_paramaps(reg.ucube, ncomps=[2, 1])



def refit_2comp_wide(reg, snr_min=3):

    print("begin wide component recovery")

    ncomp = [1, 2]

    # load the fitted parameters
    for nc in ncomp:
        if not str(nc) in reg.ucube.pcubes:
            if not str(nc) in reg.ucube.paraPaths:
                reg.ucube.paraPaths[str(nc)]= '{}/{}_{}vcomp.fits'.format(reg.ucube.paraDir, reg.ucube.paraNameRoot, nc)

            if nc==2 and not ('2_noWideDelV' in reg.ucube.paraPaths):
                reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc)
                reg.ucube.pcubes['2_noWideDelV'] =\
                    "{}_noWideDelV".format(os.path.splitext(reg.ucube.paraPaths[str(nc)])[0])
            else:
                reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc)

    wide_comp_guess = get_2comp_wide_guesses(reg)

    # use the one component fit and the refined 1-componet guess for the residual to perform the two components fit
    final_guess = np.append(reg.ucube.pcubes['1'].parcube, wide_comp_guess, axis=0)

    # fit over where one-component was a better fit in the last iteration (since we are only interested in recovering
    # a second componet that is found in wide seperation)
    lnk21 = reg.ucube.get_AICc_likelihood(2, 1)
    mask = lnk21 < 5
    mask = dilation(mask)

    # combine the mask with where 1 component model is better fitted than the noise to save some computational time
    lnk10 = reg.ucube.get_AICc_likelihood(1, 0)
    mask = np.logical_and(mask, lnk10 > 5)

    mask_size = np.sum(mask)
    print("wide recovery refit mask size: {}".format(mask_size))

    replace_bad_pix(reg.ucube, mask, snr_min, final_guess, lnk21)




def replace_bad_pix(ucube, mask, snr_min, guesses, lnk21):
    # refit bad pixels marked by the mask, save the new parameter files with the bad pixels replaced
    if np.sum(mask) >= 1:
        ucube_new = UCube.UltraCube(ucube.cubefile)
        ucube_new.fit_cube(ncomp=[2], maskmap=mask, snr_min=snr_min, guesses=guesses)

        # do a model comparison between the new two component fit verses the original one
        lnk_NvsO = UCube.calc_AICc_likelihood(ucube_new, 2, 2, ucube_B=ucube)

        # mask over where one comp fit is more robust
        good_mask = np.logical_and(lnk_NvsO > 0, lnk21 < 5)
        good_mask = np.logical_and(good_mask, np.isfinite(lnk_NvsO))

        # replace the values
        replace_para(ucube.pcubes['2'], ucube_new.pcubes['2'], good_mask)

        # save the updated results
        save_updated_paramaps(ucube, ncomps=[2, 1])
    else:
        print("not enough pixels to refit, no-refit is done")



def standard_2comp_fit(reg, planemask=None, snr_min=3):
    # two compnent fitting method using the moment map guesses method
    ncomp = [1,2]

    # only use the moment maps for the fits
    for nc in ncomp:
        # update is set to True to save the fits
        kwargs = {'update':True, 'snr_min':snr_min}
        if planemask is not None:
            kwargs['maskmap'] = planemask
        reg.ucube.get_model_fit([nc], **kwargs)


def save_updated_paramaps(ucube, ncomps):
    # save the updated parameter cubes
    for nc in ncomps:
        if not str(nc) in ucube.paraPaths:
            print("[ERROR]: the ucube does not have paraPath for '{}' components".format(nc))
        else:
            ucube.save_fit(ucube.paraPaths[str(nc)], nc)



def save_best_2comp_fit(reg):
    # should be renamed to determine_best_2comp_fit or something along that line
    # currently use np.nan for pixels with no models

    ncomp = [1, 2]

    # ideally, a copy function should be in place of reloading

    # a new Region object is created start fresh on some of the functions (e.g., aic comparison)
    reg_final = Region(reg.cubePath, reg.paraNameRoot, reg.paraDir)

    # start out clean, especially since the deepcopy function doesn't work well for pyspeckit cubes
    # load the file based on the passed in reg, rather than the default
    for nc in ncomp:
        if not str(nc) in reg.ucube.pcubes:
            reg_final.load_fits(ncomp=[nc])
        else:
            # load files using paths from reg if they exist
            print("loading model from: {}".format(reg.ucube.paraPaths[str(nc)]))
            reg_final.ucube.load_model_fit(filename=reg.ucube.paraPaths[str(nc)], ncomp=nc)

    pcube_final = reg_final.ucube.pcubes['2'].copy('deep')

    # make the 2-comp para maps with the best fit model
    lnk21 = reg_final.ucube.get_AICc_likelihood(2, 1)
    mask = lnk21 > 5
    print("pixels better fitted by 2-comp: {}".format(np.sum(mask)))
    pcube_final.parcube[:4, ~mask] = reg_final.ucube.pcubes['1'].parcube[:4, ~mask].copy()
    pcube_final.errcube[:4, ~mask] = reg_final.ucube.pcubes['1'].errcube[:4, ~mask].copy()
    pcube_final.parcube[4:8, ~mask] = np.nan
    pcube_final.errcube[4:8, ~mask] = np.nan

    lnk10 = reg_final.ucube.get_AICc_likelihood(1, 0)
    mask = lnk10 > 5
    pcube_final.parcube[:, ~mask] = np.nan
    pcube_final.errcube[:, ~mask] = np.nan

    # use the default file formate to save the finals
    nc = 2
    if not str(nc) in reg_final.ucube.paraPaths:
        reg_final.ucube.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(reg_final.ucube.paraDir, reg_final.ucube.paraNameRoot, nc)

    savename = "{}_final.fits".format(os.path.splitext(reg_final.ucube.paraPaths['2'])[0])
    UCube.save_fit(pcube_final, savename=savename, ncomp=2)

    hdr2D =reg.ucube.cube.wcs.celestial.to_header()

    paraDir = reg_final.ucube.paraDir
    paraRoot = reg_final.ucube.paraNameRoot

    def make_save_name(paraRoot, paraDir, key):
        # replace "parameters" related names in the save names
        if "parameters" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("parameters", key))
        elif "parameter" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("parameter", key))
        elif "para" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("para", key))
        else:
            return "{}/{}_{}.fits".format(paraDir, paraRoot, key)


    # save the lnk21 map
    savename = make_save_name(paraRoot, paraDir, "lnk21")
    save_map(lnk21, hdr2D, savename)

    # save the lnk10 map
    savename = make_save_name(paraRoot, paraDir, "lnk10")
    save_map(lnk10, hdr2D, savename)

    # create and save the lnk20 map for reference:
    lnk20 = reg_final.ucube.get_AICc_likelihood(2, 0)
    savename = make_save_name(paraRoot, paraDir, "lnk20")
    save_map(lnk20, hdr2D, savename)

    # save the SNR map
    snr_map = get_best_2comp_snr_mod(reg_final)
    savename = make_save_name(paraRoot, paraDir, "SNR")
    save_map(snr_map, hdr2D, savename)

    # create moment0 map
    modbest = get_best_2comp_model(reg_final)
    cube_mod = SpectralCube(data=modbest, wcs=reg_final.ucube.pcubes['2'].wcs.copy(),
                            header=reg_final.ucube.pcubes['2'].header.copy())
    # make sure the spectral unit is in km/s before making moment maps
    cube_mod = cube_mod.with_spectral_unit('km/s', velocity_convention='radio')
    mom0_mod = cube_mod.moment0()
    savename = make_save_name(paraRoot, paraDir, "model_mom0")
    mom0_mod.write(savename, overwrite=True)

    # created masked mom0 map with model as the mask
    mom0 = UCube.get_masked_moment(cube=reg_final.ucube.cube, model=modbest, order=0, expand=20, mask=None)
    savename = make_save_name(paraRoot, paraDir, "mom0")
    mom0.write(savename, overwrite=True)

    # save reduced chi-squred maps
    # would be useful to check if 3rd component is needed
    savename = make_save_name(paraRoot, paraDir, "chi2red_final")
    chi_map = UCube.get_chisq(cube=reg_final.ucube.cube, model=modbest, expand=20, reduced=True, usemask=True,
                              mask=None)
    save_map(chi_map, hdr2D, savename)

    # save reduced chi-squred maps for 1 comp and 2 comp individually
    chiRed_1c = reg_final.ucube.get_reduced_chisq(1)
    chiRed_2c = reg_final.ucube.get_reduced_chisq(2)

    savename = make_save_name(paraRoot, paraDir, "chi2red_1c")
    save_map(chiRed_1c, hdr2D, savename)

    savename = make_save_name(paraRoot, paraDir, "chi2red_2c")
    save_map(chiRed_2c, hdr2D, savename)

    return reg


def save_map(map, header, savename, overwrite=True):
    fits_map = fits.PrimaryHDU(data=map, header=header)
    fits_map.writeto(savename,overwrite=overwrite)

#=======================================================================================================================
# functions that ficilates


def get_2comp_wide_guesses(reg):

    if not hasattr(reg, 'ucube_res_cnv'):
        # fit the residual with the one component model if this has not already been done.
        fit_best_2comp_residual_cnv(reg)

    def get_mom_guesses(reg):
        # get moment guesses from no masking
        cube_res = get_best_2comp_residual_SpectralCube(reg, masked=False, window_hwidth=3.5)

        # find moment around where one component has been fitted
        pcube1 = reg.ucube.pcubes['1']
        #vmap = median_filter(pcube1.parcube[0], size=3)
        vmap = medfilt2d(pcube1.parcube[0], kernel_size=3) # median smooth within a 3x3 square
        moms_res = mmg.vmask_moments(cube_res, vmap=vmap, window_hwidth=3.5)

        ncomp = 1
        gg = mmg.moment_guesses(moms_res[1], moms_res[2], ncomp, moment0=moms_res[0])
        return gg

    if not hasattr(reg.ucube_res_cnv.pcubes['1'], 'parcube'):
        # if there were no successful fit to the convolved cube, get moment map from no masking
        return get_mom_guesses(reg)
    
    else:
        # mask over where one component model is better than a no-signal (i.e., noise) model
        aic1v0_mask = reg.ucube_res_cnv.get_AICc_likelihood(1, 0) > 5

        if np.sum(aic1v0_mask) >= 1:
            # if there are at least one well fitted pixel to the residual
            data_cnv = np.append(reg.ucube_res_cnv.pcubes['1'].parcube, reg.ucube_res_cnv.pcubes['1'].errcube, axis=0)
            preguess = data_cnv.copy()

            # set pixels that are better modelled as noise to nan
            preguess[:, ~aic1v0_mask] = np.nan

            # use the dialated mask as a footprint to interpolate the guesses
            gmask = dilation(aic1v0_mask)
            guesses_final = gss_rf.guess_from_cnvpara(preguess, reg.ucube_res_cnv.cube.header, reg.ucube.cube.header,
                                                      mask=gmask)
        else:
            # get moment guesses without masking instead
            guesses_final = get_mom_guesses(reg)

        return guesses_final



def fit_best_2comp_residual_cnv(reg, window_hwidth=3.5, res_snr_cut=5, savefit=True):
    # fit the residual of the best fitted model (note, this approach may not hold well if the two-slab model
    # insufficiently at describing the observation. Luckily, however, this fit is only to provide initial guess for the
    # final fit)
    # the default window_hwidth = 3.5 is about half-way between the main hyperfine and the satellite

    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists
    cube_res_cnv = get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=window_hwidth, res_snr_cut=res_snr_cut)

    ncomp = 1

    # note: no further masking is applied to vmap, as we assume only pixels with good vlsr will be used
    if not hasattr(reg, 'ucube_cnv'):
        # if convolved cube was not used to produce initial guesses, use the full resolution 1-comp fit as the reference
        pcube1 = reg.ucube.pcubes['1']
        #vmap = median_filter(pcube1.parcube[0], size=3) # median smooth within a 3x3 square
        vmap = medfilt2d(pcube1.parcube[0], kernel_size=3) # median smooth within a 3x3 square
        vmap = cnvtool.regrid(vmap, header1=get_skyheader(pcube1.header), header2=get_skyheader(cube_res_cnv.header))
    else:
        vmap = reg.ucube_cnv.pcubes['1'].parcube[0]

    moms_res_cnv = mmg.vmask_moments(cube_res_cnv, vmap=vmap, window_hwidth=window_hwidth)

    gg = mmg.moment_guesses(moms_res_cnv[1], moms_res_cnv[2], ncomp, moment0=moms_res_cnv[0])

    # should try to use UCubePlus??? may want to avoid saving too many intermediate cube products
    reg.ucube_res_cnv = UCube.UltraCube(cube=cube_res_cnv)
    reg.ucube_res_cnv.fit_cube(ncomp=[1], snr_min=3, guesses=gg)

    # save the residual fit
    if savefit:
        oriParaPath = reg.ucube.paraPaths[str(ncomp)]
        savename = "{}_onBestResidual.fits".format(os.path.splitext(oriParaPath)[0])
        reg.ucube_res_cnv.save_fit(savename, ncomp)



def get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=3.5, res_snr_cut=5):
    #convolved residual cube.If masked is True, only convolve over where 'excessive' residual is
    # above a peak SNR value of res_snr_cut masked

    cube_res_masked = get_best_2comp_residual_SpectralCube(reg, masked=masked, window_hwidth=window_hwidth, res_snr_cut=res_snr_cut)

    cube_res_cnv = cnvtool.convolve_sky_byfactor(cube_res_masked, factor=reg.cnv_factor, edgetrim_width=None,
                                                 snrmasked=False, iterrefine=False)

    #cube_res_cnv = cube_res_cnv.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    return cube_res_cnv


def get_best_2comp_residual_SpectralCube(reg, masked=True, window_hwidth=3.5, res_snr_cut=5):
    # return residual cube as SpectralCube oobject
    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists

    res_cube = get_best_2comp_residual(reg)
    best_res = res_cube._data
    cube_res = SpectralCube(data=best_res, wcs=reg.ucube.pcubes['2'].wcs.copy(),
                            header=reg.ucube.pcubes['2'].header.copy())

    if masked:
        best_rms = UCube.get_rms(res_cube._data)

        # calculate the peak SNR value of the best-fit residual over the main hyperfine components
        vmap = reg.ucube.pcubes['1'].parcube[0]
        # make want to double check that masked cube always masks out nan values
        res_main_hf = mmg.vmask_cube(res_cube, vmap, window_hwidth=window_hwidth)

        if res_main_hf.size > 1e7:
            # to avoid loading entire cube and stress the memory
            how = 'slice'
        else:
            # note: 'auto' currently returns slice for n > 1e8
            how = 'auto'

        # enable huge operations (note: not needed when "how" is chosen wisely, which it should be)
        res_main_hf.allow_huge_operations = True
        res_main_hf_snr = res_main_hf.max(axis=0, how=how).value / best_rms
        res_main_hf.allow_huge_operations = False

        # mask out residual with SNR values over the cut threshold
        mask_res = res_main_hf_snr > res_snr_cut
        mask_res = dilation(mask_res)

        cube_res_masked = cube_res.with_mask(~mask_res)
    else:
        # no masking
        cube_res_masked = cube_res

    cube_res_masked = cube_res_masked.with_spectral_unit(u.km / u.s, velocity_convention='radio')

    return cube_res_masked




def get_best_2comp_residual(reg):
    modbest = get_best_2comp_model(reg)
    best_res = reg.ucube.cube._data - modbest
    res_cube = SpectralCube(best_res, reg.ucube.cube.wcs)
    res_cube = res_cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    return res_cube


def get_best_2comp_snr_mod(reg):
    modbest = get_best_2comp_model(reg)
    res_cube = get_best_2comp_residual(reg)
    best_rms = UCube.get_rms(res_cube._data)
    return np.nanmax(modbest, axis=0)/best_rms


def get_best_2comp_model(reg):
    # get the log-likelihood between the fits
    lnk21 = reg.ucube.get_AICc_likelihood(2, 1)
    lnk20 = reg.ucube.get_AICc_likelihood(2, 0)
    lnk10 = reg.ucube.get_AICc_likelihood(1, 0)

    mod1 = reg.ucube.pcubes['1'].get_modelcube()
    mod2 = reg.ucube.pcubes['2'].get_modelcube()

    # get the best model based on the calculated likelihood
    modbest = mod1.copy()
    modbest[:] = 0.0

    mask = lnk10 > 5
    modbest[:, mask] = mod1[:, mask]

    mask = np.logical_and(lnk21 > 5, lnk20 > 5)
    modbest[:, mask] = mod2[:, mask]

    return modbest


def replace_para(pcube, pcube_ref, mask):
    # replace values in masked pixels with the reference values
    pcube_ref = pcube_ref.copy('deep')
    pcube.parcube[:,mask] = pcube_ref.parcube[:,mask]
    pcube.errcube[:,mask] = pcube_ref.errcube[:,mask]


def get_skyheader(cube_header):
    # a quick method to convert 3D header to 2D
    from astropy import wcs
    hdr = cube_header
    hdr2D = wcs.WCS(hdr).celestial.to_header()
    hdr2D['NAXIS'] = hdr2D['WCSAXES']
    hdr2D['NAXIS1'] = hdr['NAXIS1']
    hdr2D['NAXIS2'] = hdr['NAXIS2']
    return hdr2D



