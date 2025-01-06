# This file is auto-generated. Edit descriptions and structure as needed.

API_REFERENCE = {
    'mufasa.PCube': {
        'module': 'mufasa.PCube',
        'description': '''Undocumented''',
        'members': [
            {'name': 'PCube', 'type': 'class', 'description': '''A specialized subclass of :class:`pyspeckit.'''}
        ]
    },
    'mufasa.UltraCube': {
        'module': 'mufasa.UltraCube',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'UCubePlus', 'type': 'class', 'description': '''A subclass of UltraCube that includes directory management for parameter maps and model fits.'''},
            {'name': 'UltraCube', 'type': 'class', 'description': '''A framework for multi-component spectral cube analysis and model fitting.'''},
            {'name': 'calc_AICc', 'type': 'function', 'description': '''Calculate the corrected Akaike Information Criterion (AICc) for a spectral cube model.'''},
            {'name': 'calc_AICc_likelihood', 'type': 'function', 'description': '''Calculate the relative likelihood of two models based on their AICc values.'''},
            {'name': 'calc_chisq', 'type': 'function', 'description': '''Calculate the chi-squared (χ²) or reduced chi-squared value for a spectral cube model fit.'''},
            {'name': 'calc_rss', 'type': 'function', 'description': '''Calculate the residual sum of squares (RSS) for a spectral cube model fit.'''},
            {'name': 'convolve_sky_byfactor', 'type': 'function', 'description': '''Convolve the spatial dimensions of a spectral cube by a specified factor.'''},
            {'name': 'expand_mask', 'type': 'function', 'description': '''Expand a 3D mask along the spectral axis by a specified buffer size.'''},
            {'name': 'fit_cube', 'type': 'function', 'description': '''Fit the spectral cube using the specified fitting type.'''},
            {'name': 'get_Tpeak', 'type': 'function', 'description': '''Calculate the peak value of a model cube at each spatial pixel.'''},
            {'name': 'get_all_lnk_maps', 'type': 'function', 'description': '''Compute log-likelihood ratio maps for model comparisons up to a specified number of components.'''},
            {'name': 'get_best_2c_parcube', 'type': 'function', 'description': '''Select the best 2-component parameter cube based on AICc likelihood thresholds.'''},
            {'name': 'get_chisq', 'type': 'function', 'description': '''Calculate the chi-squared or reduced chi-squared value for a spectral cube.'''},
            {'name': 'get_masked_moment', 'type': 'function', 'description': '''Calculate a masked moment of a spectral cube.'''},
            {'name': 'get_residual', 'type': 'function', 'description': '''Calculate the residual between the data cube and the model cube.'''},
            {'name': 'get_rms', 'type': 'function', 'description': '''Compute a robust estimate of the root mean square (RMS) from the fit residuals.'''},
            {'name': 'get_rss', 'type': 'function', 'description': '''Calculate the residual sum of squares (RSS) for a spectral cube model fit.'''},
            {'name': 'is_K', 'type': 'function', 'description': '''Check if a given unit is equivalent to Kelvin (K).'''},
            {'name': 'load_model_fit', 'type': 'function', 'description': '''Load the spectral fit results from a.'''},
            {'name': 'save_fit', 'type': 'function', 'description': '''Save the fitted parameter cube to a.'''},
            {'name': 'to_K', 'type': 'function', 'description': '''Convert the unit of a spectral cube to Kelvin (K).'''}
        ]
    },
    'mufasa.aic': {
        'module': 'mufasa.aic',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'AIC', 'type': 'function', 'description': '''Calculate the Akaike Information Criterion (AIC).'''},
            {'name': 'AICc', 'type': 'function', 'description': '''Calculate the corrected Akaike Information Criterion (AICc).'''},
            {'name': 'fits_comp_AICc', 'type': 'function', 'description': '''A wrapper function to calculate corrected Akaike Information Criterion (AICc) values.'''},
            {'name': 'fits_comp_chisq', 'type': 'function', 'description': '''Calculate and save chi-squared values for the given cube and model fits.'''},
            {'name': 'get_comp_AICc', 'type': 'function', 'description': '''Calculate AICc values for two models over the same samples.'''},
            {'name': 'likelihood', 'type': 'function', 'description': '''Calculate the log-likelihood of model A relative to model B.'''}
        ]
    },
    'mufasa.clean_fits': {
        'module': 'mufasa.clean_fits',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'fit_results', 'type': 'class', 'description': '''Undocumented'''},
            {'name': 'above_ErrV_Thresh', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'clean_2comp_maps', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'exclusive_2comp_maps', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'extremeV_mask', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'remove_zeros', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.convolve_tools': {
        'module': 'mufasa.convolve_tools',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'convolve_sky', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'convolve_sky_byfactor', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'edge_trim', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_celestial_hdr', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'regrid', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'regrid_mask', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'snr_mask', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.deblend_cube': {
        'module': 'mufasa.deblend_cube',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'deblend', 'type': 'function', 'description': '''Deblend hyperfine structures in a cube based on fitted models.'''}
        ]
    },
    'mufasa.exceptions': {
        'module': 'mufasa.exceptions',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'FitTypeError', 'type': 'class', 'description': '''Fitttype provided is not valid.'''},
            {'name': 'SNRMaskError', 'type': 'class', 'description': '''SNR Mask has no valid pixel.'''},
            {'name': 'StartFitError', 'type': 'class', 'description': '''Fitting failed from the beginning.'''}
        ]
    },
    'mufasa.guess_refine': {
        'module': 'mufasa.guess_refine',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'get_celestial_hdr', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'guess_from_cnvpara', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'mask_cleaning', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'mask_swap_2comp', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'master_mask', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'quick_2comp_sort', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'refine_2c_guess', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'refine_each_comp', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'refine_guess', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'regrid', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'save_guesses', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'simple_para_clean', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'tautex_renorm', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.master_fitter': {
        'module': 'mufasa.master_fitter',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'Region', 'type': 'class', 'description': '''A class to represent the observed spectral cube to perform the model fits.'''},
            {'name': 'expand_fits', 'type': 'function', 'description': '''Expand fits in a region by incrementally fitting pixels beyond a defined.'''},
            {'name': 'fit_best_2comp_residual_cnv', 'type': 'function', 'description': '''Fit the convolved residual of the best-fit two-component spectral model.'''},
            {'name': 'fit_surroundings', 'type': 'function', 'description': '''Expand fits around a region based on model log-likelihood thresholds and.'''},
            {'name': 'get_2comp_wide_guesses', 'type': 'function', 'description': '''Generate initial guesses for fitting a two-component spectral model with.'''},
            {'name': 'get_best_2comp_model', 'type': 'function', 'description': '''Retrieve the best-fit model cube for the given Region object.'''},
            {'name': 'get_best_2comp_residual', 'type': 'function', 'description': '''Calculate the residual cube for the best-fit two-component model.'''},
            {'name': 'get_best_2comp_residual_SpectralCube', 'type': 'function', 'description': '''Generate the residual spectral cube for the best-fit two-component.'''},
            {'name': 'get_best_2comp_residual_cnv', 'type': 'function', 'description': '''Generate a convolved residual cube for the best-fit two-component.'''},
            {'name': 'get_best_2comp_snr_mod', 'type': 'function', 'description': '''Calculate the signal-to-noise ratio (SNR) map for the best-fit two-.'''},
            {'name': 'get_convolved_cube', 'type': 'function', 'description': '''Generate and save a convolved version of the spectral cube for a given.'''},
            {'name': 'get_convolved_fits', 'type': 'function', 'description': '''Fit a model to the convolved cube and save the results.'''},
            {'name': 'get_fits', 'type': 'function', 'description': '''Fit a model to the original spectral cube and save the results.'''},
            {'name': 'get_local_bad', 'type': 'function', 'description': '''Identify local pixels with significantly lower relative log-likelihood.'''},
            {'name': 'get_marginal_pix', 'type': 'function', 'description': '''Return pixels at the edge of structures with values greater than.'''},
            {'name': 'get_refit_guesses', 'type': 'function', 'description': '''Generate initial guesses for refitting based on neighboring pixels or.'''},
            {'name': 'get_skyheader', 'type': 'function', 'description': '''Generate a 2D sky projection header from a 3D spectral cube header.'''},
            {'name': 'iter_2comp_fit', 'type': 'function', 'description': '''Perform a two-component fit iterantively through two steps.'''},
            {'name': 'master_2comp_fit', 'type': 'function', 'description': '''Perform a two-component fit on the data cube within a Region object.'''},
            {'name': 'refit_2comp_wide', 'type': 'function', 'description': '''Refit pixels to recover compoents with wide velocity separation for.'''},
            {'name': 'refit_bad_2comp', 'type': 'function', 'description': '''Refit pixels with poor 2-component fits, as determined by the log-.'''},
            {'name': 'refit_marginal', 'type': 'function', 'description': '''Refit pixels with fits that appears marginally okay, as deterined by the.'''},
            {'name': 'refit_swap_2comp', 'type': 'function', 'description': '''Refit the cube by using the previous fit result as guesses, but with the.'''},
            {'name': 'replace_bad_pix', 'type': 'function', 'description': '''Refit pixels marked by the mask as "bad" and adopt the new model if it.'''},
            {'name': 'replace_para', 'type': 'function', 'description': '''Replace parameter values in a parameter cube with those from a reference.'''},
            {'name': 'replace_rss', 'type': 'function', 'description': '''Replace RSS-related maps in a `UltraCube` object for specific.'''},
            {'name': 'save_best_2comp_fit', 'type': 'function', 'description': '''Save the best two-component fit results for the specified region.'''},
            {'name': 'save_map', 'type': 'function', 'description': '''Save a 2D map as a FITS file.'''},
            {'name': 'save_updated_paramaps', 'type': 'function', 'description': '''Save the updated parameter maps for specified components.'''},
            {'name': 'standard_2comp_fit', 'type': 'function', 'description': '''Perform a two-component fit for the cube using default moment map.'''}
        ]
    },
    'mufasa.moment_guess': {
        'module': 'mufasa.moment_guess',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'LineSetup', 'type': 'class', 'description': '''Undocumented'''},
            {'name': 'adaptive_moment_maps', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_rms_prefit', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_tau', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_tex', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_window_slab', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'master_guess', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'mom_guess_wide_sep', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'moment_guesses', 'type': 'function', 'description': '''Generate reasonable initial guesses for multiple component fits based on moment maps.'''},
            {'name': 'moment_guesses_1c', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'noisemask_moment', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'peakT', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'vmask_cube', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'vmask_moments', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'window_mask_pcube', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'window_moments', 'type': 'function', 'description': '''Calculate the zeroth, first, and second moments of a spectrum or cube.'''}
        ]
    },
    'mufasa.multi_v_fit': {
        'module': 'mufasa.multi_v_fit',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'cubefit_gen', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'cubefit_simp', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'default_masking', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_chisq', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_start_point', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_vstats', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'handle_snr', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'make_header', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'match_pcube_mask', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'register_pcube', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'retry_fit', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'save_guesses', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'save_pcube', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'set_pyspeckit_verbosity', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'snr_estimate', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.signals': {
        'module': 'mufasa.signals',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'estimate_mode', 'type': 'function', 'description': '''Estimate the mode of the data using a histogram.'''},
            {'name': 'get_moments', 'type': 'function', 'description': '''Calculate moments of the signals in a cube.'''},
            {'name': 'get_rms_robust', 'type': 'function', 'description': '''Make a robust RMS estimate.'''},
            {'name': 'get_signal_mask', 'type': 'function', 'description': '''Provide a 3D mask indicating signal regions based on RMS and SNR threshold.'''},
            {'name': 'get_snr', 'type': 'function', 'description': '''Calculate the peak signal-to-noise ratio of the cube.'''},
            {'name': 'get_v_at_peak', 'type': 'function', 'description': '''Find the velocity corresponding to the peak emission.'''},
            {'name': 'get_v_mask', 'type': 'function', 'description': '''Return a mask centered on a reference velocity with a spectral window.'''},
            {'name': 'refine_rms', 'type': 'function', 'description': '''Refine the RMS estimate by masking out signal regions.'''},
            {'name': 'refine_signal_mask', 'type': 'function', 'description': '''Refine a signal mask by removing noisy features and expanding the mask.'''},
            {'name': 'trim_cube_edge', 'type': 'function', 'description': '''Remove spatial edges from a cube.'''},
            {'name': 'trim_edge', 'type': 'function', 'description': '''Trim edges using a 2D mask.'''},
            {'name': 'v_estimate', 'type': 'function', 'description': '''Estimate the velocity centroid based on peak emission.'''}
        ]
    },
    'mufasa.slab_sort': {
        'module': 'mufasa.slab_sort',
        'description': '''The `mufasa.''',
        'members': [
            {'name': 'distance_metric', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'mask_swap_2comp', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'quick_2comp_sort', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'refmap_2c_mask', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'sort_2comp', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.spec_models': {
        'module': 'mufasa.spec_models',
        'description': '''This sub-package hosts the codes for sepctral models.''',
        'members': [
            {'name': 'mufasa.spec_models.ammonia_multiv', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.spec_models.meta_model', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.spec_models.n2hp_constants', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.spec_models.n2hp_deblended', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.spec_models.n2hp_multiv', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.spec_models.nh3_deblended', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'mufasa.spec_models.ammonia_multiv': {
        'module': 'mufasa.spec_models.ammonia_multiv',
        'description': '''Undocumented''',
        'members': [
            {'name': 'T_antenna', 'type': 'function', 'description': '''Calculate antenna temperatures over nu (in GHz).'''},
            {'name': 'ammonia_multi_v', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'nh3_multi_v_model_generator', 'type': 'function', 'description': '''Works for up to 2 componet fits at the moment.'''}
        ]
    },
    'mufasa.spec_models.meta_model': {
        'module': 'mufasa.spec_models.meta_model',
        'description': '''Undocumented''',
        'members': [
            {'name': 'MetaModel', 'type': 'class', 'description': '''A class to store spectral model-specific information relevant to spectral modeling tasks, such as fitting.'''}
        ]
    },
    'mufasa.spec_models.n2hp_constants': {
        'module': 'mufasa.spec_models.n2hp_constants',
        'description': '''Undocumented''',
        'members': [

        ]
    },
    'mufasa.spec_models.n2hp_deblended': {
        'module': 'mufasa.spec_models.n2hp_deblended',
        'description': '''Undocumented''',
        'members': [
            {'name': 'n2hp_vtau_singlemodel_deblended', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.spec_models.n2hp_multiv': {
        'module': 'mufasa.spec_models.n2hp_multiv',
        'description': '''Undocumented''',
        'members': [
            {'name': 'T_antenna', 'type': 'function', 'description': '''Calculate antenna temperatures over nu (in GHz).'''},
            {'name': 'n2hp_multi_v', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'n2hp_multi_v_model_generator', 'type': 'function', 'description': '''Works for up to 2 component fits at the moment.'''}
        ]
    },
    'mufasa.spec_models.nh3_deblended': {
        'module': 'mufasa.spec_models.nh3_deblended',
        'description': '''Undocumented''',
        'members': [
            {'name': 'nh3_vtau_singlemodel_deblended', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.utils': {
        'module': 'mufasa.utils',
        'description': '''This sub-package provides utility functions and tools for data processing,.''',
        'members': [
            {'name': 'mufasa.utils.dask_utils', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.dataframe', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.interpolate', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.map_divide', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.memory', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.mufasa_log', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.multicore', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.utils.neighbours', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'mufasa.utils.dask_utils': {
        'module': 'mufasa.utils.dask_utils',
        'description': '''Undocumented''',
        'members': [
            {'name': 'calculate_batch_size', 'type': 'function', 'description': '''Dynamically calculate the optimal batch size for processing valid pixels.'''},
            {'name': 'calculate_chunks', 'type': 'function', 'description': '''Calculate chunk sizes for a Dask array based on the given criteria.'''},
            {'name': 'lazy_pix_compute', 'type': 'function', 'description': '''Lazily compute values for each valid pixel specified by the isvalid mask.'''},
            {'name': 'lazy_pix_compute_multiprocessing', 'type': 'function', 'description': '''Lazily compute values for valid pixels specified by the isvalid mask using batch processing.'''}
        ]
    },
    'mufasa.utils.dataframe': {
        'module': 'mufasa.utils.dataframe',
        'description': '''Undocumented''',
        'members': [
            {'name': 'assign_to_dataframe', 'type': 'function', 'description': '''Assign values from a new data array to an existing DataFrame based on spatial coordinates and component index.'''},
            {'name': 'make_dataframe', 'type': 'function', 'description': '''Create a DataFrame from a 3D parameter array, applying optional velocity and error thresholds.'''},
            {'name': 'read', 'type': 'function', 'description': '''Read a FITS file and convert the data to a pandas DataFrame, optionally including the header.'''}
        ]
    },
    'mufasa.utils.interpolate': {
        'module': 'mufasa.utils.interpolate',
        'description': '''Undocumented''',
        'members': [
            {'name': 'expand_interpolate', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'iter_expand', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.utils.map_divide': {
        'module': 'mufasa.utils.map_divide',
        'description': '''Undocumented''',
        'members': [
            {'name': 'dist_divide', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'watershed_divide', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.utils.memory': {
        'module': 'mufasa.utils.memory',
        'description': '''Undocumented''',
        'members': [
            {'name': 'calculate_target_memory', 'type': 'function', 'description': '''Calculate the target memory per chunk based on system memory and the number of cores.'''}
        ]
    },
    'mufasa.utils.mufasa_log': {
        'module': 'mufasa.utils.mufasa_log',
        'description': '''Undocumented''',
        'members': [
            {'name': 'OriginContextFilter', 'type': 'class', 'description': '''Filter instances are used to perform arbitrary filtering of LogRecords.'''},
            {'name': 'WarningContextFilter', 'type': 'class', 'description': '''Filter instances are used to perform arbitrary filtering of LogRecords.'''},
            {'name': 'get_logger', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'init_logging', 'type': 'function', 'description': ''':param logfile: file to save to (default mufasa.'''},
            {'name': 'reset_logger', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.utils.multicore': {
        'module': 'mufasa.utils.multicore',
        'description': '''Undocumented''',
        'members': [
            {'name': 'validate_n_cores', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.utils.neighbours': {
        'module': 'mufasa.utils.neighbours',
        'description': '''Undocumented''',
        'members': [
            {'name': 'disk_neighbour', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_neighbor_coord', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_valid_neighbors', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'maxref_neighbor_coords', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'square_neighbour', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'mufasa.visualization': {
        'module': 'mufasa.visualization',
        'description': '''This sub-package provides tools for visualizing spectral cubes,.''',
        'members': [
            {'name': 'mufasa.visualization.scatter_3D', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'mufasa.visualization.spec_viz', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'mufasa.visualization.scatter_3D': {
        'module': 'mufasa.visualization.scatter_3D',
        'description': '''Undocumented''',
        'members': [
            {'name': 'ScatterPPV', 'type': 'class', 'description': '''A class to plot the fitted parameters in 3D scatter plots.'''},
            {'name': 'scatter_3D', 'type': 'function', 'description': '''Plot a 3D scatter plot with optional opacity scaling for point ranges.'''},
            {'name': 'scatter_3D_df', 'type': 'function', 'description': '''A wrapper for scatter_3D to quickly plot a pandas DataFrame in 3D.'''}
        ]
    },
    'mufasa.visualization.spec_viz': {
        'module': 'mufasa.visualization.spec_viz',
        'description': '''Undocumented''',
        'members': [
            {'name': 'Plotter', 'type': 'class', 'description': '''Undocumented'''},
            {'name': 'ensure_units_compatible', 'type': 'function', 'description': '''Ensure the limits have compatible units with the data, converting if needed.'''},
            {'name': 'get_cube_slab', 'type': 'function', 'description': '''Extract a spectral slab from the cube over the specified velocity range.'''},
            {'name': 'get_spec_grid', 'type': 'function', 'description': '''Create a grid of subplots for spectra.'''},
            {'name': 'plot_fits_grid', 'type': 'function', 'description': '''Plot a grid of model fits from the cube centered at (x, y).'''},
            {'name': 'plot_model', 'type': 'function', 'description': '''Plot a model fit for a spectrum.'''},
            {'name': 'plot_spec', 'type': 'function', 'description': '''Plot a spectrum.'''},
            {'name': 'plot_spec_grid', 'type': 'function', 'description': '''Plot a grid of spectra from the cube centered at (x, y).'''},
            {'name': 'strip_units', 'type': 'function', 'description': '''Helper function to strip units from a limit tuple if it contains Quantity.'''}
        ]
    },
}
