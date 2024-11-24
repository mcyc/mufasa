# This file is auto-generated. Edit descriptions and structure as needed.

API_REFERENCE = {
    'mufasa.UltraCube': {
        'module': 'mufasa.UltraCube',
        'description': 'No description available.',
        'members': [
            {'name': 'UCubePlus', 'type': 'class'},
            {'name': 'UltraCube', 'type': 'class'},
            {'name': 'calc_AICc', 'type': 'function'},
            {'name': 'calc_AICc_likelihood', 'type': 'function'},
            {'name': 'calc_chisq', 'type': 'function'},
            {'name': 'calc_rss', 'type': 'function'},
            {'name': 'convolve_sky_byfactor', 'type': 'function'},
            {'name': 'expand_mask', 'type': 'function'},
            {'name': 'fit_cube', 'type': 'function'},
            {'name': 'get_Tpeak', 'type': 'function'},
            {'name': 'get_all_lnk_maps', 'type': 'function'},
            {'name': 'get_best_2c_parcube', 'type': 'function'},
            {'name': 'get_chisq', 'type': 'function'},
            {'name': 'get_masked_moment', 'type': 'function'},
            {'name': 'get_residual', 'type': 'function'},
            {'name': 'get_rms', 'type': 'function'},
            {'name': 'get_rss', 'type': 'function'},
            {'name': 'is_K', 'type': 'function'},
            {'name': 'load_model_fit', 'type': 'function'},
            {'name': 'save_fit', 'type': 'function'},
            {'name': 'to_K', 'type': 'function'}
        ]
    },
    'mufasa._version': {
        'module': 'mufasa._version',
        'description': 'No description available.',
        'members': [

        ]
    },
    'mufasa.aic': {
        'module': 'mufasa.aic',
        'description': 'No description available.',
        'members': [
            {'name': 'AIC', 'type': 'function'},
            {'name': 'AICc', 'type': 'function'},
            {'name': 'fits_comp_AICc', 'type': 'function'},
            {'name': 'fits_comp_chisq', 'type': 'function'},
            {'name': 'get_comp_AICc', 'type': 'function'},
            {'name': 'likelihood', 'type': 'function'}
        ]
    },
    'mufasa.clean_fits': {
        'module': 'mufasa.clean_fits',
        'description': 'No description available.',
        'members': [
            {'name': 'fit_results', 'type': 'class'},
            {'name': 'above_ErrV_Thresh', 'type': 'function'},
            {'name': 'clean_2comp_maps', 'type': 'function'},
            {'name': 'exclusive_2comp_maps', 'type': 'function'},
            {'name': 'extremeV_mask', 'type': 'function'},
            {'name': 'remove_zeros', 'type': 'function'}
        ]
    },
    'mufasa.convolve_tools': {
        'module': 'mufasa.convolve_tools',
        'description': 'No description available.',
        'members': [
            {'name': 'convolve_sky', 'type': 'function'},
            {'name': 'convolve_sky_byfactor', 'type': 'function'},
            {'name': 'edge_trim', 'type': 'function'},
            {'name': 'get_celestial_hdr', 'type': 'function'},
            {'name': 'regrid', 'type': 'function'},
            {'name': 'regrid_mask', 'type': 'function'},
            {'name': 'snr_mask', 'type': 'function'}
        ]
    },
    'mufasa.deblend_cube': {
        'module': 'mufasa.deblend_cube',
        'description': 'No description available.',
        'members': [
            {'name': 'deblend', 'type': 'function'}
        ]
    },
    'mufasa.exceptions': {
        'module': 'mufasa.exceptions',
        'description': 'No description available.',
        'members': [
            {'name': 'FitTypeError', 'type': 'class'},
            {'name': 'SNRMaskError', 'type': 'class'},
            {'name': 'StartFitError', 'type': 'class'}
        ]
    },
    'mufasa.guess_refine': {
        'module': 'mufasa.guess_refine',
        'description': 'No description available.',
        'members': [
            {'name': 'get_celestial_hdr', 'type': 'function'},
            {'name': 'guess_from_cnvpara', 'type': 'function'},
            {'name': 'mask_cleaning', 'type': 'function'},
            {'name': 'mask_swap_2comp', 'type': 'function'},
            {'name': 'master_mask', 'type': 'function'},
            {'name': 'quick_2comp_sort', 'type': 'function'},
            {'name': 'refine_2c_guess', 'type': 'function'},
            {'name': 'refine_each_comp', 'type': 'function'},
            {'name': 'refine_guess', 'type': 'function'},
            {'name': 'regrid', 'type': 'function'},
            {'name': 'save_guesses', 'type': 'function'},
            {'name': 'simple_para_clean', 'type': 'function'},
            {'name': 'tautex_renorm', 'type': 'function'}
        ]
    },
    'mufasa.master_fitter': {
        'module': 'mufasa.master_fitter',
        'description': 'No description available.',
        'members': [
            {'name': 'Region', 'type': 'class'},
            {'name': 'expand_fits', 'type': 'function'},
            {'name': 'fit_best_2comp_residual_cnv', 'type': 'function'},
            {'name': 'fit_surroundings', 'type': 'function'},
            {'name': 'get_2comp_wide_guesses', 'type': 'function'},
            {'name': 'get_best_2comp_model', 'type': 'function'},
            {'name': 'get_best_2comp_residual', 'type': 'function'},
            {'name': 'get_best_2comp_residual_SpectralCube', 'type': 'function'},
            {'name': 'get_best_2comp_residual_cnv', 'type': 'function'},
            {'name': 'get_best_2comp_snr_mod', 'type': 'function'},
            {'name': 'get_convolved_cube', 'type': 'function'},
            {'name': 'get_convolved_fits', 'type': 'function'},
            {'name': 'get_fits', 'type': 'function'},
            {'name': 'get_local_bad', 'type': 'function'},
            {'name': 'get_marginal_pix', 'type': 'function'},
            {'name': 'get_refit_guesses', 'type': 'function'},
            {'name': 'get_skyheader', 'type': 'function'},
            {'name': 'iter_2comp_fit', 'type': 'function'},
            {'name': 'master_2comp_fit', 'type': 'function'},
            {'name': 'refit_2comp_wide', 'type': 'function'},
            {'name': 'refit_bad_2comp', 'type': 'function'},
            {'name': 'refit_marginal', 'type': 'function'},
            {'name': 'refit_swap_2comp', 'type': 'function'},
            {'name': 'replace_bad_pix', 'type': 'function'},
            {'name': 'replace_para', 'type': 'function'},
            {'name': 'replace_rss', 'type': 'function'},
            {'name': 'save_best_2comp_fit', 'type': 'function'},
            {'name': 'save_map', 'type': 'function'},
            {'name': 'save_updated_paramaps', 'type': 'function'},
            {'name': 'standard_2comp_fit', 'type': 'function'}
        ]
    },
    'mufasa.moment_guess': {
        'module': 'mufasa.moment_guess',
        'description': 'No description available.',
        'members': [
            {'name': 'LineSetup', 'type': 'class'},
            {'name': 'adaptive_moment_maps', 'type': 'function'},
            {'name': 'get_rms_prefit', 'type': 'function'},
            {'name': 'get_tau', 'type': 'function'},
            {'name': 'get_tex', 'type': 'function'},
            {'name': 'get_window_slab', 'type': 'function'},
            {'name': 'master_guess', 'type': 'function'},
            {'name': 'mom_guess_wide_sep', 'type': 'function'},
            {'name': 'moment_guesses', 'type': 'function'},
            {'name': 'moment_guesses_1c', 'type': 'function'},
            {'name': 'noisemask_moment', 'type': 'function'},
            {'name': 'peakT', 'type': 'function'},
            {'name': 'vmask_cube', 'type': 'function'},
            {'name': 'vmask_moments', 'type': 'function'},
            {'name': 'window_mask_pcube', 'type': 'function'},
            {'name': 'window_moments', 'type': 'function'}
        ]
    },
    'mufasa.multi_v_fit': {
        'module': 'mufasa.multi_v_fit',
        'description': 'No description available.',
        'members': [
            {'name': 'cubefit_gen', 'type': 'function'},
            {'name': 'cubefit_simp', 'type': 'function'},
            {'name': 'default_masking', 'type': 'function'},
            {'name': 'get_chisq', 'type': 'function'},
            {'name': 'get_start_point', 'type': 'function'},
            {'name': 'get_vstats', 'type': 'function'},
            {'name': 'handle_snr', 'type': 'function'},
            {'name': 'make_header', 'type': 'function'},
            {'name': 'match_pcube_mask', 'type': 'function'},
            {'name': 'register_pcube', 'type': 'function'},
            {'name': 'retry_fit', 'type': 'function'},
            {'name': 'save_guesses', 'type': 'function'},
            {'name': 'save_pcube', 'type': 'function'},
            {'name': 'set_pyspeckit_verbosity', 'type': 'function'},
            {'name': 'snr_estimate', 'type': 'function'}
        ]
    },
    'mufasa.signals': {
        'module': 'mufasa.signals',
        'description': 'No description available.',
        'members': [
            {'name': 'estimate_mode', 'type': 'function'},
            {'name': 'get_moments', 'type': 'function'},
            {'name': 'get_rms_robust', 'type': 'function'},
            {'name': 'get_signal_mask', 'type': 'function'},
            {'name': 'get_snr', 'type': 'function'},
            {'name': 'get_v_at_peak', 'type': 'function'},
            {'name': 'get_v_mask', 'type': 'function'},
            {'name': 'refine_rms', 'type': 'function'},
            {'name': 'refine_signal_mask', 'type': 'function'},
            {'name': 'trim_cube_edge', 'type': 'function'},
            {'name': 'trim_edge', 'type': 'function'},
            {'name': 'v_estimate', 'type': 'function'}
        ]
    },
    'mufasa.slab_sort': {
        'module': 'mufasa.slab_sort',
        'description': 'No description available.',
        'members': [
            {'name': 'distance_metric', 'type': 'function'},
            {'name': 'mask_swap_2comp', 'type': 'function'},
            {'name': 'quick_2comp_sort', 'type': 'function'},
            {'name': 'refmap_2c_mask', 'type': 'function'},
            {'name': 'sort_2comp', 'type': 'function'}
        ]
    },
    'mufasa.spec_models': {
        'module': 'mufasa.spec_models',
        'description': 'No description available.',
        'members': [
            {'name': 'mufasa.spec_models.ammonia_multiv', 'type': 'module'},
            {'name': 'mufasa.spec_models.meta_model', 'type': 'module'},
            {'name': 'mufasa.spec_models.n2hp_constants', 'type': 'module'},
            {'name': 'mufasa.spec_models.n2hp_deblended', 'type': 'module'},
            {'name': 'mufasa.spec_models.n2hp_multiv', 'type': 'module'},
            {'name': 'mufasa.spec_models.nh3_deblended', 'type': 'module'}
        ]
    },
    'mufasa.spec_models.ammonia_multiv': {
        'module': 'mufasa.spec_models.ammonia_multiv',
        'description': 'No description available.',
        'members': [
            {'name': 'T_antenna', 'type': 'function'},
            {'name': '_ammonia_spectrum', 'type': 'function'},
            {'name': 'ammonia_multi_v', 'type': 'function'},
            {'name': 'nh3_multi_v_model_generator', 'type': 'function'}
        ]
    },
    'mufasa.spec_models.meta_model': {
        'module': 'mufasa.spec_models.meta_model',
        'description': 'No description available.',
        'members': [
            {'name': 'MetaModel', 'type': 'class'}
        ]
    },
    'mufasa.spec_models.n2hp_constants': {
        'module': 'mufasa.spec_models.n2hp_constants',
        'description': 'No description available.',
        'members': [

        ]
    },
    'mufasa.spec_models.n2hp_deblended': {
        'module': 'mufasa.spec_models.n2hp_deblended',
        'description': 'No description available.',
        'members': [
            {'name': 'n2hp_vtau_singlemodel_deblended', 'type': 'function'}
        ]
    },
    'mufasa.spec_models.n2hp_multiv': {
        'module': 'mufasa.spec_models.n2hp_multiv',
        'description': 'No description available.',
        'members': [
            {'name': 'T_antenna', 'type': 'function'},
            {'name': '_n2hp_spectrum', 'type': 'function'},
            {'name': 'n2hp_multi_v', 'type': 'function'},
            {'name': 'n2hp_multi_v_model_generator', 'type': 'function'}
        ]
    },
    'mufasa.spec_models.nh3_deblended': {
        'module': 'mufasa.spec_models.nh3_deblended',
        'description': 'No description available.',
        'members': [
            {'name': 'nh3_vtau_singlemodel_deblended', 'type': 'function'}
        ]
    },
    'mufasa.utils': {
        'module': 'mufasa.utils',
        'description': 'No description available.',
        'members': [
            {'name': 'mufasa.utils.dataframe', 'type': 'module'},
            {'name': 'mufasa.utils.interpolate', 'type': 'module'},
            {'name': 'mufasa.utils.map_divide', 'type': 'module'},
            {'name': 'mufasa.utils.mufasa_log', 'type': 'module'},
            {'name': 'mufasa.utils.multicore', 'type': 'module'},
            {'name': 'mufasa.utils.neighbours', 'type': 'module'}
        ]
    },
    'mufasa.utils.dataframe': {
        'module': 'mufasa.utils.dataframe',
        'description': 'No description available.',
        'members': [
            {'name': 'assign_to_dataframe', 'type': 'function'},
            {'name': 'make_dataframe', 'type': 'function'},
            {'name': 'read', 'type': 'function'}
        ]
    },
    'mufasa.utils.interpolate': {
        'module': 'mufasa.utils.interpolate',
        'description': 'No description available.',
        'members': [
            {'name': 'expand_interpolate', 'type': 'function'},
            {'name': 'iter_expand', 'type': 'function'}
        ]
    },
    'mufasa.utils.map_divide': {
        'module': 'mufasa.utils.map_divide',
        'description': 'No description available.',
        'members': [
            {'name': 'dist_divide', 'type': 'function'},
            {'name': 'watershed_divide', 'type': 'function'}
        ]
    },
    'mufasa.utils.mufasa_log': {
        'module': 'mufasa.utils.mufasa_log',
        'description': 'No description available.',
        'members': [
            {'name': 'OriginContextFilter', 'type': 'class'},
            {'name': 'WarningContextFilter', 'type': 'class'},
            {'name': 'get_logger', 'type': 'function'},
            {'name': 'init_logging', 'type': 'function'},
            {'name': 'reset_logger', 'type': 'function'}
        ]
    },
    'mufasa.utils.multicore': {
        'module': 'mufasa.utils.multicore',
        'description': 'No description available.',
        'members': [
            {'name': 'validate_n_cores', 'type': 'function'}
        ]
    },
    'mufasa.utils.neighbours': {
        'module': 'mufasa.utils.neighbours',
        'description': 'No description available.',
        'members': [
            {'name': 'disk_neighbour', 'type': 'function'},
            {'name': 'get_neighbor_coord', 'type': 'function'},
            {'name': 'get_valid_neighbors', 'type': 'function'},
            {'name': 'maxref_neighbor_coords', 'type': 'function'},
            {'name': 'square_neighbour', 'type': 'function'}
        ]
    },
    'mufasa.visualization': {
        'module': 'mufasa.visualization',
        'description': 'No description available.',
        'members': [
            {'name': 'mufasa.visualization.scatter_3D', 'type': 'module'},
            {'name': 'mufasa.visualization.spec_viz', 'type': 'module'}
        ]
    },
    'mufasa.visualization.scatter_3D': {
        'module': 'mufasa.visualization.scatter_3D',
        'description': 'No description available.',
        'members': [
            {'name': 'ScatterPPV', 'type': 'class'},
            {'name': 'scatter_3D', 'type': 'function'},
            {'name': 'scatter_3D_df', 'type': 'function'}
        ]
    },
    'mufasa.visualization.spec_viz': {
        'module': 'mufasa.visualization.spec_viz',
        'description': 'No description available.',
        'members': [
            {'name': 'Plotter', 'type': 'class'},
            {'name': 'ensure_units_compatible', 'type': 'function'},
            {'name': 'get_cube_slab', 'type': 'function'},
            {'name': 'get_spec_grid', 'type': 'function'},
            {'name': 'plot_fits_grid', 'type': 'function'},
            {'name': 'plot_model', 'type': 'function'},
            {'name': 'plot_spec', 'type': 'function'},
            {'name': 'plot_spec_grid', 'type': 'function'},
            {'name': 'strip_units', 'type': 'function'}
        ]
    },
}
