:html_theme.sidebar_secondary.remove:

.. _api_ref:

=============
API Reference
=============

This is the class and function reference of **MUFASA**.

.. toctree::
   :maxdepth: 2
   :hidden:

   mufasa.UltraCube
   mufasa.aic
   mufasa.clean_fits
   mufasa.convolve_tools
   mufasa.deblend_cube
   mufasa.exceptions
   mufasa.guess_refine
   mufasa.master_fitter
   mufasa.moment_guess
   mufasa.multi_v_fit
   mufasa.signals
   mufasa.slab_sort
   mufasa.spec_models
   mufasa.utils
   mufasa.visualization
   

.. list-table::
   :header-rows: 1
   :class: apisearch-table

   * - Object
     - Description
   * - :obj:`~mufasa.UltraCube.UCubePlus`
     - No description available.
   * - :obj:`~mufasa.UltraCube.UltraCube`
     - No description available.
   * - :obj:`~mufasa.UltraCube.calc_AICc`
     - No description available.
   * - :obj:`~mufasa.UltraCube.calc_AICc_likelihood`
     - No description available.
   * - :obj:`~mufasa.UltraCube.calc_chisq`
     - No description available.
   * - :obj:`~mufasa.UltraCube.calc_rss`
     - No description available.
   * - :obj:`~mufasa.UltraCube.convolve_sky_byfactor`
     - No description available.
   * - :obj:`~mufasa.UltraCube.expand_mask`
     - No description available.
   * - :obj:`~mufasa.UltraCube.fit_cube`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_Tpeak`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_all_lnk_maps`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_best_2c_parcube`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_chisq`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_masked_moment`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_residual`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_rms`
     - No description available.
   * - :obj:`~mufasa.UltraCube.get_rss`
     - No description available.
   * - :obj:`~mufasa.UltraCube.is_K`
     - No description available.
   * - :obj:`~mufasa.UltraCube.load_model_fit`
     - No description available.
   * - :obj:`~mufasa.UltraCube.save_fit`
     - No description available.
   * - :obj:`~mufasa.UltraCube.to_K`
     - No description available.
   * - :obj:`~mufasa.aic.AIC`
     - No description available.
   * - :obj:`~mufasa.aic.AICc`
     - No description available.
   * - :obj:`~mufasa.aic.fits_comp_AICc`
     - No description available.
   * - :obj:`~mufasa.aic.fits_comp_chisq`
     - No description available.
   * - :obj:`~mufasa.aic.get_comp_AICc`
     - No description available.
   * - :obj:`~mufasa.aic.likelihood`
     - No description available.
   * - :obj:`~mufasa.clean_fits.fit_results`
     - No description available.
   * - :obj:`~mufasa.clean_fits.above_ErrV_Thresh`
     - No description available.
   * - :obj:`~mufasa.clean_fits.clean_2comp_maps`
     - No description available.
   * - :obj:`~mufasa.clean_fits.exclusive_2comp_maps`
     - No description available.
   * - :obj:`~mufasa.clean_fits.extremeV_mask`
     - No description available.
   * - :obj:`~mufasa.clean_fits.remove_zeros`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.convolve_sky`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.convolve_sky_byfactor`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.edge_trim`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.get_celestial_hdr`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.regrid`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.regrid_mask`
     - No description available.
   * - :obj:`~mufasa.convolve_tools.snr_mask`
     - No description available.
   * - :obj:`~mufasa.deblend_cube.deblend`
     - No description available.
   * - :obj:`~mufasa.exceptions.FitTypeError`
     - No description available.
   * - :obj:`~mufasa.exceptions.SNRMaskError`
     - No description available.
   * - :obj:`~mufasa.exceptions.StartFitError`
     - No description available.
   * - :obj:`~mufasa.guess_refine.get_celestial_hdr`
     - No description available.
   * - :obj:`~mufasa.guess_refine.guess_from_cnvpara`
     - No description available.
   * - :obj:`~mufasa.guess_refine.mask_cleaning`
     - No description available.
   * - :obj:`~mufasa.guess_refine.mask_swap_2comp`
     - No description available.
   * - :obj:`~mufasa.guess_refine.master_mask`
     - No description available.
   * - :obj:`~mufasa.guess_refine.quick_2comp_sort`
     - No description available.
   * - :obj:`~mufasa.guess_refine.refine_2c_guess`
     - No description available.
   * - :obj:`~mufasa.guess_refine.refine_each_comp`
     - No description available.
   * - :obj:`~mufasa.guess_refine.refine_guess`
     - No description available.
   * - :obj:`~mufasa.guess_refine.regrid`
     - No description available.
   * - :obj:`~mufasa.guess_refine.save_guesses`
     - No description available.
   * - :obj:`~mufasa.guess_refine.simple_para_clean`
     - No description available.
   * - :obj:`~mufasa.guess_refine.tautex_renorm`
     - No description available.
   * - :obj:`~mufasa.master_fitter.Region`
     - No description available.
   * - :obj:`~mufasa.master_fitter.expand_fits`
     - No description available.
   * - :obj:`~mufasa.master_fitter.fit_best_2comp_residual_cnv`
     - No description available.
   * - :obj:`~mufasa.master_fitter.fit_surroundings`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_2comp_wide_guesses`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_best_2comp_model`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual_SpectralCube`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual_cnv`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_best_2comp_snr_mod`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_convolved_cube`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_convolved_fits`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_fits`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_local_bad`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_marginal_pix`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_refit_guesses`
     - No description available.
   * - :obj:`~mufasa.master_fitter.get_skyheader`
     - No description available.
   * - :obj:`~mufasa.master_fitter.iter_2comp_fit`
     - No description available.
   * - :obj:`~mufasa.master_fitter.master_2comp_fit`
     - No description available.
   * - :obj:`~mufasa.master_fitter.refit_2comp_wide`
     - No description available.
   * - :obj:`~mufasa.master_fitter.refit_bad_2comp`
     - No description available.
   * - :obj:`~mufasa.master_fitter.refit_marginal`
     - No description available.
   * - :obj:`~mufasa.master_fitter.refit_swap_2comp`
     - No description available.
   * - :obj:`~mufasa.master_fitter.replace_bad_pix`
     - No description available.
   * - :obj:`~mufasa.master_fitter.replace_para`
     - No description available.
   * - :obj:`~mufasa.master_fitter.replace_rss`
     - No description available.
   * - :obj:`~mufasa.master_fitter.save_best_2comp_fit`
     - No description available.
   * - :obj:`~mufasa.master_fitter.save_map`
     - No description available.
   * - :obj:`~mufasa.master_fitter.save_updated_paramaps`
     - No description available.
   * - :obj:`~mufasa.master_fitter.standard_2comp_fit`
     - No description available.
   * - :obj:`~mufasa.moment_guess.LineSetup`
     - No description available.
   * - :obj:`~mufasa.moment_guess.adaptive_moment_maps`
     - No description available.
   * - :obj:`~mufasa.moment_guess.get_rms_prefit`
     - No description available.
   * - :obj:`~mufasa.moment_guess.get_tau`
     - No description available.
   * - :obj:`~mufasa.moment_guess.get_tex`
     - No description available.
   * - :obj:`~mufasa.moment_guess.get_window_slab`
     - No description available.
   * - :obj:`~mufasa.moment_guess.master_guess`
     - No description available.
   * - :obj:`~mufasa.moment_guess.mom_guess_wide_sep`
     - No description available.
   * - :obj:`~mufasa.moment_guess.moment_guesses`
     - No description available.
   * - :obj:`~mufasa.moment_guess.moment_guesses_1c`
     - No description available.
   * - :obj:`~mufasa.moment_guess.noisemask_moment`
     - No description available.
   * - :obj:`~mufasa.moment_guess.peakT`
     - No description available.
   * - :obj:`~mufasa.moment_guess.vmask_cube`
     - No description available.
   * - :obj:`~mufasa.moment_guess.vmask_moments`
     - No description available.
   * - :obj:`~mufasa.moment_guess.window_mask_pcube`
     - No description available.
   * - :obj:`~mufasa.moment_guess.window_moments`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.cubefit_gen`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.cubefit_simp`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.default_masking`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.get_chisq`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.get_start_point`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.get_vstats`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.handle_snr`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.make_header`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.match_pcube_mask`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.register_pcube`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.retry_fit`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.save_guesses`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.save_pcube`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.set_pyspeckit_verbosity`
     - No description available.
   * - :obj:`~mufasa.multi_v_fit.snr_estimate`
     - No description available.
   * - :obj:`~mufasa.signals.estimate_mode`
     - No description available.
   * - :obj:`~mufasa.signals.get_moments`
     - No description available.
   * - :obj:`~mufasa.signals.get_rms_robust`
     - No description available.
   * - :obj:`~mufasa.signals.get_signal_mask`
     - No description available.
   * - :obj:`~mufasa.signals.get_snr`
     - No description available.
   * - :obj:`~mufasa.signals.get_v_at_peak`
     - No description available.
   * - :obj:`~mufasa.signals.get_v_mask`
     - No description available.
   * - :obj:`~mufasa.signals.refine_rms`
     - No description available.
   * - :obj:`~mufasa.signals.refine_signal_mask`
     - No description available.
   * - :obj:`~mufasa.signals.trim_cube_edge`
     - No description available.
   * - :obj:`~mufasa.signals.trim_edge`
     - No description available.
   * - :obj:`~mufasa.signals.v_estimate`
     - No description available.
   * - :obj:`~mufasa.slab_sort.distance_metric`
     - No description available.
   * - :obj:`~mufasa.slab_sort.mask_swap_2comp`
     - No description available.
   * - :obj:`~mufasa.slab_sort.quick_2comp_sort`
     - No description available.
   * - :obj:`~mufasa.slab_sort.refmap_2c_mask`
     - No description available.
   * - :obj:`~mufasa.slab_sort.sort_2comp`
     - No description available.
   * - :obj:`~mufasa.spec_models.ammonia_multiv.T_antenna`
     - No description available.
   * - :obj:`~mufasa.spec_models.ammonia_multiv._ammonia_spectrum`
     - No description available.
   * - :obj:`~mufasa.spec_models.ammonia_multiv.ammonia_multi_v`
     - No description available.
   * - :obj:`~mufasa.spec_models.ammonia_multiv.nh3_multi_v_model_generator`
     - No description available.
   * - :obj:`~mufasa.spec_models.meta_model.MetaModel`
     - No description available.
   * - :obj:`~mufasa.spec_models.n2hp_deblended.n2hp_vtau_singlemodel_deblended`
     - No description available.
   * - :obj:`~mufasa.spec_models.n2hp_multiv.T_antenna`
     - No description available.
   * - :obj:`~mufasa.spec_models.n2hp_multiv._n2hp_spectrum`
     - No description available.
   * - :obj:`~mufasa.spec_models.n2hp_multiv.n2hp_multi_v`
     - No description available.
   * - :obj:`~mufasa.spec_models.n2hp_multiv.n2hp_multi_v_model_generator`
     - No description available.
   * - :obj:`~mufasa.spec_models.nh3_deblended.nh3_vtau_singlemodel_deblended`
     - No description available.
   * - :obj:`~mufasa.utils.dataframe.assign_to_dataframe`
     - No description available.
   * - :obj:`~mufasa.utils.dataframe.make_dataframe`
     - No description available.
   * - :obj:`~mufasa.utils.dataframe.read`
     - No description available.
   * - :obj:`~mufasa.utils.interpolate.expand_interpolate`
     - No description available.
   * - :obj:`~mufasa.utils.interpolate.iter_expand`
     - No description available.
   * - :obj:`~mufasa.utils.map_divide.dist_divide`
     - No description available.
   * - :obj:`~mufasa.utils.map_divide.watershed_divide`
     - No description available.
   * - :obj:`~mufasa.utils.mufasa_log.OriginContextFilter`
     - No description available.
   * - :obj:`~mufasa.utils.mufasa_log.WarningContextFilter`
     - No description available.
   * - :obj:`~mufasa.utils.mufasa_log.get_logger`
     - No description available.
   * - :obj:`~mufasa.utils.mufasa_log.init_logging`
     - No description available.
   * - :obj:`~mufasa.utils.mufasa_log.reset_logger`
     - No description available.
   * - :obj:`~mufasa.utils.multicore.validate_n_cores`
     - No description available.
   * - :obj:`~mufasa.utils.neighbours.disk_neighbour`
     - No description available.
   * - :obj:`~mufasa.utils.neighbours.get_neighbor_coord`
     - No description available.
   * - :obj:`~mufasa.utils.neighbours.get_valid_neighbors`
     - No description available.
   * - :obj:`~mufasa.utils.neighbours.maxref_neighbor_coords`
     - No description available.
   * - :obj:`~mufasa.utils.neighbours.square_neighbour`
     - No description available.
   * - :obj:`~mufasa.visualization.scatter_3D.ScatterPPV`
     - No description available.
   * - :obj:`~mufasa.visualization.scatter_3D.scatter_3D`
     - No description available.
   * - :obj:`~mufasa.visualization.scatter_3D.scatter_3D_df`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.Plotter`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.ensure_units_compatible`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.get_cube_slab`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.get_spec_grid`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.plot_fits_grid`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.plot_model`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.plot_spec`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.plot_spec_grid`
     - No description available.
   * - :obj:`~mufasa.visualization.spec_viz.strip_units`
     - No description available.
   