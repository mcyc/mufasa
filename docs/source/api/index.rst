:html_theme.sidebar_secondary.remove:

.. _api_ref:

=============
API Reference
=============

This is **MUFASA**'s class and function reference. In addition to the table below,
you can also search with the search bar located on the top right corner of the webpage.

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
     - .. container:: sk-apisearch-desc

          A subclass of UltraCube that includes directory management for parameter maps and model fits.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.UltraCube`
     - .. container:: sk-apisearch-desc

          A framework to manage and fit multi-component spectral models for spectral cubes.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.calc_AICc`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.calc_AICc_likelihood`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.calc_chisq`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.calc_rss`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.convolve_sky_byfactor`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.expand_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.fit_cube`
     - .. container:: sk-apisearch-desc

          Fit the spectral cube using the specified fitting type.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_Tpeak`
     - .. container:: sk-apisearch-desc

          Calculate the peak value of a model cube at each spatial pixel.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_all_lnk_maps`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_best_2c_parcube`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_chisq`
     - .. container:: sk-apisearch-desc

          Calculate the chi-squared or reduced chi-squared value for a spectral cube.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_masked_moment`
     - .. container:: sk-apisearch-desc

          Calculate a masked moment of a spectral cube.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_residual`
     - .. container:: sk-apisearch-desc

          Calculate the residual between the data cube and the model cube.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_rms`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.get_rss`
     - .. container:: sk-apisearch-desc

          Calculate the residual sum of squares (RSS) for a spectral cube model fit.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.is_K`
     - .. container:: sk-apisearch-desc

          Check if a given unit is equivalent to Kelvin (K).

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.load_model_fit`
     - .. container:: sk-apisearch-desc

          Load the spectral fit results from a.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.save_fit`
     - .. container:: sk-apisearch-desc

          Save the fitted parameter cube to a.

          .. container:: caption

             :mod:`mufasa.UltraCube`
   * - :obj:`~mufasa.UltraCube.to_K`
     - .. container:: sk-apisearch-desc

          Convert the unit of a spectral cube to Kelvin (K).

          .. container:: caption

             :mod:`mufasa.UltraCube`
   
   * - :obj:`~mufasa.aic.AIC`
     - .. container:: sk-apisearch-desc

          Calculate the Akaike Information Criterion (AIC).

          .. container:: caption

             :mod:`mufasa.aic`
   * - :obj:`~mufasa.aic.AICc`
     - .. container:: sk-apisearch-desc

          Calculate the corrected Akaike Information Criterion (AICc).

          .. container:: caption

             :mod:`mufasa.aic`
   * - :obj:`~mufasa.aic.fits_comp_AICc`
     - .. container:: sk-apisearch-desc

          A wrapper function to calculate corrected Akaike Information Criterion (AICc) values.

          .. container:: caption

             :mod:`mufasa.aic`
   * - :obj:`~mufasa.aic.fits_comp_chisq`
     - .. container:: sk-apisearch-desc

          Calculate and save chi-squared values for the given cube and model fits.

          .. container:: caption

             :mod:`mufasa.aic`
   * - :obj:`~mufasa.aic.get_comp_AICc`
     - .. container:: sk-apisearch-desc

          Calculate AICc values for two models over the same samples.

          .. container:: caption

             :mod:`mufasa.aic`
   * - :obj:`~mufasa.aic.likelihood`
     - .. container:: sk-apisearch-desc

          Calculate the log-likelihood of model A relative to model B.

          .. container:: caption

             :mod:`mufasa.aic`
   
   * - :obj:`~mufasa.clean_fits.fit_results`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   * - :obj:`~mufasa.clean_fits.above_ErrV_Thresh`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   * - :obj:`~mufasa.clean_fits.clean_2comp_maps`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   * - :obj:`~mufasa.clean_fits.exclusive_2comp_maps`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   * - :obj:`~mufasa.clean_fits.extremeV_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   * - :obj:`~mufasa.clean_fits.remove_zeros`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.clean_fits`
   
   * - :obj:`~mufasa.convolve_tools.convolve_sky`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.convolve_sky_byfactor`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.edge_trim`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.get_celestial_hdr`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.regrid`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.regrid_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   * - :obj:`~mufasa.convolve_tools.snr_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.convolve_tools`
   
   * - :obj:`~mufasa.deblend_cube.deblend`
     - .. container:: sk-apisearch-desc

          Deblend hyperfine structures in a cube based on fitted models.

          .. container:: caption

             :mod:`mufasa.deblend_cube`
   
   * - :obj:`~mufasa.exceptions.FitTypeError`
     - .. container:: sk-apisearch-desc

          Fitttype provided is not valid.

          .. container:: caption

             :mod:`mufasa.exceptions`
   * - :obj:`~mufasa.exceptions.SNRMaskError`
     - .. container:: sk-apisearch-desc

          SNR Mask has no valid pixel.

          .. container:: caption

             :mod:`mufasa.exceptions`
   * - :obj:`~mufasa.exceptions.StartFitError`
     - .. container:: sk-apisearch-desc

          Fitting failed from the beginning.

          .. container:: caption

             :mod:`mufasa.exceptions`
   
   * - :obj:`~mufasa.guess_refine.get_celestial_hdr`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.guess_from_cnvpara`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.mask_cleaning`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.mask_swap_2comp`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.master_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.quick_2comp_sort`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.refine_2c_guess`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.refine_each_comp`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.refine_guess`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.regrid`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.save_guesses`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.simple_para_clean`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   * - :obj:`~mufasa.guess_refine.tautex_renorm`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.guess_refine`
   
   * - :obj:`~mufasa.master_fitter.Region`
     - .. container:: sk-apisearch-desc

          A class to represent the observed spectral cube to perform the model fits.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.expand_fits`
     - .. container:: sk-apisearch-desc

          Expand fits in a region by incrementally fitting pixels beyond a defined.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.fit_best_2comp_residual_cnv`
     - .. container:: sk-apisearch-desc

          Fit the convolved residual of the best-fit two-component spectral model.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.fit_surroundings`
     - .. container:: sk-apisearch-desc

          Expand fits around a region based on model log-likelihood thresholds and.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_2comp_wide_guesses`
     - .. container:: sk-apisearch-desc

          Generate initial guesses for fitting a two-component spectral model with.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_best_2comp_model`
     - .. container:: sk-apisearch-desc

          Retrieve the best-fit model cube for the given Region object.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual`
     - .. container:: sk-apisearch-desc

          Calculate the residual cube for the best-fit two-component model.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual_SpectralCube`
     - .. container:: sk-apisearch-desc

          Generate the residual spectral cube for the best-fit two-component.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_best_2comp_residual_cnv`
     - .. container:: sk-apisearch-desc

          Generate a convolved residual cube for the best-fit two-component.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_best_2comp_snr_mod`
     - .. container:: sk-apisearch-desc

          Calculate the signal-to-noise ratio (SNR) map for the best-fit two-.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_convolved_cube`
     - .. container:: sk-apisearch-desc

          Generate and save a convolved version of the spectral cube for a given.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_convolved_fits`
     - .. container:: sk-apisearch-desc

          Fit a model to the convolved cube and save the results.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_fits`
     - .. container:: sk-apisearch-desc

          Fit a model to the original spectral cube and save the results.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_local_bad`
     - .. container:: sk-apisearch-desc

          Identify local pixels with significantly lower relative log-likelihood.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_marginal_pix`
     - .. container:: sk-apisearch-desc

          Return pixels at the edge of structures with values greater than.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_refit_guesses`
     - .. container:: sk-apisearch-desc

          Generate initial guesses for refitting based on neighboring pixels or.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.get_skyheader`
     - .. container:: sk-apisearch-desc

          Generate a 2D sky projection header from a 3D spectral cube header.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.iter_2comp_fit`
     - .. container:: sk-apisearch-desc

          Perform a two-component fit iterantively through two steps.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.master_2comp_fit`
     - .. container:: sk-apisearch-desc

          Perform a two-component fit on the data cube within a Region object.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.refit_2comp_wide`
     - .. container:: sk-apisearch-desc

          Refit pixels to recover compoents with wide velocity separation for.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.refit_bad_2comp`
     - .. container:: sk-apisearch-desc

          Refit pixels with poor 2-component fits, as determined by the log-.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.refit_marginal`
     - .. container:: sk-apisearch-desc

          Refit pixels with fits that appears marginally okay, as deterined by the.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.refit_swap_2comp`
     - .. container:: sk-apisearch-desc

          Refit the cube by using the previous fit result as guesses, but with the.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.replace_bad_pix`
     - .. container:: sk-apisearch-desc

          Refit pixels marked by the mask as "bad" and adopt the new model if it.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.replace_para`
     - .. container:: sk-apisearch-desc

          Replace parameter values in a parameter cube with those from a reference.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.replace_rss`
     - .. container:: sk-apisearch-desc

          Replace RSS-related maps in a `UltraCube` object for specific.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.save_best_2comp_fit`
     - .. container:: sk-apisearch-desc

          Save the best two-component fit results for the specified region.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.save_map`
     - .. container:: sk-apisearch-desc

          Save a 2D map as a FITS file.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.save_updated_paramaps`
     - .. container:: sk-apisearch-desc

          Save the updated parameter maps for specified components.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   * - :obj:`~mufasa.master_fitter.standard_2comp_fit`
     - .. container:: sk-apisearch-desc

          Perform a two-component fit for the cube using default moment map.

          .. container:: caption

             :mod:`mufasa.master_fitter`
   
   * - :obj:`~mufasa.moment_guess.LineSetup`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.adaptive_moment_maps`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.get_rms_prefit`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.get_tau`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.get_tex`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.get_window_slab`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.master_guess`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.mom_guess_wide_sep`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.moment_guesses`
     - .. container:: sk-apisearch-desc

          Generate reasonable initial guesses for multiple component fits based on moment maps.

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.moment_guesses_1c`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.noisemask_moment`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.peakT`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.vmask_cube`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.vmask_moments`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.window_mask_pcube`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.moment_guess`
   * - :obj:`~mufasa.moment_guess.window_moments`
     - .. container:: sk-apisearch-desc

          Calculate the zeroth, first, and second moments of a spectrum or cube.

          .. container:: caption

             :mod:`mufasa.moment_guess`
   
   * - :obj:`~mufasa.multi_v_fit.cubefit_gen`
     - .. container:: sk-apisearch-desc

          Perform n velocity component fit on the GAS ammonia 1-1 data.

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.cubefit_simp`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.default_masking`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.get_chisq`
     - .. container:: sk-apisearch-desc

          cube : SpectralCube.

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.get_start_point`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.get_vstats`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.handle_snr`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.make_header`
     - .. container:: sk-apisearch-desc

          Create a new header while retaining.

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.match_pcube_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.register_pcube`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.retry_fit`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.save_guesses`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.save_pcube`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.set_pyspeckit_verbosity`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   * - :obj:`~mufasa.multi_v_fit.snr_estimate`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.multi_v_fit`
   
   * - :obj:`~mufasa.signals.estimate_mode`
     - .. container:: sk-apisearch-desc

          Estimate the mode of the data using a histogram.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_moments`
     - .. container:: sk-apisearch-desc

          Calculate moments of the signals in a cube.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_rms_robust`
     - .. container:: sk-apisearch-desc

          Make a robust RMS estimate.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_signal_mask`
     - .. container:: sk-apisearch-desc

          Provide a 3D mask indicating signal regions based on RMS and SNR threshold.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_snr`
     - .. container:: sk-apisearch-desc

          Calculate the peak signal-to-noise ratio of the cube.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_v_at_peak`
     - .. container:: sk-apisearch-desc

          Find the velocity corresponding to the peak emission.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.get_v_mask`
     - .. container:: sk-apisearch-desc

          Return a mask centered on a reference velocity with a spectral window.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.refine_rms`
     - .. container:: sk-apisearch-desc

          Refine the RMS estimate by masking out signal regions.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.refine_signal_mask`
     - .. container:: sk-apisearch-desc

          Refine a signal mask by removing noisy features and expanding the mask.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.trim_cube_edge`
     - .. container:: sk-apisearch-desc

          Remove spatial edges from a cube.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.trim_edge`
     - .. container:: sk-apisearch-desc

          Trim edges using a 2D mask.

          .. container:: caption

             :mod:`mufasa.signals`
   * - :obj:`~mufasa.signals.v_estimate`
     - .. container:: sk-apisearch-desc

          Estimate the velocity centroid based on peak emission.

          .. container:: caption

             :mod:`mufasa.signals`
   
   * - :obj:`~mufasa.slab_sort.distance_metric`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.slab_sort`
   * - :obj:`~mufasa.slab_sort.mask_swap_2comp`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.slab_sort`
   * - :obj:`~mufasa.slab_sort.quick_2comp_sort`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.slab_sort`
   * - :obj:`~mufasa.slab_sort.refmap_2c_mask`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.slab_sort`
   * - :obj:`~mufasa.slab_sort.sort_2comp`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.slab_sort`
   
   
   * - :obj:`~mufasa.spec_models.ammonia_multiv.T_antenna`
     - .. container:: sk-apisearch-desc

          Calculate antenna temperatures over nu (in GHz).

          .. container:: caption

             :mod:`mufasa.spec_models.ammonia_multiv`
   * - :obj:`~mufasa.spec_models.ammonia_multiv.ammonia_multi_v`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.spec_models.ammonia_multiv`
   * - :obj:`~mufasa.spec_models.ammonia_multiv.nh3_multi_v_model_generator`
     - .. container:: sk-apisearch-desc

          Works for up to 2 componet fits at the moment.

          .. container:: caption

             :mod:`mufasa.spec_models.ammonia_multiv`
   
   * - :obj:`~mufasa.spec_models.meta_model.MetaModel`
     - .. container:: sk-apisearch-desc

          A class to store spectral model-specific information relevant to spectral modeling tasks, such as fitting.

          .. container:: caption

             :mod:`mufasa.spec_models.meta_model`
   
   
   * - :obj:`~mufasa.spec_models.n2hp_deblended.n2hp_vtau_singlemodel_deblended`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.spec_models.n2hp_deblended`
   
   * - :obj:`~mufasa.spec_models.n2hp_multiv.T_antenna`
     - .. container:: sk-apisearch-desc

          Calculate antenna temperatures over nu (in GHz).

          .. container:: caption

             :mod:`mufasa.spec_models.n2hp_multiv`
   * - :obj:`~mufasa.spec_models.n2hp_multiv.n2hp_multi_v`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.spec_models.n2hp_multiv`
   * - :obj:`~mufasa.spec_models.n2hp_multiv.n2hp_multi_v_model_generator`
     - .. container:: sk-apisearch-desc

          Works for up to 2 component fits at the moment.

          .. container:: caption

             :mod:`mufasa.spec_models.n2hp_multiv`
   
   * - :obj:`~mufasa.spec_models.nh3_deblended.nh3_vtau_singlemodel_deblended`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.spec_models.nh3_deblended`
   
   
   * - :obj:`~mufasa.utils.dataframe.assign_to_dataframe`
     - .. container:: sk-apisearch-desc

          Assign values from a new data array to an existing DataFrame based on spatial coordinates and component index.

          .. container:: caption

             :mod:`mufasa.utils.dataframe`
   * - :obj:`~mufasa.utils.dataframe.make_dataframe`
     - .. container:: sk-apisearch-desc

          Create a DataFrame from a 3D parameter array, applying optional velocity and error thresholds.

          .. container:: caption

             :mod:`mufasa.utils.dataframe`
   * - :obj:`~mufasa.utils.dataframe.read`
     - .. container:: sk-apisearch-desc

          Read a FITS file and convert the data to a pandas DataFrame, optionally including the header.

          .. container:: caption

             :mod:`mufasa.utils.dataframe`
   
   * - :obj:`~mufasa.utils.interpolate.expand_interpolate`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.interpolate`
   * - :obj:`~mufasa.utils.interpolate.iter_expand`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.interpolate`
   
   * - :obj:`~mufasa.utils.map_divide.dist_divide`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.map_divide`
   * - :obj:`~mufasa.utils.map_divide.watershed_divide`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.map_divide`
   
   * - :obj:`~mufasa.utils.mufasa_log.OriginContextFilter`
     - .. container:: sk-apisearch-desc

          Filter instances are used to perform arbitrary filtering of LogRecords.

          .. container:: caption

             :mod:`mufasa.utils.mufasa_log`
   * - :obj:`~mufasa.utils.mufasa_log.WarningContextFilter`
     - .. container:: sk-apisearch-desc

          Filter instances are used to perform arbitrary filtering of LogRecords.

          .. container:: caption

             :mod:`mufasa.utils.mufasa_log`
   * - :obj:`~mufasa.utils.mufasa_log.get_logger`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.mufasa_log`
   * - :obj:`~mufasa.utils.mufasa_log.init_logging`
     - .. container:: sk-apisearch-desc

          :param logfile: file to save to (default mufasa.

          .. container:: caption

             :mod:`mufasa.utils.mufasa_log`
   * - :obj:`~mufasa.utils.mufasa_log.reset_logger`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.mufasa_log`
   
   * - :obj:`~mufasa.utils.multicore.validate_n_cores`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.multicore`
   
   * - :obj:`~mufasa.utils.neighbours.disk_neighbour`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.neighbours`
   * - :obj:`~mufasa.utils.neighbours.get_neighbor_coord`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.neighbours`
   * - :obj:`~mufasa.utils.neighbours.get_valid_neighbors`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.neighbours`
   * - :obj:`~mufasa.utils.neighbours.maxref_neighbor_coords`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.neighbours`
   * - :obj:`~mufasa.utils.neighbours.square_neighbour`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.utils.neighbours`
   
   
   * - :obj:`~mufasa.visualization.scatter_3D.ScatterPPV`
     - .. container:: sk-apisearch-desc

          A class to plot the fitted parameters in 3D scatter plots.

          .. container:: caption

             :mod:`mufasa.visualization.scatter_3D`
   * - :obj:`~mufasa.visualization.scatter_3D.scatter_3D`
     - .. container:: sk-apisearch-desc

          Plot a 3D scatter plot with optional opacity scaling for point ranges.

          .. container:: caption

             :mod:`mufasa.visualization.scatter_3D`
   * - :obj:`~mufasa.visualization.scatter_3D.scatter_3D_df`
     - .. container:: sk-apisearch-desc

          A wrapper for scatter_3D to quickly plot a pandas DataFrame in 3D.

          .. container:: caption

             :mod:`mufasa.visualization.scatter_3D`
   
   * - :obj:`~mufasa.visualization.spec_viz.Plotter`
     - .. container:: sk-apisearch-desc

          Undocumented

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.ensure_units_compatible`
     - .. container:: sk-apisearch-desc

          Ensure the limits have compatible units with the data, converting if needed.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.get_cube_slab`
     - .. container:: sk-apisearch-desc

          Extract a spectral slab from the cube over the specified velocity range.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.get_spec_grid`
     - .. container:: sk-apisearch-desc

          Create a grid of subplots for spectra.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.plot_fits_grid`
     - .. container:: sk-apisearch-desc

          Plot a grid of model fits from the cube centered at (x, y).

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.plot_model`
     - .. container:: sk-apisearch-desc

          Plot a model fit for a spectrum.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.plot_spec`
     - .. container:: sk-apisearch-desc

          Plot a spectrum.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.plot_spec_grid`
     - .. container:: sk-apisearch-desc

          Plot a grid of spectra from the cube centered at (x, y).

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   * - :obj:`~mufasa.visualization.spec_viz.strip_units`
     - .. container:: sk-apisearch-desc

          Helper function to strip units from a limit tuple if it contains Quantity.

          .. container:: caption

             :mod:`mufasa.visualization.spec_viz`
   
   