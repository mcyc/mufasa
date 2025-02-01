Custom Usage
============

MUFASA's default pipeline is built using
modular components that can be reconfigured for custom usage and pipeline builds. Here, we will walk
you through the purpose and usage of these components.

Modular Fitters
---------------------

The modular functions behind MUFASA's default pipeline operate primarily within the following methodological categories:
:ref:`signal boosting <signal-boosting>`, :ref:`quality masking <quality-masking>`,
:ref:`residual searching <residual-searching>`, and :ref:`spatial expansion <spatial-expansion>`.
These categories are not
necessarily mutually exclusive and all use corrected Akaike Information Criterion (AICc) to assess the quality
of the fits. Let's look more closely at MUFASA's modular line fitter grouped by these categories:

.. _signal-boosting:

Signal Boosting
^^^^^^^^^^^^^^^^

:func:`iter_2comp_fit() <mufasa.master_fitter.refit_bad_2comp>` achieves signal boosting by first fitting
a spatially convolved cube using moment-based initial guesses. This function then undergo a second iteration of
fits using results from the initial fit for the subsequent guesses.

Use when:

- Wanting to detect emission deeper into the noise
- Interested in fainter, larger-scale emissions


Omit when:

- The targeted emission already has a strong signal
- The targeted emission cannot not be resolved by the convolved beam

.. _quality-masking:

Quality Masking
^^^^^^^^^^^^^^^^^^^^

Quality masking involves identifying where a model is poorly fitted using AICc-derived relative likelihood values
to compare a model to its simpler counterpart, accounting for extra parameters against over-fitting.

:func:`refit_bad_2comp() <mufasa.master_fitter.refit_bad_2comp>` specifically flags where the log likelihood of a
two component model relative to a one component model (i.e., :math:`\ln{K^2_1}`) is below a specified threshold (default -5)
and uses :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>` to find flagged (i.e., bad) pixels' best fitted
neighbours and use those neighbours' fits as guesses for refitting. The refitting of these bad pixel is carried out
by :func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>`.

The usage of :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>` in tandem with
:func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>` is a powerful and highly versatile combination
to refit bad pixels using the good fits nearby. Use these two functions when:

- attempting to recover poor fits in pixels surrounded by decent fits
- attempting to fit new pixels within the proximity of already fitted pixels


.. _residual-searching:

Residual Searching
^^^^^^^^^^^^^^^^^^

Residual searching looks for signal in the fit residual by attempting to fit another single-component spectra
to the residual using moment guesses. Residual searching performed by
:func:`refit_2comp_wide() <mufasa.master_fitter.refit_2comp_wide>` also employees signal boosting by convolve the
residual first before attempting the fit. The model fitted to the residual is subsequently used as the guesses for the
''missing'' second component while the results of the original one-component fit acts as the guesses for the other
component in attempts to recovery the possibly missing second component.

Use when:

- Looking for a spectral component that is not near the initial guessing velocity range
- Attempting to fit an additional spectral component with the current fits as a base

.. _spatial-expansion:

Spatial Expansion
^^^^^^^^^^^^^^^^^

Spatial expansion is very similar to refitting with quality masking in that they both use existing models in the
nearby pixels as guesses for the fit/refit, using :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>`
alongside :func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>` to accomplish this. Spatial expansion is
distinct in that expand the fits into unfitted surroundings rather than trying to recover poor fits.

:func:`refit_marginal() <mufasa.master_fitter.refit_marginal>` is the higher-level wrapper functions uses
:func:`expand_fits() <mufasa.master_fitter.expand_fits>` to fit the surroundings iteratively. MUFASA's default
pipeline uses such an approach to expand fits from robust, high signal-to-noise fits into the noisy, fainter
surroundings. To ensure the quality of the expansion, MUFASA further employees
:func:`refit_marginal() <mufasa.master_fitter.refit_marginal>` to refit pixels that have models that appears
to be only marginally better than their simpler counterpart before pushing further into the highly uncertain regions.

Use when:

- Avoid brute force guessing in faint regions
- Wanting

Avoid when:

- The surrounding emission may have very distinct properties (e.g., line-of-sight velocity) than fitted emission
