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
necessarily mutually exclusive and all use corrected Akaike Information Criterion (AICc) to assess the fit
quality. Let's have a closer look at MUFASA's modular line fitters within these categories:

.. _signal-boosting:

Signal Boosting
^^^^^^^^^^^^^^^^

:func:`iter_2comp_fit() <mufasa.master_fitter.refit_bad_2comp>` achieves signal boosting by first fitting
a spatially convolved cube using moment-based initial guesses. This function then undergoes a second iteration of
fits using results from the initial fit for the subsequent guesses.

Use when:

- Wanting to detect emissions deeper into the noise
- Interested in fainter, larger-scale emissions


Omit when:

- The targeted emission already has a strong signal
- The targeted emission can not be resolved by the convolved beam

.. _quality-masking:

Quality Masking
^^^^^^^^^^^^^^^^^^^^

Quality masking involves identifying where a model is poorly fitted using AICc-derived relative likelihood values
to compare a model to its simpler counterpart, accounting for extra parameters against over-fitting.

:func:`refit_bad_2comp() <mufasa.master_fitter.refit_bad_2comp>` specifically flags where the log-likelihood of a
two-component model relative to a one-component model (i.e., :math:`\ln{K^2_1}`) is below a specified threshold (default -5)
and uses :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>` to find the best-fitted
neighbours of the flagged (i.e., bad) pixels and use those neighbours' fits as guesses to refit bad pixels.
The refitting of these bad pixels is carried out
by :func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>`.

The usage of :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>` in tandem with
:func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>` is a powerful and versatile tool for
refitting bad pixels using the information from the good fits nearby. Use these two functions when:

- attempting to recover poorly fitted pixels near decent fits
- attempting to fit new pixels in the proximity of already-fitted pixels (see :ref:`Spatial Expansion <spatial-expansion>`)


.. _residual-searching:

Residual Searching
^^^^^^^^^^^^^^^^^^

Residual searching looks for signals in the fit residual by attempting to fit another single-component model
to the residual using moment guesses. Residual searching performed by
:func:`refit_2comp_wide() <mufasa.master_fitter.refit_2comp_wide>` also employees signal boosting by convolving the
residual first before attempting the refit. The residual's fitted model is subsequently used as guesses for the
''missing'' second component alongside results of the original one-component fit as guesses for the other
component in attempts to recover the potentially missing component.

Use when:

- Looking for a spectral component with a significantly different velocity than that of the pre-existing model
- Attempting to fit an additional spectral component on top of pre-existing model

.. _spatial-expansion:

Spatial Expansion
^^^^^^^^^^^^^^^^^

Spatial expansion is closely related to quality masking in that they both use existing models in the
nearby pixels to help (re)fit pixels using :func:`get_refit_guesses() <mufasa.master_fitter.get_refit_guesses>`
alongside :func:`replace_bad_pix() <mufasa.master_fitter.replace_bad_pix>`. Spatial expansion, however, is
distinct from quality masking in that it expand the fits into unfitted surroundings rather than attempting
to recover pre-existing poor fits.

:func:`refit_marginal() <mufasa.master_fitter.refit_marginal>` is MUFASA's higher-level wrapper that performs
spatial expansion, which employs :func:`expand_fits() <mufasa.master_fitter.expand_fits>` internally to fit
the surroundings iteratively. MUFASA's default
pipeline uses these tools to expand fits from robust, high signal-to-noise regions into their fainter
surroundings. For quality assurance, MUFASA further employs
:func:`refit_marginal() <mufasa.master_fitter.refit_marginal>` to first refit pixels with models that are only
marginally better than their simpler counterpart before using them as initial guesses for the spatial expansion.

Use when:

- Trying to fit the fainter emissions using pre-existing fits
- Wanting to improve fitting efficiency by leveraging pre-existing fits for guesses over spectral moments

Avoid when:

- The surrounding emission potentially has gas properties (e.g., line-of-sight velocity) that are significantly
  different from those of the already-fitted emission