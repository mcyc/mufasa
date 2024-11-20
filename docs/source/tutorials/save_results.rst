Saving Results
==============

If you performed the standard 2-component fit with **MUFASA** using
:func:`~mufasa.master_fitter.master_2comp_fit`, as described in the
:doc:`Get Started <../quick_start>` guide or the :doc:`Fitting Tutorial <fitting>`,
the outputs are automatically saved. Refer to the :ref:`Outputs <outputs-section>`
section below for details on the output files.

If you have performed custom fits, including additional refinements to pipeline
results, see the :ref:`Example <example>` section for instructions on saving
the associated data products using :func:`~mufasa.master_fitter.save_best_2comp_fit`.

.. note::

   **Output Directory**

   All output files are stored in the `region.`:attr:`~mufasa.master_fitter.Region.paraDir`
   directory after a fitting run. Please ensure the contents of the files are verified
   for accuracy.


.. _outputs-section:

Outputs
=======


Raw Outputs
-----------

The following file(s) are automatically generated after each run, without calling
:func:`~mufasa.master_fitter.save_best_2comp_fit`:

+-------------------------------+---------------------------------------------------------+
| **File Name**                 | **Description**                                         |
+===============================+=========================================================+
| ``NH3_para_{n}vcomp.fits``    | Fitted parameter maps for the ``n`` component model     |
+-------------------------------+---------------------------------------------------------+

.. note::

   **Naming Convention**

   The first part of the file names is automatically generated based on the input cube's file name,
   which in this example is `'NH3'`.

   By default, MUFASA uses `'paramaps'` in the output file names. For clarity, this example
   uses `'para'` instead of the default `'paramaps'`.


Best-model Outputs
------------------

The following output files are generated after calling
:func:`~mufasa.master_fitter.save_best_2comp_fit`, which first performs model selection in
each pixel to select the best-fit model:

+-------------------------------+---------------------------------------------------------+
| **File Name**                 | **Description**                                         |
+===============================+=========================================================+
| ``NH3_para_2vcomp_final.fits``| Final model-selected best-fit parameters (1 or 2 comp). |
+-------------------------------+---------------------------------------------------------+
| ``NH3_lnk21.fits``            | Relative log-likelihood of 2-comp vs 1-comp fits.       |
+-------------------------------+---------------------------------------------------------+
| ``NH3_lnk10.fits``            | Relative log-likelihood of 1-comp fit vs noise          |
+-------------------------------+---------------------------------------------------------+
| ``NH3_lnk20.fits``            | Relative log-likelihood of 2-comp fit vs noise          |
+-------------------------------+---------------------------------------------------------+
| ``NH3_SNR.fits``              | Estimated peak signal-to-noise ratio map.               |
+-------------------------------+---------------------------------------------------------+
| ``NH3_model_mom0.fits``       | Moment 0 map (integrated intensity) of the best model.  |
+-------------------------------+---------------------------------------------------------+
| ``NH3_mom0.fits``             | Moment 0 map (integrated intensity) of the data,        |
|                               | masked by the best model.                               |
+-------------------------------+---------------------------------------------------------+
| ``NH3_chi2red_1c.fits``       | Reduced chi-squared map for the 1-component model.      |
+-------------------------------+---------------------------------------------------------+
| ``NH3_chi2red_2c.fits``       | Reduced chi-squared map for the 2-component model.      |
+-------------------------------+---------------------------------------------------------+

.. note::

   :func:`~mufasa.master_fitter.save_best_2comp_fit`: is specifically build for 2-component
   models. Similar function for higher number of components has not been implemented.


.. _example:

Example
========
To save the results after spectral fitting, run the following:

.. code-block:: python

    from mufasa.master_fitter import save_best_2comp_fit

    # Save the best-fitting results
    save_best_2comp_fit(
        reg=region,
        multicore=True,
        from_saved_para=False,
        lnk21_thres=5,
        lnk10_thres=5
    )

    print("Results saved to:", region.ucube.paraDir)


For initializing your :class:`~mufasa.master_fitter.Region` object, see :doc:`Loading Data and Results <load_data_n_results>`.

Next Steps
==========
Once the results are saved, explore them using **MUFASA**'s visualization tools. See :doc:`Visualizing Results <visualization>` for more details. For an overview of the complete workflow, refer to :doc:`Guides <../guides>`.
