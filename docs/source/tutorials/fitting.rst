Fitting
=======

This tutorial demonstrates how to perform spectral fitting using **MUFASA**.

Preparing for Fitting
---------------------
Before performing spectral fitting, ensure that your data and results are properly loaded. Refer to the :doc:`Loading Data and Results <load_data_n_results>` tutorial for detailed instructions on setting up the input data and initializing the :class:`~mufasa.master_fitter.Region` object.

Performing Spectral Fits
------------------------
Once the :class:`~mufasa.master_fitter.Region` object is set up, you can perform automated spectral fitting. For example, to fit up to 2 velocity components:

.. code-block:: python

    # Perform a two-component fit
    region.master_2comp_fit(snr_min=0)

Supported Spectral Models
~~~~~~~~~~~~~~~~~~~~~~~~~
The currently available spectral models are:

- **NH₃ (1,1)** (2-components; ``fittype='nh3_multi_v'``)
- **N₂H⁺ (1-0)** (2-components; ``fittype='n2hp_multi_v'``)

Next Steps
----------
After performing a fit, consider visualizing the results to verify the quality of the fit and analyze the data further. See :doc:`Visualization <visualization>` for details.
