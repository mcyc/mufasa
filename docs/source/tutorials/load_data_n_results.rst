Loading Data and Results
========================

This tutorial explains how to load data for processing and access saved results in **MUFASA**.

Loading Data
------------
To load your input data files into MUFASA for processing, initialize a :class:`~mufasa.master_fitter.Region` object. The `~mufasa.master_fitter.Region` object manages the input data, model parameters, and output files.

.. code-block:: python

    from mufasa import master_fitter as mf

    # Define the path to your data cube
    cubePath = "path/to/cube.fits"

    # Specify output paths
    paraDir = "output_dir"
    paraNameRoot = "results"

    # Specify the spectral model type
    fittype = 'nh3_multi_v'

    # Initialize the Region object
    region = mf.Region(cubePath, paraNameRoot, paraDir, fittype)

Accessing Saved Results
-----------------------
Reload results from previous fitting sessions for further analysis. After reinitializing the :class:`~mufasa.master_fitter.Region` object, load the saved fits using the following example:

.. code-block:: python

    # Load saved fits for 1- and 2-component models
    region.load_fits(ncomp=[1, 2])

Inspecting Results
-------------------
Access detailed fit results programmatically using the `Region` object:

.. code-block:: python

    # Inspect parameters for the first component at position (x, y)
    params = region.get_fit_params(x, y, ncomp=1)
    print(params)

Next Steps
----------
After loading your data and results, explore the following tutorials:

- Perform fitting: :doc:`fitting`
- Visualize your results: :doc:`visualization`
