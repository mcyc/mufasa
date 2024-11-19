Fitting Tutorial
=================

This tutorial guides you through the process of performing spectral fitting using MUFASA's tools.

Contents
--------

- **Basic Fitting**: Learn how to perform single-component and multi-component fits.
- **Advanced Fitting Options**: Customize your fitting pipeline with additional parameters.

Getting Started
---------------
To perform a basic fit on NH₃ (1,1) data, follow these steps:

.. code-block:: python

    from mufasa import master_fitter as mf

    # Define paths and parameters
    cubePath = "path/to/cube.fits"
    paraNameRoot = "results"
    paraDir = "output_dir"

    # Initialize the Region object
    region = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')

    # Perform a fit
    region.master_2comp_fit(snr_min=0)

More details will be added in future updates.
