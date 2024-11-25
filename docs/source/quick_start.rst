Quick Start
===========

Let's help you get started quickly with **MUFASA**. Use the navigation bar on the right to quickly locate specific topics.

Installation
------------
If you haven't installed **MUFASA** yet, please follow the instructions in :doc:`Instal <installation>`.

Setting Up Fits
---------------
To set up for spectral fits, initialize a :class:`~mufasa.master_fitter.Region` object,
which manages the input data, model parameters, and output files.

.. code-block:: python

    from mufasa import master_fitter as mf

    # Define input and output paths
    cubePath = "path/to/cube.fits"   # Path to the data cube
    paraDir = "output_dir"           # Directory for the output files
    paraNameRoot = "results"         # Prefix for the output file names

    # Specify the spectral model type
    fittype = 'nh3_multi_v'

    # Initialize the Region object
    region = mf.Region(cubePath, paraNameRoot, paraDir, fittype)


Fitting Spectra
---------------
.. _fitting-spectra:

To perform automated spectral fitting, use the following example:

.. code-block:: python

    # Perform an automated two-components fit
    region.master_2comp_fit(snr_min=3)

The currently available spectral models are:

- **NH₃ (1,1)** (2-components; ``fittype='nh3_multi_v'``)
- **N₂H⁺ (1-0)** (2-components; ``fittype='n2hp_multi_v'``)

Visualizing Results
-------------------

Reading the Saved Results
~~~~~~~~~~~~~~~~~~~~~~~~~
If you have exited the fitting session already, you can quickly load the saved results by
reinitializing ``region`` (see the :ref:`Fitting Spectra <fitting-spectra>` section) and running:

.. code-block:: python

    # Load the saved fits for the 1 and 2 component models
    region.load_fits(ncomp=[1, 2])

Plotting Spectral Fits
~~~~~~~~~~~~~~~~~~~~~~
To plot the fitted spectra for a specific position ``(x, y)``:

.. code-block:: python

    # Plot the fits for position (x, y) on a 3x3 grid for a 2-component model
    region.ucube.plot_fits_grid(x, y, ncomp=2, size=3, xlim=None, ylim=None)

3D Scatter Plots
~~~~~~~~~~~~~~~~
To plot the fitted parameters in 3D, such as in position-position-velocity (PPV) space:

.. code-block:: python

    # Plot fitted results as an interactive 3D HTML file, saved to `savepath`
    region.plot_ppv_scatter(savepath, vel_scale=0.5, showfig=True, auto_open_html=False)

Exploring More
--------------
- For more examples and detailed guides, see :doc:`Tutorials <tutorials/index>`.
- For a full reference of available modules, visit the :doc:`API Reference <api/index>`.
