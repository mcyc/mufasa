.. MUFASA documentation master file, created by
   sphinx-quickstart on Sat Nov 16 13:26:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MUFASA Documentation
====================

Welcome to the documentation for **MUFASA**: the MUlti-component Fitter for Astrophysical Spectral Applications. MUFASA is a Python library designed to automate spectral-fitting with multi-component models, particularly for molecular specifies with hyperfine lines. MUFASA also offers visualzation tools for the fits and special modules for costume pipe building to meet special user cases.

Overview
========

**MUFASA** is a Python library for the analysis and fitting of astrophysical spectral data. It provides tools for:

- **Multi-Component Spectral Fitting**:
  Fit complex spectral data with multiple components to extract detailed information.
- **Data Visualization**:
  Generate plots and and 3D visualizations to interpret the spectral model effectively.
- **Data Manipulation**:
  Tools to handle and process astrophysical unstructred (e.g., images) and strucgtured (e.g., tables) datasets for analysis.


Installation
------------
See the :doc:`Installation <installation>` page for details. To quickly install the latest version of **MUFASA**, clone its GitHub `repository <https://github.com/mcyc/mufasa>`_ and run the following command in your local repository directory:

.. code-block:: bash

    pip install -e .


Reference
------------
Please cite the following paper when publishing with MUFASA-generated data products:

- Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).


Quick Start
-----------

Running 2-Component Fits
~~~~~~~~~~~~~~~~~~~~~~~~~

To perform an NH₃ (1,1) fit automatically, up to two components, simply run the following:

.. code-block:: python

    from mufasa import master_fitter as mf
    reg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
    reg.master_2comp_fit(snr_min=0)

Supported Line Models
^^^^^^^^^^^^^^^^^^^^^

- NH₃ (1,1) multi-component model (``fittype='nh3_multi_v'``)
- N₂H⁺ (1-0) multi-component model (``fittype='n2hp_multi_v'``)

Visualizing the Results
~~~~~~~~~~~~~~~~~~~~~~~~

Plotting Fitted Spectra
^^^^^^^^^^^^^^^^^^^^^^^

To quickly plot the fitted spectra from the saved fits for position `(x, y)` and its surrounding pixels, use:

.. code-block:: python

    from mufasa import UltraCube as UCube

    ucube = UCube.UCubePlus(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
    ucube.read_model_fit(ncomps=[1, 2])
    ucube.plot_fits_grid(x, y, ncomp=2, size=3, xlim=None, ylim=None)

Guides to MUFASA
=================
To learn more about MUFASA’s modules, classes, and methods, explore the following guides:

.. toctree::
   :maxdepth: 2

   introduction
   installation
   usage


API Documentation
=================
For detailed API documentation, refer to the following:

.. toctree::
   :maxdepth: 2

   api/modules



For more detailed information, visit the `MUFASA GitHub repository <https://github.com/mcyc/mufasa>`_.

