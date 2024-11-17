.. MUFASA documentation master file, created by
   sphinx-quickstart on Sat Nov 16 13:26:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MUFASA Documentation
====================

Welcome to the documentation for **MUFASA**: the MUlti-component Fitter for Astrophysical Spectral Applications. MUFASA is a Python library designed to streamline the analysis and fitting of astrophysical spectral data. Whether you're processing spectral cubes, visualizing datasets, or performing multi-component fits, MUFASA offers a robust toolkit for astrophysical research.

Overview
========

**MUFASA** is a Python library for the analysis and fitting of astrophysical spectral data. It provides tools for:

- **Multi-Component Spectral Fitting**:
  Fit complex spectral data with multiple components to extract detailed information.
- **Data Visualization**:
  Generate plots and visualizations to interpret spectral data effectively.
- **Data Manipulation**:
  Tools to handle and process astrophysical datasets for analysis.

Installation
------------
To install the latest version of **MUFASA**, clone its GitHub `repository <https://github.com/mcyc/mufasa>`_ and run the following command in your local directory:

.. code-block:: bash

    pip install -e .

Alternatively, to install the latest release from PyPI:

.. code-block:: bash

    pip install mufasa

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

- `nh3_multi_v`: Multi-component NH₃ (1,1) model.
- `n2hp_multi_v`: Multi-component N₂H⁺ (1-0) model.

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

Plotting Arguments
""""""""""""""""""
- **`ncomp`**: Number of components in the model.
- **`size`**: Size of the plot grid (e.g., `size=3` results in a 3x3 grid centered on position `(x, y)`).
- **`xlim` and `ylim`**: Constrain the plot's x and y limits.

Plotting in PPV with 3D Scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the best-fit models in position-position-velocity (PPV) space, run:

.. code-block:: python

    reg.plot_ppv_scatter(savepath, vel_scale=0.5)

Scatter Plot Arguments
""""""""""""""""""""""
- **`savepath`**: File path to save the plot as an HTML file.
- **`vel_scale`**: Scale factor for the velocity axis relative to the x and y axes.

For more detailed information, advanced examples, and contributing guidelines, visit the `MUFASA GitHub repository <https://github.com/mcyc/mufasa>`_.

API Reference
=============
Explore the API documentation to learn about MUFASA’s modules, classes, and methods:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   api/modules
   api/mufasa.UltraCube
   api/mufasa.master_fitter
   api/mufasa.aic
   api/mufasa.spec_models
   api/mufasa.utils
