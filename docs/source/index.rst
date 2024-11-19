MUFASA Documentation
====================

Welcome to the documentation for **MUFASA**: the MUlti-component Fitter for Astrophysical Spectral Applications.


Overview
--------
**MUFASA** is a Python library for the analysis and fitting of astrophysical spectral data. It provides tools for:

- **Multi-Component Spectral Fitting**:
  Fit complex spectral data with multiple velocity components to extract gas kinematics more accurate.
- **Data Visualization**:
  Generate plots and 3D visualizations to interpret the spectral model effectively.
- **Custom Modules**:
  Modules can be used to build custom pipelines to handle specific fitting needs.

To learn more about MUFASA, check out the :doc:`Guides <guides>` section.

Citation
~~~~~~~~
Please cite the following paper when publishing with MUFASA-generated data products:

- Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).

Quick Install
~~~~~~~~~~~~~
To quickly install the latest version of **MUFASA**, clone its GitHub `repository <https://github.com/mcyc/mufasa>`_ and run the following command in your local repository directory:

.. code-block:: bash

    pip install -e .

See the :doc:`Installation <installation>` page for details.

Quick Links
-----------
- **Installation**: :doc:`installation`
- **Getting Started**: :doc:`quick_start`
- **Guides**: :doc:`guides`
- **GitHub Repository**: `MUFASA GitHub <https://github.com/mcyc/mufasa>`_
- **Tutorials**: :doc:`tutorials/index`
- **API Reference**: :doc:`api/modules`
- **GitHub Repository**: `MUFASA GitHub <https://github.com/mcyc/mufasa>`_


.. toctree::
   :hidden:

   installation
   quick_start
   guides
   tutorials/index
   api/modules
