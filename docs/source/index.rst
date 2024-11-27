MUFASA Documentation
====================
        Welcome to **MUFASA**: the MUlti-component Fitter for Astrophysical Spectral Applications!

Overview
--------
**MUFASA** is a Python library for automating multi-component spectral fits to astrophysical data. It provides tools for:

- **Automated Multi-Component Fitting**:
  Fit spectral cubes automatically with multi-component models, leveraging spatial information for enhanced performance. The currently available models are:

    - **NH₃ (1,1)** (2-components; ``fittype='nh3_multi_v'``)
    - **N₂H⁺ (1-0)** (2-components; ``fittype='n2hp_multi_v'``)

- **Data Visualization**:
  Plots spectral and 3D visualizations of the fitted models.

- **Custom Modules**:
  For building custom pipelines or post-processing refinements, tailored for unique fitting needs.

Citation
~~~~~~~~
When publishing with **MUFASA**-generated data products, please cite the following paper:

    - Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).

Navigation
~~~~~~~~~~

To get started quickly, please see the :doc:`Install <installation>` and :doc:`Quick Start <quick_start>` pages.
Use the navigation bar at the top to explore other pages, including,
:doc:`Tutorials <tutorials/index>`, :doc:`Guides <guides>`, and :doc:`API Reference <api/index>`.

.. toctree::
   :hidden:

   Install <installation>
   Starting <quick_start>
   guides
   tutorials/index
   api/index
