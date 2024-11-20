MUFASA Documentation
====================

Welcome to the documentation for **MUFASA**: the MUlti-component Fitter for Astrophysical Spectral Applications.


Overview
--------
**MUFASA** is a Python library for automating multi-component spectral fits to astrophysical data. It provides tools for:

- **Multi-Component Fitting**:
  Fit complex spectral data automatically with multi-component models, leveraging spatial information in a data cube for enhanced performance. The currently available models are:

    - **NH₃ (1,1)** (2-components; ``fittype='nh3_multi_v'``)
    - **N₂H⁺ (1-0)** (2-components; ``fittype='n2hp_multi_v'``)

- **Data Visualization**:
  Generate spectral plots and 3D visualizations to interpret fitted models.

- **Custom Modules**:
  Provide modules for custom building pipelines tailored towards unique fitting needs.

To learn more, check out the :doc:`Guides <guides>` section or use the navigation bar at the top.

Citation
~~~~~~~~
When publishing with MUFASA-generated data products, please cite the following paper:

    - Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).

Quick Install
~~~~~~~~~~~~~
To quickly clone and install the latest version of **MUFASA** locally from `GitHub <https://github.com/mcyc/mufasa>`_:

.. code-block:: bash

    git clone https://github.com/mcyc/mufasa.git
    cd mufasa
    pip install -e .

For more details, see the :doc:`Instal <installation>` page.

Quick Start
~~~~~~~~~~~
To get started with **MUFASA** quickly, check out the :doc:`Quick Start <quick_start>` guide.

Contents
---------

Use the navigation bar at the top to explore other sections, including :doc:`Tutorials <tutorials/index>`, :doc:`Guides <guides>`, and :doc:`API Reference <api/modules>`.

.. toctree::
   :hidden:

   installation
   quick_start
   guides
   tutorials/index
   api/modules
