Guides
======

Welcome to **MUFASA**'s guides. To get started or to learn from tutorials, please see
:doc:`Quick Start <quick_start>` and :doc:`Tutorials <tutorials/index>`.

Preinstalled Access:
--------------------------
For the astrophysics community, MUFASA is available pre-installed on the `CANFAR Science Portal <https://www.canfar.net>`_
via `docker containers <https://www.docker.com/resources/what-container/>`_. Interactive container sessions are
accessible by browser-based Jupyter Lab interface, which also supports an Unix terminal and direct upload/download interfaces.
To request an account and get started with the Science Portal, please see CANFAR's
`documentation <https://www.opencadc.org/science-containers/complete/>`_ page.

To launch a MUFASA notebook container session:

1. Sign onto the `Science Portal <https://www.canfar.net/science-portal/>`_
2. Select ''notebook'' for a session type under ''new session''
3. Select ''crispasa'' under ''project''
4. Select a version of pre-installed MUFASA before launching a session

.. note::

    When requesting memory with the Science Portal, pick a value that's about 20 times the size of your image cube.

Data Products
--------------

Fitted models
^^^^^^^^^^^^^

MUFASA's data products include fitted parameter maps and images derived from the best-fit model images, such as
the model moment map. For more details, please see the :ref:`outputs-section` Section.


Metadata
^^^^^^^^^
The MUFASA version that produced a data product, as well as the time the product was written, can be found under
'HISTORY' in products' FITS headers.

Contributing
-------------

Contributing to MUFASA is most welcome! Main areas for improvements are:

- Adding new molecular spectral models (e.g., CO, HCN)
- Implementing a three-component fit pipline
- Perform quantitative tests on MUFASA's performance with different spectral models

The first step to contributing is by joining a GitHub `discussion <https://github.com/mcyc/mufasa/issues>`_
on your topic of interest or starting a new `issue <https://github.com/mcyc/mufasa/issues>`_. Conversations
on how to contribute will proceed from there.


Custom Usage & Pipline
----------------------
Explore advanced use cases and customization options:

- :doc:`Custom Usage <tutorials/custom_usage>`: Coming soon! Learn how to extend and customize **MUFASA** for unique workflows.

Reference Materials
-------------------
For detailed information on all available modules, classes, and methods, refer to:

- :doc:`API Reference <api/index>`: A complete reference for the **MUFASA** API.
