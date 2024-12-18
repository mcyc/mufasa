Installation
=================
.. note::

   **Recommended Install**
    MUFASA is undergoing rapid developments at the moment,
    install from the source (see Option 2) to stay up-to-date with the latest version.

System Requirements
--------------------
MUFASA is compatible with the following environments:
- Python 3.8 or later.
- Dependencies include `NumPy`, `Astropy`, `Spectral-Cube`, and `pyspeckit`.

Instructions
-------------

1. **Installing from PyPI**
   The easiest way to install MUFASA is through PyPI. Run the following command in your terminal:

   .. code-block:: bash

       pip install mufasa

2. **Installing from Source**
   To use the latest development version, clone the MUFASA GitHub repository:

   .. code-block:: bash

       git clone https://github.com/mcyc/mufasa.git
       cd mufasa
       pip install -e .

   .. note::
       If you encounter issues with pre-existing versions of dependencies or want to ensure
       that the pinned versions of MUFASA's dependencies are installed, use the following command:

       .. code-block:: bash

           pip install --upgrade --force-reinstall --no-cache-dir -e .

       This command ensures that all dependencies are freshly installed, replacing any older or conflicting versions.

3. **Dependencies**
   MUFASA will automatically install its dependencies during installation. If any issues occur, install them manually:

   .. code-block:: bash

       pip install numpy astropy spectral-cube pyspeckit

Verification
~~~~~~~~~~~~~
To verify the installation, open a Python interpreter and import MUFASA:

.. code-block:: python

    import mufasa
    print(mufasa.__version__)

If no errors occur, MUFASA is installed correctly!
