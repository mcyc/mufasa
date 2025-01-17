Installation
=================
.. note::

   **Recommended Install**
    MUFASA is undergoing rapid developments at the moment,
    install from the source (see Option 1) to stay up-to-date with the latest version.

System Requirements
--------------------
MUFASA is compatible with the following environments:
- Python 3.8 or later.
- Dependencies include `NumPy`, `Astropy`, `Spectral-Cube`, and `pyspeckit`.

Instructions
-------------

1. **Installing from Source**:
   to install the latest developing version, clone the MUFASA GitHub repository:

   .. code-block:: bash

        git clone https://github.com/mcyc/mufasa.git
        cd mufasa
        pip install -e .


   To use a specific version using a tag, for example, v1.4.2, run the following after
   the initial install:

   .. code-block:: bash

        git checkout v1.4.2
        git pull

   .. note::
       If you encounter issues with pre-existing versions of dependencies or want to ensure
       that the pinned versions of MUFASA's dependencies are installed, use the following command:

       .. code-block:: bash

           pip install --upgrade --force-reinstall --no-cache-dir -e .

       This command ensures that all dependencies are freshly installed, replacing any older or conflicting versions.

2. **Dependencies**:
   MUFASA will automatically install its dependencies during installation. If any issues occur, install them manually:

   .. code-block:: bash

       pip install numpy astropy spectral-cube pyspeckit

3. **Installing from PyPI**:

   .. warning::

        The PyPI version of MUFASA is currently out of date. It will be updated once PyPI
        dependencies issues has been resolved.


   The easiest way to install MUFASA is through PyPI. Run the following command in your terminal:

   .. code-block:: bash

       pip install mufasa

Verification
~~~~~~~~~~~~~
To verify the installation, open a Python interpreter and import MUFASA:

.. code-block:: python

    import mufasa
    print(mufasa.__version__)

If no errors occur, MUFASA is installed correctly!
