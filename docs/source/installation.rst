Installation
=================

System Requirements
--------------------
MUFASA is compatible with the following environments:
- Python 3.8 or later.
- Dependencies include `NumPy`, `Astropy`, `Spectral-Cube`, and `pyspeckit`.

Instructions
-------------

1. **Install from PyPI**:

   The easiest way to install MUFASA is through PyPI. Run the following command in your terminal
   to get the latest stable version:

   .. code-block:: bash

       pip install mufasa

2. **Install from Source**:
   to install the latest developing version, clone the MUFASA GitHub repository and install by
   using:

   .. code-block:: bash

        git clone https://github.com/mcyc/mufasa.git
        cd mufasa
        pip install -e .


   To switch to a particular version specified by a tag, (e.g., v1.4.3) or a git branch, run the following
   from your local git repository:

   .. code-block:: bash

        git checkout v1.4.3
        git pull

   .. note::
       If you encounter issues with pre-existing versions of dependencies or want to ensure
       that the pinned versions of MUFASA's dependencies are installed, use the following command:

       .. code-block:: bash

           pip install --upgrade --force-reinstall --no-cache-dir -e .

       This command ensures that all dependencies are freshly installed, replacing any older or conflicting versions.

3. **Dependencies**:
   MUFASA will automatically install its dependencies during installation. If any issues occur, install them manually:

   .. code-block:: bash

       pip install numpy astropy spectral-cube pyspeckit


Verification
~~~~~~~~~~~~~
To verify the installation, open a Python interpreter and import MUFASA:

.. code-block:: python

    import mufasa
    print(mufasa.__version__)

If no errors occur, MUFASA is installed correctly.
