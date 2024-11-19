Get Starting
============

Getting Started
---------------
MUFASA automates multi-component, astrophysical spectral fitting. Below are some examples to help you get started.

1. **Loading a Spectral Cube**

   Load your data cube using the `Region` class, which hosts the data and handles all the fitting-related tasks:

   .. code-block:: python

       from mufasa import master_fitter as mf

       reg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
       print(f"Loaded cube with dimensions: {reg.data.shape}")

2. **Performing a Multi-Component Fit**

   Fit an NH₃ (1,1) spectrum with up to two velocity components using MUFASA's standard recipe:

   .. code-block:: python

       reg.master_2comp_fit(snr_min=3)

   The results will be saved automatically in the specified `paraDir`.

3. **Visualizing the Fits**

   To generate a grid of plots for the fitted spectra at a specific pixel `(x, y)`:

   .. code-block:: python

       from mufasa import UltraCube as UCube

       ucube = UCube.UCubePlus(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
       ucube.read_model_fit(ncomps=[1, 2])
       ucube.plot_fits_grid(x, y, ncomp=2, size=3, xlim=None, ylim=None)

4. **Visualizing Fits in 3D**

   Plot best-fit models in PPV space using a 3D scatter plot:

   .. code-block:: python

       reg.plot_ppv_scatter(savepath, vel_scale=0.5)

Common Use Cases
----------------

1. **Supported Spectral Models**

   MUFASA currently supports the following spectral line models:

   - **NH₃ (1,1)** 2-component model (``fittype='nh3_multi_v'``).
   - **N₂H⁺ (1-0)** 2-component model (``fittype='n2hp_multi_v'``).

Advanced Usage
--------------
For advanced configurations, including custom spectral models or detailed parameter tuning, refer to the `API Reference` section of this documentation.
