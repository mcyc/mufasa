Visualization Tutorial
=======================

Learn how to visualize your spectral fits and data using MUFASA's built-in tools.

Contents
--------

- **Plotting Fitted Spectra**: Generate plots for spectral fits.
- **3D Visualization**: Create interactive 3D visualizations of your data.

Plotting Fitted Spectra
-----------------------
To plot the fitted spectra for a specific position `(x, y)`:

.. code-block:: python

    from mufasa import UltraCube as UCube

    ucube = UCube.UCubePlus(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
    ucube.read_model_fit(ncomps=[1, 2])

    # Plot the fits for position (x, y)
    ucube.plot_fits_grid(x=10, y=15, ncomp=2, size=3, xlim=None, ylim=None)

More examples and explanations will be added in future updates.
