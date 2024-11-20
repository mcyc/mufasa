Visualization
=============

This tutorial explains how to visualize spectral fits and explore the fitted parameters.

Preparing for Visualization
---------------------------
Before visualizing your results, ensure that the data and results are properly loaded. Refer to the :doc:`Loading Data and Results <load_data_n_results>` tutorial for instructions on loading the saved results and initializing the :class:`~mufasa.master_fitter.Region` object.

Plotting Fitted Spectra
-----------------------
To plot the fitted spectra for a specific position ``(x, y)`` in a data cube:

.. code-block:: python

    # Plot the fits for position (x, y) on a 3x3 grid for a 2-component model
    region.ucube.plot_fits_grid(x, y, ncomp=2, size=3, xlim=None, ylim=None)

3D Scatter Plots
----------------
To visualize the fitted parameters in 3D, such as in position-position-velocity (PPV) space:

.. code-block:: python

    # Create an interactive 3D scatter plot of the fitted results
    region.plot_ppv_scatter(savepath="scatter_plot.html", vel_scale=0.5, showfig=True, auto_open_html=False)

Next Steps
----------
For advanced visualization techniques or to build custom workflows, refer to :doc:`Custom Usage <custom_usage>`.
