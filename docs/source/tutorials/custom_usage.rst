Custom Usage
============

Learn how to adapt and extend MUFASA for your specific use cases. This section includes examples and tips for using MUFASA in flexible ways, combining its tools, and integrating it with other libraries.

Contents
--------

- **Adapting Fit Parameters**: Customize MUFASA’s models for unique datasets.
- **Combining Modules**: Build workflows using multiple components.
- **Integrating with External Libraries**: Extend MUFASA’s functionality with external tools.

Adapting Fit Parameters
------------------------
MUFASA allows you to fine-tune its behavior to match your data's characteristics. For example, you can adjust the signal-to-noise ratio (SNR) threshold or other fitting parameters:

.. code-block:: python

    from mufasa import master_fitter as mf

    # Define file paths and parameters
    cubePath = "cube.fits"
    paraNameRoot = "custom_fit"
    paraDir = "output"

    # Initialize the Region object with custom parameters
    reg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')

    # Perform a fit with a custom SNR threshold
    reg.master_2comp_fit(snr_min=5)

Combining Modules
------------------
Use MUFASA’s tools together to build flexible workflows. For instance, you can process data with `UltraCube` and visualize results with built-in plotting functions:

.. code-block:: python

    from mufasa import UltraCube as UCube

    # Initialize the UltraCube object
    ucube = UCube.UCubePlus(cubePath, paraNameRoot="custom_usage", paraDir="output")

    # Read model fits and plot results
    ucube.read_model_fit(ncomps=[1, 2])
    ucube.plot_fits_grid(x=10, y=15, ncomp=2, size=3)

Integrating with External Libraries
------------------------------------
MUFASA can integrate seamlessly with external Python libraries like NumPy, Matplotlib, or Pandas for additional functionality:

.. code-block:: python

    import numpy as np
    from mufasa import UltraCube as UCube

    # Use UltraCube to read data and analyze it with NumPy
    ucube = UCube.UCubePlus(cubePath, paraNameRoot="integration_example", paraDir="output")
    data = ucube.read_data()

    # Perform custom analysis with NumPy
    mean_spectrum = np.mean(data, axis=0)
    print("Mean Spectrum:", mean_spectrum)
