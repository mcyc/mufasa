"""
The `mufasa.visualization` module provides tools for visualizing spectral cubes,
fitted models, and residuals. These tools are designed to work with
the `UltraCube` framework and its associated spectral fitting workflows.

Main Features
-------------
- Plotting individual spectra with model fits.
- Creating grids of spectral plots for spatial regions of interest.
- Visualizing residuals and fitted parameter maps.
- Supporting flexible customization of plot aesthetics and labels.

Notes
-----
This module relies on `matplotlib` for visualization and integrates directly with
the `UltraCube` class for seamless plotting of spectral and spatial data.

"""

from ._version import __version__