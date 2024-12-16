# MUFASA
MUlti-component Fitter for Astrophysical Spectral Applications.

[![Documentation Status](https://readthedocs.org/projects/mufasa/badge/?version=latest)](https://mufasa.readthedocs.io/en/latest/)

---

## Documentation

For detailed documentation, including installation instructions, usage examples, and API details, visit the MUFASA [Documentation](https://mufasa.readthedocs.io/en/latest/) on Read the Docs (RTD).

---

## Reference

If you use MUFASA in your work, please cite the following paper:
1. Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ ([2020](https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...891...84C/doi:10.3847/1538-4357/ab7378)).

---

## Installation

To install the latest version of `MUFASA`, clone this repository and run the following in your local directory:

```bash
pip install -e .
```

### Requirements

```MUFASA``` runs on ```python > v3.8```. For more details on the installation and dependencies, please see the [Install](https://mufasa.readthedocs.io/en/latest/installation.html) page on RTD.

If you are running a later version of Python, for example, ```Python 3.11```, you likely will have to install the latest versions of ```pyspeckit``` and  ```FITS_tools``` directly from their respective GitHub repository. The `setup.py` for `MUFASA >= v1.4.0` takes care of such a dependency automatically with `pip` installs.

## Getting Started

To get started quickly, please see MUFASA's [Quick Start](https://mufasa.readthedocs.io/en/latest/index.html#quick-start) page on RTD.

## Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
