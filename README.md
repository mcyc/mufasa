# MUFASA
MUlti-component Fitter for Astrophysical Spectral Applications.

## Reference

Please cite the following paper when using the code:
1. Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ ([2020](https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...891...84C/doi:10.3847/1538-4357/ab7378)).

## Installation

To install the latest version of ```MUFASA```, clone this repository and run the following in your local directory:

```
pip install -e .
```

To pip install a 'stable' release, run:
```
pip install mufasa
```
### Requirements

```MUFASA``` runs on ```python > v3.6``` and depends on the following packages:

- ```numpy > v1.19.2```

- ```skimage > v0.17.2```

- ```spectral_cube > v0.6.0```

- ```pyspeckit > v1.0.1```

- ```reproject > v0.7.1```

- ```FITS_tools > v0.2```

If you are running a later version of Python, for example, ```Python 3.11```, you likely will have to install the latest versions of ```pyspeckit``` and  ```FITS_tools``` directly from their respective GitHub repository. 

## Getting Started

### Minimum Working Example

To perform an NH<sub>3</sub> (1,1) fit automatically, up to two components, simply run the following: 

```
from mufasa import master_fitter as mf
reg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
reg.master_2comp_fit(snr_min=0)
```

In the example above, ```cubePath``` is the path to the FITS data cube, ```paraNameRoot``` is the common 'root' name to all the output files, ```paraDir``` is the directory of all the outputfiles, and ```fittype``` is the name of the line model to be fitted. 

```MUFASA``` currently offers two spectral line models, specified by the ```fittype``` argument:
- ```nh3_multi_v```: multi-component NH<sub>3</sub> (1,1) model
- ```n2hp_multi_v```: multi-component N<sub>2</sub>H<sup>+</sup> (1-0) model

If one wishes to fit pixels only above a specific signal-to-noise-ratio (SNR) threshold, one can specify such a threshold using the ```snr_min``` argument.



