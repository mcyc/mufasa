# MUFASA
MUlti-component Fitter for Astrophysical Spectral Applications.

## Reference

Please cite the following paper when using the code:
1. Chen, M. C.-Y. et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" ApJ ([2020](https://ui.adsabs.harvard.edu/link_gateway/2020ApJ...891...84C/doi:10.3847/1538-4357/ab7378)).

## Installation

To install the latest version of ```MUFASA``` from this repository, run:

```
python setup.py install
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


## Getting Started

### Minimum Working Example

To perform a two-component NH<sub>3</sub> (1,1) fit automatically, simply run the following: 

```
from mufasa import master_fitter as mf
uReg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
uReg.master_2comp_fit(snr_min=0)
```

In the example above, ```cubePath``` is the path to the FITS data cube, ```paraNameRoot``` is the commmon 'root' name to all the outputfiles, and ```paraDir``` is the directory of all the outputfiles. The species being fit is specified by```fittype```. N<sub>2</sub>H+ (1-0) can be fit instead of NH<sub>3</sub> (1,1) by setting ```fittype='n2hp_multi_v'```. If one wishes to fit pixels only above a certain signal-to-noise-ratio (SNR) threshold, use ```snr_min``` to set such a threshold.



