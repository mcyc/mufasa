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

```MUFASA``` runs on ```python > v3.7``` and depends on the following packages:

- ```numpy >= v1.19.2```
- ```scipy >= v1.7.3```
- ```skimage >= v0.17.2```
- ```spectral_cube >= v0.6.0```
- ```pyspeckit >= v1.0.1```
- ```reproject >= v0.7.1```
- ```FITS_tools >= v0.2```

If you are running a later version of Python, for example, ```Python 3.11```, you likely will have to install the latest versions of ```pyspeckit``` and  ```FITS_tools``` directly from their respective GitHub repository. The `setup.py` for `MUFASA >= v1.4.0` takes care of such a depdendcy automatically with `pip` installs.

## Getting Started

### Running 2-component fits

To perform an NH<sub>3</sub> (1,1) fit automatically, up to two components, simply run the following: 

```
from mufasa import master_fitter as mf
reg = mf.Region(cubePath, paraNameRoot=paraNameRoot, paraDir=paraDir, fittype='nh3_multi_v')
reg.master_2comp_fit(snr_min=0)
```

In the example above, ```cubePath``` is the path to the FITS data cube, ```paraNameRoot``` is the common 'root' name to all the output files, ```paraDir``` is the directory of all the outputfiles, and ```fittype``` is the name of the line model to be fitted. 

```MUFASA``` currently offers two spectral line models, specified by the ```fittype``` argument:
- ```nh3_multi_v```: multi-component NH<sub>3</sub> (1,1) model
- ```n2hp_multi_v```: multi-component N<sub>2</sub>H<sup>+</sup> (1-0) model

If one wishes to fit pixels only above a specific signal-to-noise-ratio (SNR) threshold, one can specify such a threshold using the ```snr_min``` argument.

### Visualize the fits

To quickly visualize the saved fits, one can plot the spectral model at position x, y, and its neighbors with:

```
from mufasa import UltraCube as UCube

# read the fits from the saved files
ucube = UCube.UCubePlus(cubePath, paraNameRoot=paraNameRoot, paraDir=paraDir, fittype='nh3_multi_v')
ucube.read_model_fit(ncomps=[1,2])

# visualize the fitted models (2 component model in this example)
ucube.plot_fits_grid(x,y, ncomp=2, size=3, xlim=None, ylim=None)
```

where the ```comp``` and ```size``` arguments in ```plot_fits_grid``` are the number of components in the model and the plot's grid size. For example, setting ```size=3``` results in a plot with 3x3 grid of spectral models centered on the position ```x``` and ```y```. To zoom in on the plot (e.g., to look closely at a hyperfine group), one can use the ```xlim``` and ```ylim``` arguments to constrain the plot's x and y limits. 



