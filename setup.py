from setuptools import setup, find_packages
import os

pkg_vars = {}

# Safely read the version file
version_file = "mufasa/_version.py"
if os.path.exists(version_file):
    with open(version_file) as fp:
        exec(fp.read(), pkg_vars)
else:
    raise RuntimeError(f"Version file '{version_file}' not found!")

# Define the setup configuration
if __name__ == "__main__":
    setup(
        name='mufasa',
        version=pkg_vars['__version__'],  # Dynamically load version
        description='MUlti-component Fitter for Astrophysical Spectral Applications',
        author='Michael Chun-Yuan Chen',
        author_email='mkid.chen@gmail.com',
        url='https://github.com/mcyc/mufasa',
        packages=find_packages(),  # Automatically find sub-packages
        install_requires=[
            'numpy>=1.19.2',
            'astropy',
            'matplotlib',
            'scipy>=1.7.3',
            'scikit-image>=0.17.2',
            'spectral-cube>=0.6.0',
            'radio-beam',
            'pvextractor',
            'pandas',
            'plotly',
            'nbformat',
            'reproject>=0.7.1',
            'pyspeckit @ git+https://github.com/pyspeckit/pyspeckit.git@342713015af8cbe55c31494d6f2c446ed75521a2#egg=pyspeckit',
            'FITS_tools @ git+https://github.com/keflavich/FITS_tools.git@b1fe5166ccf8a43105efe8201e37ab5993e880be#egg=FITS_tools',
        ],
        extras_require={  # Dependencies for building the documentation
            "docs": [
                "sphinx>=4.0",
                "sphinx-rtd-theme>=1.0,<2.0",
                "sphinx-autodoc-typehints",
                "nbsphinx",
                "sphinx-astropy>=1.8",
                "sphinx-copybutton>=0.5.0",
                "pydata-sphinx-theme>=0.13.0",
                "numpydoc>=1.1.0,<2.0",
                "sphinx-issues>=2.0",
                "sphinxext-opengraph>=0.4.0",
            ],
            "dev": [  # Dependencies for development and testing
                "pytest",
                "flake8",
            ],
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.7',  # Specify minimum Python version
    )
