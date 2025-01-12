from setuptools import setup, find_packages
import os
import re

# Dynamically locate the metadata file in the same directory
metadata_file = os.path.join(os.path.dirname(__file__), "mufasa/_metadata.py")

def get_metadata():
    metadata = {}
    with open(metadata_file, "r") as f:
        for line in f:
            match = re.match(r"__([a-z_]+)__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata

# Load metadata
metadata = get_metadata()

# Define the setup configuration
if __name__ == "__main__":
    setup(
        name=metadata["project"],  # e.g., mufasa
        version=metadata["version"],  # e.g., 1.4.0
        description="MUlti-component Fitter for Astrophysical Spectral Applications",
        author=metadata["author"],  # Michael Chun-Yuan Chen
        url=metadata["github_url"],  # GitHub URL
        packages=find_packages(),
        install_requires=[
            "numpy>=1.24.4,<2",
            "astropy",
            "matplotlib",
            "scipy>=1.7.3",
            "scikit-image>=0.17.2",
            "spectral-cube>=0.6.0",
            "radio-beam",
            "pvextractor",
            "pandas", #if enforcing 2.0, use Python > 3.11.2, Numpy > 1.24.2, Astropy > 5.2.1
            "plotly",
            "nbformat",
            "reproject>=0.7.1",
            "dask[complete]>2024.0",  # no longer pinned to <2024.0"
            "graphviz",
            "pyspeckit @ git+https://github.com/pyspeckit/pyspeckit.git@342713015af8cbe55c31494d6f2c446ed75521a2#egg=pyspeckit",
            "FITS_tools @ git+https://github.com/keflavich/FITS_tools.git@b1fe5166ccf8a43105efe8201e37ab5993e880be#egg=FITS_tools",
        ],
        extras_require={
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
            "dev": ["pytest", "flake8"],
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
    )
