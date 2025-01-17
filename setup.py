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
        name=metadata["project"],
        version=metadata["version"],
        description="MUlti-component Fitter for Astrophysical Spectral Applications",
        author=metadata["author"],
        url=metadata["github_url"], 
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
            "pandas",
            "plotly",
            "nbformat",
            "reproject>=0.7.1",
            "pyspeckit @ git+https://github.com/pyspeckit/pyspeckit.git@2d19ddfb965a99b1fdf094517d8c905c6527b3b1#egg=pyspeckit",
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
