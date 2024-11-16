# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Adjust if your source is in a different folder

project = 'MUFASA'
copyright = '2024, Mike Chen'
author = 'Mike Chen'
release = 'v1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',   # Links to other projects (e.g., NumPy, Astropy)
    'numpydoc',                 # For advanced NumPy-style parsing
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
html_static_path = ['_static']
