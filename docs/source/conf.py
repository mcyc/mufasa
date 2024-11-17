# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust based on your `mufasa` location

project = 'MUFASA'
release = 'v1.4.0'
copyright = '2024, Mike Chen'
author = 'Mike Chen'

# Use the theme directly
html_theme = 'bootstrap-astropy'
#html_theme = 'sphinx-astropy'

# Remove the incorrect `html_theme_path` assignment
# It is unnecessary to specify the path manually when using a known installed theme.

# Optional theme customizations
html_theme_options = {
    'logotext1': 'MUFA',  # Custom text
    'logotext2': 'SA',    # Highlighted part of the name
    'logotext3': ':docs', # Additional tagline
}

# Add logo if available
html_logo = 'path/to/logo.png'  # Update with the actual path
html_favicon = 'path/to/favicon.ico'  # Update with the actual path

# Extensions to enable
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',  # For Jupyter Notebook integration (optional)
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}
