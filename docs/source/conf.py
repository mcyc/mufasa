# Configuration file for the Sphinx documentation builder.
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project's Python package to sys.path for autodoc
sys.path.insert(0, os.path.abspath('../mufasa'))
sys.path.insert(0, os.path.abspath('../../mufasa'))

# -- Project information ------------------------------------------------------
project = 'MUFASA'
release = 'v1.4.0-dev'
copyright = '2024, Mike Chen'
author = 'Mike Chen'

# Use Astropy's configuration setup
from sphinx_astropy.conf.v2 import *  # Provides confv2 settings from sphinx-astropy

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Automatically document code
    'sphinx.ext.autosummary',   # Create summary tables
    'sphinx.ext.intersphinx',   # Link to other projects' documentation
    'sphinx.ext.napoleon',      # Support Google-style docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx_copybutton',        # Add a "copy" button to code blocks
    'nbsphinx',                 # Support Jupyter Notebooks (optional)
    # 'sphinx_search.extension',  # Uncomment if search functionality is used
]

# Map external documentation for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Exclude patterns to ignore during the build process
exclude_patterns = [
    'api/mufasa/__pycache__/*',
    'api/mufasa/tests/*',
    'api/setup.rst',
    'api/tests/*',
    '**/__pycache__/*',
]

# Disable prepending module names to functions/methods in the documentation
add_module_names = False

# -- HTML output configuration ------------------------------------------------
html_theme = "pydata_sphinx_theme"

# Customize theme options
html_theme_options = {
    "show_nav_level": 2,          # Breadcrumb depth for navigation
    "navigation_depth": 3,        # Sidebar navigation depth
    "collapse_navigation": False, # Keep sidebar expanded
    "navbar_align": "left",       # Align navbar to the left
    "github_url": "https://github.com/mcyc/mufasa",  # Link to GitHub repository
    "show_prev_next": False,      # Disable prev/next buttons
    "logo": {
        "text": "MUFASA",         # Add site title in the navbar
    },
}

# Customize sidebars for specific pages
html_sidebars = {
    "quick_start": [],     # Disable sidebar for "Quick Start"
    "installation": [],    # Disable sidebar for "Installation"
    "guides": [],          # Disable sidebar for "Guides"
    #"tutorials/**": ["globaltoc.html", "searchbox.html"],  # Use global TOC for tutorials
    #"**": ["globaltoc.html", "searchbox.html"],           # Default global TOC
}

# Set static and CSS paths
html_static_path = ["_static"]          # Directory for custom static assets
html_css_files = ["custom.css"]         # Custom CSS file
html_favicon = '_static/favicon.ico'    # Favicon path
html_logo = "_static/logo.svg"          # Logo path
html_title = "MUFASA DOC"               # HTML document title

# -- Autodoc configuration ----------------------------------------------------
autodoc_default_options = {
    "members": True,                  # Document all members
    "inherited-members": False,       # Skip inherited methods
    "undoc-members": True,            # Include undocumented members
    "show-inheritance": True,         # Show inheritance diagrams
    "private-members": False,         # Skip private members (e.g., _private)
    "special-members": "__init__",    # Include __init__ docstrings
    "exclude-members": "__weakref__", # Exclude __weakref__
}

# Adjust how type hints are displayed
autodoc_typehints = "description"  # Display type hints in descriptions

# Enable automatic summary tables
autosummary_generate = True

# -- Enable breadcrumbs -------------------------------------------------------
html_theme_options.update({
    "show_nav_level": 1,  # Ensure breadcrumbs appear for top-level sections
})
