# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../mufasa'))
sys.path.insert(0, os.path.abspath('../../mufasa'))

project = 'MUFASA'
release = 'v1.4.0-dev'
copyright = '2024, Mike Chen'
author = 'Mike Chen'

from sphinx_astropy.conf.v2 import *  # Enable confv2

# Theme configuration
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 1,          # Display navigation up to this level
    "navigation_depth": 2,        # Sidebar depth
    "collapse_navigation": True,  # Collapsible sidebar
    "navbar_align": "left",
    "github_url": "https://github.com/mcyc/mufasa",  # Link to GitHub
    "show_prev_next": False,      # Hide prev/next buttons
    "logo": {
        "text": "MUFASA",  # Add the title text
    }
}

# Custom logo
html_favicon = '_static/favicon.ico'
html_logo = "_static/logo.svg"  # Path to your logo

# Add paths to static files
html_static_path = ["_static"]

# Customize CSS (optional)
html_css_files = ["custom.css"]

html_title = "MUFASA DOC"

# Extensions to enable
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',  # For Jupyter Notebook integration (optional)
    'sphinx_copybutton',  # Add the copybutton extension
    "sphinx.ext.autosummary",  # Enables summary tables
    "sphinx.ext.viewcode",     # Adds links to source code
    #"sphinx_search.extension",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

exclude_patterns = [
    'api/mufasa/__pycache__/*',
    'api/mufasa/tests/*',
    'api/setup.rst',
    'api/tests/*',
    '**/__pycache__/*',
]


add_module_names = False

# For using Astropy's Bootstrap
# Use the theme directly
#html_theme = 'bootstrap-astropy'
#html_theme = 'sphinx-astropy'

# Optional theme customizations
#html_theme_options = {
#    'logotext1': 'MUFA',  # Custom text
#    'logotext2': 'SA',    # Highlighted part of the name
#}

autodoc_default_options = {
    "members": True,
    "inherited-members": False,  # Do not document inherited methods
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,  # Exclude private attributes (e.g., _private_var)
    "special-members": False,  # Exclude dunder methods (e.g., __init__)
    "special-members": "__init__",  # Include __init__ docstrings
    "exclude-members": "__weakref__",
    #'exclude-members': 'read_model_fit, get_model_fit',
}



autodoc_typehints = "description"  # Show type hints in descriptions

#autodoc_typehints = "none"  # Disable showing type hints
autosummary_generate = True  # Automatically generate summary tables

# Enable breadcrumbs in pydata_sphinx_theme
html_theme_options.update({
    "show_nav_level": 1,  # Show breadcrumbs for the top-level section
})


