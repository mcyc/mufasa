# Configuration file for the Sphinx documentation builder.
import os
import sys

# Import project metadata
from mufasa._metadata import __title__, __version__, __author__, __copyright__, __github_url__

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../mufasa'))
sys.path.insert(0, os.path.abspath('../sphinxext'))

# -- Project information ------------------------------------------------------
project = __title__
release = __version__
version = ".".join(release.split(".")[:2])
author = __author__
copyright = __copyright__

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
    'sphinx_issues',
    'sphinx.ext.doctest',
    'numpydoc',
    'sphinxext.opengraph',
    'nbsphinx',
    'autoshortsummary',
    #'sphinx_search.extension',  # Add search extension
]


autosummary_generate = True
autosummary_imported_members = False
numpydoc_show_class_members = False  # Suppresses showing members for classes
napoleon_google_docstring = False
napoleon_numpy_docstring = True


autodoc_default_options = {
    'members': True,                # Document all members
    'undoc-members': False,         # Skip undocumented members
    'show-inheritance': True,       # Show class inheritance diagrams
    'inherited-members': False,     # Skip inherited members
    'special-members': False,       # Skip special methods (e.g., __init__)
    'exclude-members': '__weakref__'  # Exclude undesired members
}

# -- HTML output options ------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {"text": "MUFASA"},
    "github_url": "https://github.com/mcyc/mufasa",
    "show_toc_level": 4,
    "navbar_align": "left",
    "collapse_navigation": True,  # Enable collapsible sidebar
    "navigation_depth": 4,  # Ensure the sidebar reflects submodules
    "show_nav_level": 4,
    #"navbar_end": ["version-switcher"],
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'spectral_cube': ('https://spectral-cube.readthedocs.io/en/latest/', None),
}

templates_path = ['_templates']
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = '_static/favicon.ico'
html_logo = "_static/logo.png"

html_sidebars = {
    "quick_start": [],
    "installation": [],
    "guides": [],
}

nitpick_ignore = [
    ('class', 'SpectralCube'),
]

#exclude_patterns = ['api/generated/generated*']

# Remove unused gallery config if not in use
# sphinx_gallery_conf = {
#     'doc_module': 'mufasa',
#     'backreferences_dir': os.path.join('modules', 'generated'),
#     'examples_dirs': os.path.join('..', 'examples'),
#     'gallery_dirs': 'auto_examples',
# }

