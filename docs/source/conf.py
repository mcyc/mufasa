# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

#master_doc = "index"

import os
#print(f"Current working directory: {os.getcwd()}")
#print(f"Source directory: {os.path.abspath('.')}")
#print(f"Master document path: {os.path.abspath(master_doc)}")
import sys
#sys.path.insert(0, os.path.abspath('../../'))
#sys.path.insert(0, os.path.abspath('../../mufasa'))

import os
import sys
sys.path.insert(0, os.path.abspath("../../mufasa"))


#sys.path.insert(0, os.path.abspath("/Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/mufasa"))




project = 'MUFASA'
copyright = '2024, Mike Chen'
author = 'Mike Chen'
release = 'v1.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',   # Links to other projects (e.g., NumPy, Astropy)
    'numpydoc',                 # For advanced NumPy-style parsing
]



templates_path = ['_templates']

# Remove 'setup' if it exists
autodoc_mock_imports = ['setup']

exclude_patterns = ['**/setup.rst', '**/mufasa.setup.rst']

#exclude_patterns = ['setup']

# the following commend seems to work for what's needed
# sphinx-apidoc -o docs/source/api mufasa -f -e

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/mcyc/mufasa",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/source/",
}

pygments_style = 'friendly'#"sphinx"  # Alternatives: "friendly", "default", "native"
highlight_language = "python"

#html_static_path = ['_static']
html_static_path = ['_static']
html_css_files = [
    'styles/pygments.css',
]

autosummary_generate = True
add_module_names = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}



