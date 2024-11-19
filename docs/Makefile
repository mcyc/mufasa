# Minimal makefile for Sphinx documentation
#


# Variables
#SPHINXOPTS    ?=
#SPHINXBUILD   ?= sphinx-build
#SPHINXAPIDOC  ?= sphinx-apidoc
#SOURCEDIR     = source
#BUILDDIR      = build
#TEMPLATEDIR   = templates  # Directory for custom templates
#APIDESTDIR    = $(SOURCEDIR)/api  # Destination for generated .rst files
#PACKAGEDIR    = ../mufasa  # Relative path to the Python package
#EXCLUDE       = mufasa/__pycache__ mufasa/tests  # Exclude patterns

#TEMPLATEDIR   = $(SOURCEDIR)/templates  # Directory for custom templates
#APIDESTDIR    = $(SOURCEDIR)/api  # Destination for generated .rst files
#PACKAGEDIR    = ../mufasa  # Relative path to the Python package


# Variables
#SPHINXOPTS    ?=
#SPHINXBUILD   ?= sphinx-build
#SPHINXAPIDOC  ?= sphinx-apidoc
#SOURCEDIR     = source
#BUILDDIR      = build
#TEMPLATEDIR   = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/templates
#APIDESTDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/api
#PACKAGEDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/mufasa


# Sphinx configuration variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc

# Directory paths
SOURCEDIR     = source
BUILDDIR      = build
APIDESTDIR    = $(SOURCEDIR)/api
TEMPLATEDIR   = templates
PACKAGEDIR    = ../mufasa

.PHONY: all clean distclean html apidocs

all: clean apidocs html

html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@echo "Cleaning build directory..."
	rm -rf "$(BUILDDIR)"
	find "$(APIDESTDIR)" -type f -name '*.rst' ! -name 'modules.rst' -delete

apidocs:
	@echo "Generating API documentation..."
	@if [ -d "$(PACKAGEDIR)" ]; then \
		$(SPHINXAPIDOC) -o "$(APIDESTDIR)" "$(PACKAGEDIR)" "$(PACKAGEDIR)/setup" "$(PACKAGEDIR)/tests" "$(PACKAGEDIR)/__pycache__" --templatedir "$(TEMPLATEDIR)" --separate; \
	else \
		echo "Error: Package directory '$(PACKAGEDIR)' not found!"; \
		exit 1; \
	fi
