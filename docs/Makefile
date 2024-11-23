# Sphinx configuration variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc

# Simplified directory paths
SOURCEDIR     = source
BUILDDIR      = build
APIDESTDIR    = source/api/generated
TEMPLATEDIR   = templates
PACKAGEDIR    = ../mufasa

# Verbosity control
VERBOSE ?= 0
ifeq ($(VERBOSE), 1)
    SILENT =
else
    SILENT = @
endif

.DEFAULT_GOAL := all

.PHONY: all clean distclean html apidocs check_env help

all: check_env clean apidocs html

html:
	$(SILENT)echo "Building HTML documentation..."
	$(SILENT)$(SPHINXBUILD) -j auto -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	$(SILENT)echo "Cleaning build directory..."
	$(SILENT)rm -rf "$(BUILDDIR)"

apidocs:
	@echo "Generating API documentation..."
	sphinx-apidoc -o source/api/generated ../mufasa --force --separate --no-toc
	python scripts/make_api_rst.py
	@echo "Customized API index.rst generated."

distclean: clean
	$(SILENT)echo "WARNING: Performing deep clean..."
	$(SILENT)rm -rf "$(APIDESTDIR)"/*.rst

check_env:
	$(SILENT)command -v $(SPHINXBUILD) > /dev/null || { echo "$(SPHINXBUILD) not found! Install Sphinx."; exit 1; }
	$(SILENT)command -v $(SPHINXAPIDOC) > /dev/null || { echo "$(SPHINXAPIDOC) not found! Install Sphinx."; exit 1; }

help:
	@echo "Usage:"
	@echo "  make           Build all (clean, apidocs, html)"
	@echo "  make html      Build HTML documentation"
	@echo "  make apidocs   Generate API documentation from mufasa source"
	@echo "  make clean     Remove build directory"
	@echo "  make distclean Deep clean, including API documentation"
	@echo "  make help      Show this help message"
