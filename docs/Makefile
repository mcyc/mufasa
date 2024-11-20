# Sphinx configuration variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc

# Simplified directory paths
SOURCEDIR     = source
BUILDDIR      = build
APIDESTDIR    = source/api
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

apidocs:
	@echo "Generating API documentation..."
	@if [ -d "$(PACKAGEDIR)" ]; then \
		$(SPHINXAPIDOC) -o "$(APIDESTDIR)" "$(PACKAGEDIR)" --templatedir "$(TEMPLATEDIR)" --separate; \
	else \
		echo "Error: Package directory '$(PACKAGEDIR)' not found!"; \
		exit 1; \
	fi

distclean: clean
	@echo "WARNING: This will delete all files in '$(APIDESTDIR)' and '$(BUILDDIR)'!"
	@read -p "Are you sure you want to continue? [y/N]: " CONFIRM && [ "$$CONFIRM" = "y" ] || { echo "Aborting distclean."; exit 1; }
	@echo "Performing deep clean..."
	rm -rf "$(APIDESTDIR)"

