# Variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SCRIPTSDIR    = scripts
APIDIR        = $(SOURCEDIR)/api
GENERATEDDIR  = $(APIDIR)/generated

.DEFAULT_GOAL := html

# Targets

all: apidocs html

check-deps:
	@command -v python > /dev/null || { echo "Python not found. Aborting."; exit 1; }
	@python -c "import sphinx" 2>/dev/null || { echo "Sphinx not installed. Aborting."; exit 1; }

generate-api-reference: check-deps
	@echo "Generating API_REFERENCE..."
	python $(SCRIPTSDIR)/generate_api_reference.py

apidocs: check-deps
	@echo "Generating .rst files for API documentation..."
	python $(SCRIPTSDIR)/generate_rst_files.py

html: check-deps
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html -j auto "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

view-html:
	@echo "Opening built HTML documentation..."
	xdg-open "$(BUILDDIR)/index.html" || open "$(BUILDDIR)/index.html" || echo "Unable to open HTML documentation."

clean:
	@echo "Cleaning build and generated files..."
	rm -rf "$(BUILDDIR)"
	rm -rf "$(APIDIR)/*.rst"

distclean: confirm-distclean clean
	@echo "WARNING: Removing all auto-generated .rst files..."
	rm -f "$(GENERATEDDIR)/*.rst"
	rm -f "$(SCRIPTSDIR)/api_reference.py"

confirm-distclean:
	@read -p "Are you sure you want to perform a distclean? This will remove all generated files (y/n): " confirm; \
	if [ "$$confirm" != "y" ]; then \
		echo "Distclean aborted."; \
		exit 1; \
	fi

help:
	@echo "Makefile targets:"
	@echo "  make generate-api-reference  Generate or overwrite API_REFERENCE dynamically."
	@echo "  make apidocs                 Generate .rst files for API documentation."
	@echo "  make html                    Build the HTML documentation."
	@echo "  make view-html               Open the built HTML documentation in a browser."
	@echo "  make clean                   Clean up build and generated files."
	@echo "  make distclean               Clean everything, including auto-generated files (requires confirmation)."
	@echo "  make help                    Show this help message."
