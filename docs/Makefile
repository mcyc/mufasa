# Variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SCRIPTSDIR    = scripts

.DEFAULT_GOAL := html

# Targets

all: apidocs html

generate-api-reference:
	@echo "Generating API_REFERENCE..."
	python $(SCRIPTSDIR)/generate_api_reference.py

apidocs:
	@echo "Generating .rst files for API documentation..."
	python $(SCRIPTSDIR)/generate_rst_files.py

html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@echo "Cleaning build and generated files..."
	rm -rf "$(BUILDDIR)"
	rm -rf "$(SOURCEDIR)/api/generated"

distclean: confirm-distclean clean
	@echo "WARNING: Removing all auto-generated .rst files..."
	rm -f "$(SOURCEDIR)/api/generated/*.rst"
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
	@echo "  make clean                   Clean up build and generated files."
	@echo "  make distclean               Clean everything, including auto-generated files (requires confirmation)."
	@echo "  make help                    Show this help message."
