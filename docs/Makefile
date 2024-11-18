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

# Variables
#SPHINXAPIDOC  ?= sphinx-apidoc
#SOURCEDIR     = source
#BUILDDIR      = build
#TEMPLATEDIR   = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/templates
#APIDESTDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/api
#PACKAGEDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/mufasa
#EXCLUDE       = mufasa/__pycache__ mufasa/tests  # Exclude patterns

# Generate API .rst files
#apidocs:
#	@$(SPHINXAPIDOC) -o "$(APIDESTDIR)" "$(PACKAGEDIR)" --templatedir "$(TEMPLATEDIR)" --separate



# Help
#help:
#	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean generated .rst and build files
#clean:
#	@rm -rf "$(BUILDDIR)" "$(APIDESTDIR)"/*.rst

#clean:
#	@find "$(APIDESTDIR)" -type f -name '*.rst' ! -name 'modules.rst' -delete
#	@rm -rf "$(BUILDDIR)"



clean:
#	@rm -rf "$(BUILDDIR)"/*.html "$(BUILDDIR)"/*.js "$(BUILDDIR)"/*.css


# Generate API .rst files
#apidocs:
#	@$(SPHINXAPIDOC) -o "$(APIDESTDIR)" "$(PACKAGEDIR)" --templatedir "$(TEMPLATEDIR)" --separate
#	@rm -f "$(APIDESTDIR)/__pycache__.rst" "$(APIDESTDIR)/tests.rst"



# Catch-all target: route all unknown targets to Sphinx using "make mode".
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# All-in-one target
#all: clean apidocs html


# Variables
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
TEMPLATEDIR   = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/templates
APIDESTDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/api
PACKAGEDIR    = /Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/mufasa

# Rename .rst files
rename:
	python rename_rst.py

# All-in-one target
all: clean apidocs html

# Build HTML
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Clean build files
clean:
	@find "$(APIDESTDIR)" -type f -name '*.rst' ! -name 'modules.rst' -delete
	@rm -rf "$(BUILDDIR)"

# Generate API .rst files
apidocs:
	@$(SPHINXAPIDOC) -o "$(APIDESTDIR)" "$(PACKAGEDIR)" --templatedir "$(TEMPLATEDIR)" --separate

# Catch-all target: Route unknown targets to Sphinx only if they are not custom targets
.PHONY: all clean html apidocs
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
