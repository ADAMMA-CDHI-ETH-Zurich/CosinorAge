# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    = -E -D html_theme_options.docsearch_container_id=null
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
HTMLDIR       = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html clean

# Custom html target to copy files directly to docs folder - always rebuilds
html: clean
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(HTMLDIR)" $(SPHINXOPTS) $(O)

# Clean target to remove both build and docs contents
clean:
	rm -rf "$(BUILDDIR)"
	rm -rf "$(HTMLDIR)"/*

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)