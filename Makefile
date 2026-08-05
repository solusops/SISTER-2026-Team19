# ==============================================================================
# Makefile for LaTeX Paper Compilation
# ==============================================================================

MAIN = main
BUILD_DIR = build
PDF = $(MAIN).pdf

.PHONY: all clean watch pvc diff

all: $(PDF)

$(PDF): $(MAIN).tex references.bib $(wildcard sections/*.tex) $(wildcard macros/*.tex)
	latexmk -pdf -outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf ./$(PDF)

watch pvc:
	latexmk -pdf -pvc -outdir=$(BUILD_DIR) $(MAIN).tex

clean:
	latexmk -C -outdir=$(BUILD_DIR) $(MAIN).tex
	rm -rf $(BUILD_DIR) $(PDF)

# Generate latexdiff against git revision (e.g. make diff REF=HEAD~1)
REF ?= HEAD~1
diff:
	git-latexdiff --pdf --out=$(BUILD_DIR)/diff.pdf $(REF) -- $(MAIN).tex
