# Build the preprint PDF.
# Requires: pdflatex, bibtex (TeX Live) or latexmk.
# Overleaf: set main.tex as the root; no Makefile needed.

MAIN := main
OUT  := build

.PHONY: all pdf clean watch arxiv-bundle

all: pdf

pdf: $(OUT)/$(MAIN).pdf

$(OUT)/$(MAIN).pdf: $(MAIN).tex references.bib arxiv.sty
	@mkdir -p $(OUT)
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(OUT) $(MAIN).tex

watch:
	@mkdir -p $(OUT)
	latexmk -pdf -pvc -interaction=nonstopmode -outdir=$(OUT) $(MAIN).tex

# Zip suitable for arXiv upload (includes .bbl; no .bib required on their end).
arxiv-bundle: pdf
	@mkdir -p $(OUT)/arxiv-src
	cp $(MAIN).tex arxiv.sty references.bib $(OUT)/$(MAIN).bbl $(OUT)/arxiv-src/
	@if [ -d figures ] && ls figures/* >/dev/null 2>&1; then cp -r figures $(OUT)/arxiv-src/; fi
	@# Point bibliography at the bundled .bbl for a self-contained submit tree:
	@echo "Bundle ready under $(OUT)/arxiv-src/ — see README for .bbl paste option."
	cd $(OUT) && zip -r arxiv-bundle.zip arxiv-src

clean:
	latexmk -C -outdir=$(OUT) $(MAIN).tex 2>/dev/null || true
	rm -rf $(OUT)
