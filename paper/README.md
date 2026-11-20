# Preprint template (arXiv / Zenodo / OSF)

Single-column LaTeX preprint for **causal-automata**, styled with
[`kourgeorge/arxiv-style`](https://github.com/kourgeorge/arxiv-style) (MIT).
The same PDF works for **arXiv**, **Zenodo**, and **OSF** — write once, deposit
wherever you prefer.

## Files

| Path | Role |
|------|------|
| `main.tex` | Manuscript (edit this) |
| `references.bib` | BibTeX sources |
| `arxiv.sty` | Preprint style |
| `figures/` | Drop PNGs/PDFs here; `\includegraphics` from `main.tex` |
| `Makefile` | Local `latexmk` build → `build/main.pdf` |

Placeholder sections and Benchpress numbers are seeded from the project README /
docs so you can start drafting immediately. Search for `TODO` in `main.tex`.

## Build locally

```bash
# TeX Live / MacTeX / MiKTeX with latexmk
make -C paper
# → paper/build/main.pdf
```

Continuous rebuild:

```bash
make -C paper watch
```

## Build on Overleaf (fastest if you lack TeX)

1. New project → Upload `main.tex`, `references.bib`, `arxiv.sty`, and `figures/`.
2. Set **Menu → Main document** to `main.tex`.
3. Compiler: **pdfLaTeX** (default).

## Submit to arXiv

1. Build so `build/main.bbl` exists (`make -C paper`).
2. arXiv does **not** run BibTeX. Either:
   - upload `main.tex`, `arxiv.sty`, `figures/`, **and** `main.bbl`, or
   - paste the contents of `main.bbl` into `main.tex` in place of
     `\bibliography{references}` / `\bibliographystyle{...}` and comment those
     two lines out.
3. `make -C paper arxiv-bundle` prepares `build/arxiv-bundle.zip` as a starting
   point (still verify the `.bbl` is included).
4. Category suggestion for this work: **stat.ML** (also consider **cs.LG**).

## Deposit on Zenodo or OSF

1. Build `build/main.pdf`.
2. Upload the PDF (and optionally the `.tex` source zip).
3. Add metadata (title, author, abstract, license — Apache-2.0 matches the
   software, or choose CC-BY for the text if you prefer).
4. Reserve a DOI, then put that DOI into the “Data and code availability”
   section and rebuild once before the final deposit.

Zenodo/OSF need only a good PDF; they do not require LaTeX. Use this template
anyway so you can later push the same source to arXiv without reformatting.

## License note

`arxiv.sty` is MIT (© George Kour / arxiv-style). Manuscript text and figures
you add are yours; the surrounding software repo remains Apache-2.0.
