# VeriPet IEEE Research Article

This folder contains the English IEEE-style research article for the PPG-CC / UNIFESP Artificial Intelligence course.

## Contents

- `main.tex`: article source using the IEEE conference template.
- `IEEEtran.cls`: IEEE class file copied from the provided template.
- `references.bib`: bibliography adapted from the TCC source package.
- `figures/`: figures used in the article.

## Expected Format

The article is written as an original research article and follows the requested structure:

- introduction and motivation;
- research gaps;
- related work and state of the art;
- detailed methodology;
- materials and methods;
- experimental results and evaluation;
- scientific implications and conclusion;
- reproducibility with repository/code/results references.

## Compilation

Use pdfLaTeX and BibTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The target length is 8 IEEE pages.
