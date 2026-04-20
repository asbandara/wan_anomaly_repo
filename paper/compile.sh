#!/bin/bash
# Compile the ICLR-style paper.
#
# Requires the ICLR 2025 style files in this directory:
#   iclr2025_conference.sty
#   iclr2025_conference.bst  (or use natbib/plain for testing)
#
# Download the official ICLR 2025 LaTeX template from:
#   https://github.com/ICLR/Master-Template
# and copy iclr2025_conference.sty and iclr2025_conference.bst here.
#
# Usage:  bash compile.sh

set -e
cd "$(dirname "$0")"

# Use full path in case TeX Live is not on PATH (common after fresh install)
PDFLATEX=$(which pdflatex 2>/dev/null || echo /usr/local/texlive/2026/bin/universal-darwin/pdflatex)
BIBTEX=$(which bibtex 2>/dev/null || echo /usr/local/texlive/2026/bin/universal-darwin/bibtex)

$PDFLATEX -interaction=nonstopmode main.tex
$BIBTEX main
$PDFLATEX -interaction=nonstopmode main.tex
$PDFLATEX -interaction=nonstopmode main.tex

echo "Done — output: paper/main.pdf"
