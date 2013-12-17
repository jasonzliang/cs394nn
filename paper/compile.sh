#!/bin/bash
f='selfrep'
b='selfrep'
pdflatex $f.tex \
&& bibtex $f \
&& pdflatex $f.tex \
&& pdflatex $f.tex \
&& rm $f.log $f.aux $b.bbl $b.blg
