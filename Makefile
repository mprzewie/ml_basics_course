EXEC=jupyter nbconvert --to notebook --execute
LATEX=jupyter nbconvert --to latex 
PDF=pdflatex


clean:
	git clean -xdf

%.pdf: %.nbconvert.tex
	cd `dirname $^` && $(PDF) `basename $^`

%.tex: %.nbconvert.ipynb
	$(LATEX) $^ --output=$@ 	

%.nbconvert.ipynb: %.ipynb 
	$(EXEC) $^ 




