EXEC=jupyter nbconvert --to notebook --execute
LATEX=jupyter nbconvert --execute  --to latex 
PDF=pdflatex


clean:
	git clean -xdf

%.pdf: %.tex
	cd `dirname $^` && $(PDF) `basename $^`

%.tex: %.ipynb
	$(LATEX) $^ --output=`basename $@` 	

