# ml_basics_course
Solutions for "Basics of Machine Learning" course by mgr. Radosław Łazarz @ AGH UST

## Setup

```bash
pip install -r requirements.txt
```
## Building reports

```bash
jupyter nbconvert <notebook_name>.ipynb --to notebook --execute
jupyter nbconvert <notebook_name>.ipynb --to latex
pdflatex <notebook_name>.tex
```
