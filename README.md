# ml_basics_course
Solutions for "Basics of Machine Learning" course by mgr. Radosław Łazarz @ AGH UST

## Setup
The notebooks have been built and tested with Python v3.6.6.

Obviously, `jupyter notebook` is required to run the notebooks.

Because of Polish signs, notebooks are converted to `.tex` files and only then to `.pdf`. Therefore, `pdflatex` must also be installed. 

### Installing python requirements

#### With Makefile
```bash
make install
```
#### With pip 

```bash
pip install -r requirements.txt
```


## Building reports

### With Makefile
To build `.pdf`s of all notebooks:
```bash
make 
```

To build `.pdf`s of only selected notebooks:

```bash
make KEY=<an element of the notebooks' path to be matched>
```

So, for example

```bash
make KEY=lab1 
```

will build `pdf`s of only the notebooks which contain `lab1` in their path.

### Using regular tools

```bash
jupyter nbconvert <notebook_path>.ipynb --execute --to latex 
pdflatex <notebook_path>.tex
```
