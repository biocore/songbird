# Installation
First make sure to install Tensorflow, namely
```
conda create -n regression tensorflow biom-format scikit-bio tqdm -c conda-forge -c biocore
source activate regression
```
Then install this package from source
```
pip install git+https://github.com/mortonjt/regression.git
```
# Getting started

Try running the following
```
songbird multinomial \
    --formula "your_formula"
    --input-biom <your-training-biom>.biom\
    --metadata-file <your-training-metadata>.txt \
    --summary-dir <results>
```
All of the coefficients are stored under `<results>/beta.csv`.

The most important aspect of the coefficients are the rankings, or the ordering of the coefficients within a covariate.

Diagnostics can be run via Tensorboard

```
tensorboard --logdir <results>
```

This will give cross validation results and information about the loss. 

See more information about the multinomial regression through

```
songbird multinomial --help
```
