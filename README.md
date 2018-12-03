# Installation
First make sure to install Tensorflow, namely
```
conda create -n regression python=3.5 tensorflow numpy scipy pandas scikit-bio tqdm pip
conda install -n regression biom-format -c conda-forge
source activate regression
```
Then install this package from source
```
pip install h5py git+https://github.com/mortonjt/songbird.git
```

If you are getting errors, it is likely because you have garbage channels under your .condarc.  Make sure to delete your .condarc -- you shouldn't need it.

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

# Qiime2 tutorial

First make sure that qiime2 is installed before installing songbird.  Then run

```
qiime dev refresh-cache
```

Once qiime2 is properly interfaced with songbird, you can import your biom tables
into Artifacts.  Here we will be using the [Redsea metagenome dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315489/)
as an example.  Starting from the songbird root folder, you can import this dataset as follows

```
qiime tools import \
	--input-path data/redredsea.biom \
	--output-path redsea.biom.qza \
	--type FeatureTable[Frequency]
```

You can then run the qiime2 songbird multinomial commmand as follows.

```
qiime songbird multinomial \
	--i-table redsea.biom.qza \
	--m-metadata-file data/redsea/redsea_metadata.txt \
	--p-formula "Depth+Temperature+Salinity+Oxygen+Fluorescence+Turbidity+Nitrate+Phosphate" \
	--p-min-feature-count 10 \
	--p-batch-size 3 \
	--p-epoch 10000 \
	--p-num-random-test-examples 5 \
	--o-coefficients coefficients.qza
```

The resulting coefficients learned from the regression model can be visualized as a biplot.
The command to construct the resulting ordination is as follows
```
qiime songbird regression-biplot --i-coefficients coefficients.qza --o-biplot ordination.qza
```

Once you have this, you can directly visualize this in emperor

```
qiime emperor biplot \
	--i-biplot ordination.qza \
	--m-sample-metadata-file data/redsea/feature_metadata.txt \
	--o-visualization emperor-biplot \
	--p-number-of-features 8
```

You can view the resulting visualization at https://view.qiime2.org

It should look as follows
Inline-style:
![biplot](https://github.com/mortonjt/songbird/raw/master/images/redsea-biplot.png "Regression biplot")

