[![Build Status](https://travis-ci.org/biocore/songbird.svg?branch=master)](https://travis-ci.org/biocore/songbird)

# Getting started with Songbird
### What does Songbird produce?
The primary output from Songbird is a file containing *differentials*.
These describe the log-fold change of features with respect to certain
field(s) in your sample metadata. The most important aspect of these differentials are *rankings*, which are obtained by just sorting each column of differentials from lowest to highest. These rankings give information on the relative associations of features with a given covariate.

For more details, please see
[the paper introducing Songbird](https://www.nature.com/articles/s41467-019-10656-5).

### What do I need to run Songbird?

Songbird has a lot of parameters you can specify, but the three required
parameters are:

1. A BIOM table containing feature counts for the samples in your dataset
2. A sample metadata file containing the covariates you're interested in
   studying
3. A "formula" specifying the covariates to be included in the model Songbird
   produces, and their interactions

### How do I run Songbird?
You can run Songbird as a standalone tool from the command-line or as a
[QIIME 2](https://qiime2.org) plugin. Either works!

### The "Red Sea" dataset
This README's tutorials use a subset of the [Red Sea metagenome dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315489/) as an example dataset. This data is stored in the `data/redsea/` directory within this repository. Throughout the rest of this README, we'll just refer to this dataset as "the Red Sea dataset", "Red Sea", or something like that.

### What will this README cover?

The **"Getting started with Songbird"** section will briefly go over basic usage of
Songbird both outside and inside of QIIME 2, using the Red Sea dataset (see
above) as an example.
This section will also discuss ways of checking the fit of the model Songbird
creates, in order to validate that it looks reasonable.

The next main section of the README, **FAQs**, will cover some common questions
asked about Songbird in a variety of situations. If you have a question about
Songbird, we highly suggest reading over this!

After that, the final few sections of the README are all pretty short. These
include links to further documentation, additional packages, how to cite
Songbird, etc.

## Option 1: Using Songbird "standalone"
### Installation
First off, make sure that [conda](https://docs.conda.io/) is installed.

Songbird can be installed from the `conda-forge` channel as follows:
```
conda create -n songbird_env songbird -c conda-forge
source activate songbird_env
```

### Running Songbird
Let's try running Songbird on the "Red Sea" example data included with this
repository. You can just copy-and-paste the command below into your terminal,
assuming you've activated your `songbird_env`.
```
songbird multinomial \
    --input-biom data/redsea/redsea.biom \
    --metadata-file data/redsea/redsea_metadata.txt \
    --formula "Depth+Temperature+Salinity+Oxygen+Fluorescence+Nitrate" \
    --epochs 10000 \
    --differential-prior 0.5 \
    --summary-interval 1 \
    --summary-dir results
```
The output differentials will be stored in `results/differentials.tsv`.

### REQUIRED: Checking the fit of your model

**You need to diagnose the fit of the model Songbird creates (e.g. to
make sure it isn't "overfitting" to the data).** When running Songbird
standalone, you can do this using Tensorboard:

```
tensorboard --logdir .
```

When you open up Tensorboard in a web browser, it will show cross validation results and information about the loss. See [this section](https://github.com/biocore/songbird/#interpreting-model-fitting-information) for a description of how to interpret this information, and see [this section](https://github.com/biocore/songbird/#1-faqs-running-songbird-standalone) for details on how to use Tensorboard.

## Option 2: Using Songbird through [QIIME 2](https://qiime2.org)
### Installation
First, you'll need to make sure that QIIME 2 is installed before installing
Songbird. In your QIIME 2 conda environment, you can install Songbird by
running:

```
conda install songbird -c conda-forge
```

Next, run the following command to get QIIME 2 to "recognize" Songbird:
```
qiime dev refresh-cache
```

### Importing data into QIIME 2
Once QIIME 2 is properly interfaced with Songbird, you can import your BIOM tables
into QIIME 2 "Artifacts." Starting from the Songbird root folder, let's import
the Red Sea example data into QIIME 2 by running:

```
qiime tools import \
    --input-path data/redsea/redsea.biom \
    --output-path redsea.biom.qza \
    --type FeatureTable[Frequency]
```

### Running Songbird
After importing your feature table, you can then run Songbird through QIIME 2 as follows:

```
qiime songbird multinomial \
    --i-table redsea.biom.qza \
    --m-metadata-file data/redsea/redsea_metadata.txt \
    --p-formula "Depth+Temperature+Salinity+Oxygen+Fluorescence+Nitrate" \
    --p-epochs 10000 \
    --p-differential-prior 0.5 \
    --p-summary-interval 1 \
    --o-differentials differentials.qza \
    --o-regression-stats regression-stats.qza \
    --o-regression-biplot regression-biplot.qza
```
You can add the `--verbose` option to see a progress bar while this is running.

### REQUIRED: Checking the fit of your model

**You need to diagnose the fit of the model Songbird creates (e.g. to
make sure it isn't "overfitting" to the data).** When you run Songbird
through QIIME 2, it'll generate a `regression-stats.qza` Artifact. This can be
visualized by running:

```
qiime songbird summarize-single \
    --i-regression-stats regression-stats.qza \
    --o-visualization regression-summary.qzv
```

The resulting visualization (viewable using `qiime tools view` or at
[view.qiime2.org](https://view.qiime2.org)) contains two plots.
These plots are analogous to the two
plots shown in Tensorboard's interface (the top plot shows cross-validation
results, and the bottom plot shows loss information). The interpretation of
these plots is the same as with the Tensorboard plots: see [this section](https://github.com/biocore/songbird/#interpreting-model-fitting-information) for a description of how to interpret this information.

## Interpreting model fitting information

Regardless of whether you run Songbird standalone or through QIIME 2, **you'll need to check up on how Songbird's model has fit to your dataset.**

If you ran Songbird standalone, you can open up Tensorboard in your results directory (or the directory above your results directory) to see something like this:

![Tensorboard shown for the standalone Songbird run on the Red Sea data](https://github.com/biocore/songbird/raw/master/images/redsea-tutorial-tensorboard-output.png "Tensorboard shown for the standalone Songbird run on the Red Sea data")
_(You may need to wait for a while for Tensorboard to fully load everything.
Refreshing the page seems to help.)_

And if you ran Songbird through QIIME 2, you can open up your `regression-summary.qzv` using `qiime tools view` or using [view.qiime2.org](https://view.qiime2.org) to see something like this:

![Summary of the QIIME 2 Songbird run on the Red Sea data](https://github.com/biocore/songbird/raw/master/images/redsea-tutorial-summarize-single-output.png "Summary of the QIIME 2 Songbird run on the Red Sea data")
_(Note that this screenshot has been zoomed out a lot in order to show
the full display.)_

### Hey! I don't see anything in my plots, what's up with that?
**If you don't see anything in these plots, or if the plots only show a small handful of points, you probably need to decrease your `--summary-interval`/`--p-summary-interval` parameter.** This parameter (specified in seconds) impacts how often a "measurement" is taken for these plots. By default, it's set to 10 seconds -- but if your Songbird runs are finishing up in only a few seconds, that isn't very helpful!

Try setting the `--summary-interval`/`--p-summary-interval` to `1` to record the loss at every second. This should give you more detail in these plots. (If what you want to do is make Songbird run *longer*, then you can also do something like increasing the `--epochs`/`--p-epochs` parameter -- but that's a different story.)

Also, as mentioned above: if you're using Tensorboard, you may also need to refresh the graph a few times to get stuff to show up.

### Explaining Graph 1: `cv_error` or `Cross validation score`

This is a graph of the prediction accuracy of the model; the model will try to guess the count values for the training samples that were set aside in the script above, using only the metadata categories it has. Then it looks at the real values and sees how close it was.

The x-axis is the number of iterations (meaning times the model is training across the entire dataset). Every time you iterate across the training samples, you also run the test samples and the averaged results are being plotted on the y-axis.

**Number of iterations = (`--epochs` or `--p-epochs`) multiplied by (`--batch-size` or `--p-batch-size`)**

The y-axis is the average number of counts off for each feature. The model is predicting the sequence counts for each feature in the samples that were set aside for testing. So in the Cross Validation graphs shown above it means that, on average, the model is off by around 5 to 10 counts, which is low. However, this is ABSOLUTE error not relative error (unfortunately we can’t do relative errors because of the sparsity of metagenomic datasets)

#### How can I tell if this graph "looks good"?

The raw numbers will be variable, so it is difficult to make a blanket statement, but the most important thing is the shape of the graph. You want to see exponential decay and a stable plateau (further discussion below)

### Explaining Graph 2: `loss`

This graph is labelled "loss" because "loss" is the function being optimized. The goal here is to reduce the error of the training samples.

This graph represents how well the model fits your data.
Just like the prediction accuracy graph, the x-axis is the number of iterations (meaning times the model is training across the entire dataset).

The y-axis is MINUS log probability of the model actually fitting: so LOWER is better (maximizing the probability = minimizing the negative log probability).

#### How can I tell if *this* graph "looks good"?
Again, the numbers vary greatly by dataset. But you want to see the curve decaying, and plateau as close to zero as possible (the loss graphs shown in the Tensorboard/QIIME 2 summaries above
are nice).

## Adjusting parameters to get reasonable fitting

### An introductory note
It's worth noting that, ignoring stuff like `--output-dir`,
**the only required parameters to Songbird** are a feature table, metadata, and a formula.

In the example Songbird runs on the Red Sea dataset above, the reason we
specifically set `--epochs` and `--differential-prior` to different values was
due to consulting Tensorboard to make sure the model was properly fitting.

### Okay, so *how* should I adjust parameters to get my model to fit properly?

It's recommended to start with a small formula (with only a few variables in the model) and increase from there, because it makes debugging easier. _If your graphs are going down but not exponentially and not plateauing_, you should consider increasing the number of iterations by increasing `--epochs`/`--p-epochs`. (For more information about specifying formulas, see [this section](https://github.com/biocore/songbird/#3-faqs-formula-and-other-parameters).)

_If your graphs are going down but then going back up_, this suggests overfitting; try reducing the number of variables in your formula, or reducing `--differential-prior`/`--p-differential-prior`. As a rule of thumb, you should try to keep the number of metadata categories less than 10% the number of samples (e.g. for 100 samples, no more than 10 metadata categories).

_If you're using Songbird standalone_, Tensorboard makes it particularly easy to try out different parameters:
if you simply change a parameter and run Songbird again (under a different output file name) that graph will pop up on top of the first graphs in Tensorboard! You can click the graphs on and off in the lower left hand panel, and read just the axis for a given graph (or set of graphs) by clicking the blue expansion rectangle underneath the graph. (You'll need to run Tensorboard in the directory *directly above* your various result directories in order to get this to work.)

### TL;DR
Basically, you'll want to futz around with the parameters until you see two
nice exponential decay graphs. Once you have that, you can view the
`differentials.tsv` output to look at the differentials Songbird produced!

# FAQs

## 1. FAQs: Running Songbird standalone
**Q.** What am I looking at in the output directory?

**A.** There are 3 major types of files to note:

1. `differentials.tsv`: This contains the ranks of the microbes for certain metadata categories.  The higher the rank, the more associated it is with that category.  The lower the rank, the more negatively associated it is with a category.  The recommended way to view these files is to sort the microbes within a given column in this file and investigate the top/bottom microbes with the highest/lowest ranks. ([Qurro](https://github.com/biocore/qurro) makes this sort of analysis easier.)

   The first column in `differentials.tsv` contains the IDs of your features (if you plugged in a QIIME 2 feature table, then you can look up the sequence or bacterial name by merging with rep-seqs or taxonomy, respectively).  Once you have identified the microbes that change the most and least (have the highest and lowest differentials) you can plot the log ratio of these microbes across metadata categories or gradients!

2. `checkpoint`: this points to checkpoint files -- this can be used for saving intermediate results.  This is more important for jobs that will take days to run, where the models parameter can be investigated while the program is running, rather than waiting for `differentials.tsv` to be written.

3. `events.out.tfevents.*`: These files are what is being read into Tensorboard when it visualizes model fitting plots.

**Q.** I'm confused, what is Tensorboard?

**A.** Tensorboard is a diagnostic tool that runs in a web browser. To open tensorboard, make sure you’re in the songbird environment (`songbird_env`) and `cd` into the folder you are running the script above from. Then run:

```bash
tensorboard --logdir .

Returning line will look something like:
TensorBoard 1.9.0 at http://Lisas-MacBook-Pro-2.local:6006 (Press CTRL+C to quit)
```

Open the website (highlighted in red) in a browser. (Hint; if that doesn’t work try putting only the port number (here it is 6006), adding `localhost`, localhost:6006). Leave this tab alone. Now any songbird output directories that you add to the folder that tensorflow is running in will be added to the webpage.

This should produce a website with 2 graphs, which tensorflow actively updates as songbird is running.
![tensorboard](https://github.com/biocore/songbird/raw/master/images/tensorboard-output.png "Tensorboard")
A description of how to interpret these graphs is contained in [this section](https://github.com/biocore/songbird/#interpreting-model-fitting-information).

## 2. FAQs: Running Songbird through QIIME 2

**Q.** What are all of these QZA files, and what can I do with them?

**A.** When you run `qiime songbird multinomial`, you'll get three output QIIME 2 artifacts:

1. `differentials.qza`: This is analagous to the `differentials.tsv` file described above. This is represented as a QIIME 2 `FeatureData[Differential]` artifact, so you can directly load it into QIIME 2 plugins that accept differentials like [Qurro](https://github.com/biocore/qurro).

2. `regression-stats.qza`: This artifact contains information about how Songbird's model fitting went. You can visualize this using `qiime songbird summarize-single`, and if you have multiple Songbird runs on the same dataset you can visualize two artifacts of this type by using `qiime songbird summarize-paired`. A description of how to interpret these graphs is contained in [this section](https://github.com/biocore/songbird/#interpreting-model-fitting-information).

3. `regression-biplot.qza`: This is a biplot. It's a bit unconventionally structured, in that points in the biplot correspond to features and arrows in the biplot correspond to covariates. We'll show how to visualize this later in this FAQ section.

**Q.** What exactly does the `qiime songbird summarize-paired` command do? Why is it useful?

**A.** _(This answer uses the Red Sea dataset.)_
As we've discussed, diagnostic plots can be created from the QIIME 2 interface as follows:
```
qiime songbird summarize-single \
    --i-regression-stats regression-stats.qza \
    --o-visualization regression-summary.qzv
```

We can also generate _Q<sup>2</sup>_ values by comparing these plots to a baseline model's plots as follows:
```
qiime songbird multinomial \
    --i-table redsea.biom.qza \
    --m-metadata-file data/redsea/redsea_metadata.txt \
    --p-formula "1" \
    --p-epochs 5000 \
    --p-summary-interval 1 \
    --o-differentials baseline-diff.qza \
    --o-regression-stats baseline-stats.qza \
    --o-regression-biplot baseline-biplot.qza

qiime songbird summarize-paired \
    --i-feature-table redsea.biom.qza \
    --i-regression-stats regression-stats.qza \
    --i-baseline-stats baseline-stats.qza \
    --o-visualization paired-summary.qzv
```

The baseline model above just looks at the means (i.e. intercept), to determine how much better the first model can perform compared to the baseline model.
But one can imagine using other baseline models to contrast - for instance, fitting a model on just Temperature to gauge how informative other variables such as Salinity and Oxygen are.  The _Q<sup>2</sup>_ value is the predictive accuracy estimated from the samples left out of the regression fit.

**Q.**  _(This answer uses the Red Sea dataset.)_ What can I do with that `regression-biplot.qza` file I get from running `qiime songbird multinomial`? Can I eat it?

**A.** You can't eat it, unfortunately. But you can visualize it in [Emperor](https://github.com/biocore/emperor)!

```
qiime emperor biplot \
    --i-biplot regression-biplot.qza \
    --m-sample-metadata-file data/redsea/feature_metadata.txt \
    --p-ignore-missing-samples \
    --p-number-of-features 7 \
    --o-visualization emperor-biplot
```

You can view the resulting visualization using `qiime tools view` or at
[view.qiime2.org](https://view.qiime2.org).

These biplots have a different interpretation - the points correspond to features and the arrows correspond to covariates of interest. Running these models on the full dataset can yield something similar to as follows:
![biplot](https://github.com/biocore/songbird/raw/master/images/redsea-biplot.png "Regression biplot")

## 3. FAQs: "Formula" and other parameters

**Q.** What is a formula?  What should be passed into here?

**A.** A formula specifies the statistical model to be built based on the columns in the metadata file.
For example, if a user wanted to build a statistical model testing for differences between disease states
while controlling for gender, the formula would look something like:
```bash
--formula "diseased+gender"
```
where "diseased" and "gender" are the columns of the sample metadata file.
This is similar to the statistical formulas used in R, but the order of the variables is not important. The backend we use here is called patsy.
More details can be found here: https://patsy.readthedocs.io/en/latest/formulas.html

**Q.** That's cool!  How many variables can be passed into the formula?

**A.** That depends on the number of samples you have -- the rule of thumb is to only have about 10% of your samples.
So if you have 100 samples, you should not have a formula with more than 10 variables.  This measure needs to be used with caution, since the number of categories will also impact this.  A categorical variable with *k* categories counts as *k-1* variables, so a column with 3 categories will be represented as 2 variables in the model.  Continuous variables will only count as 1 variable.  **Beware of overfitting, though!** You can migitate the risk of overfitting with the `--differential-prior` parameter.

**Q.** Wait a minute, what do you mean that I can migitate overfitting with the `--differential-prior`?

**A.** When I mean overfitting, I'm referring to scenarios when the models attempts to memorize data points rather than
building predictive models to undercover biological patterns.  See https://xkcd.com/1725/

The `--differential-prior` command specifies the width of the prior distribution of the differentials. For `--differential-prior 1`, this means 99% of rankings (given in differentials.tsv) are within -3 and +3 (log fold change). The higher differential-prior is, the more parameters can have bigger changes, so you want to keep this relatively small.  If you see overfitting (accuracy and fit increasing over iterations in tensorboard) you may consider reducing the differential-prior in order to reduce the parameter space.

**Q.** What's up with the `--training-column` argument?

**A.** That is used for cross-validation if you have a specific reproducibility question that you are interested in answering.  If this is specified, only samples labeled "Train" under this column will be used for building the model and samples labeled "Test" will be used for cross validation.  In other words the model will attempt to predict the microbe abundances for the "Test" samples.  The resulting prediction accuracy is used to evaluate the generalizability of the model in order to determine if the model is overfitting or not.  If this argument is not specified, then 10 random samples will be chosen for the test dataset.  If you want to specify more random samples to allocate for cross-validation, the `--num-random-test-examples` argument can be specified.

**Q.** How long should I expect this program to run?

**A.** That primarily depends on a few things, namely how many samples and microbes are in your dataset, and the number of `--epoch` and the `--batch-size`.  The `--batch-size` specifies the number of samples to analyze for each iteration, and the `--epoch` specifies the number of total passes through the dataset.  For example, if you have a 100 samples in your dataset and you specify `--batch-size 5` and `--epochs 200`, then you will have `(100/5)*200=4000` iterations total.

The larger the batch size, the more samples you average per iteration, but the less iterations you have - which can sometimes buy you less time to reach convergence (so you may have to compensate by increasing the epoch).  On the other hand, if you decrease the batch size, you can have more iterations, but the variability between each iteration is higher. This also depends on if your program will converge.  This may also depend on the `--learning-rate` which specifies the resolution (smaller step size = smaller resolution, but may take longer to converge). **You will need to [consult with Tensorboard (or your regression-stats.qza output) to make sure that your model fit is sane](https://github.com/biocore/songbird/#interpreting-model-fitting-information).**  See this paper for more details on gradient descent: https://arxiv.org/abs/1609.04747

## 4. FAQs: Output files

**Q.** Why do I have so many columns in my `differentials` even when I'm only using one continuous variable?

**A.** If Songbird sees that a sample metadata column contains **any** non-numeric values -- even things like `Not Applicable`, `NA`, `nan`, etc. -- then Songbird will assume that that sample metadata column is categorical (and not quantitative). This is likely why you're seeing multiple columns for a single sample metadata column.

**If you're having this problem and you're running Songbird standalone, you will need to delete any `#q2:` headers at the start of the sample metadata file -- otherwise, Songbird will interpret these lines as describing actual samples.**


# Further Documentation

For a more complete tutorial, see the following url that includes real datasets:

https://github.com/knightlab-analyses/reference-frames


# Acknowledgements

Credits to Lisa Marotz ([@lisa55asil](https://github.com/lisa55asil)) for the initial FAQs and a ton of the documentation in this README.

# Related packages

For interactively visualizing the differentials produced by Songbird, check out [Qurro](https://github.com/biocore/qurro).

# Citations

If you use this tool and you like it, feel free to cite at

```
@article{morton2019establishing,
  title={Establishing microbial composition measurement standards with reference frames},
  author={Morton, James T and Marotz, Clarisse and Washburne, Alex and Silverman, Justin and Zaramela, Livia S and Edlund, Anna and Zengler, Karsten and Knight, Rob},
  journal={Nature communications},
  volume={10},
  number={1},
  pages={2719},
  year={2019},
  publisher={Nature Publishing Group}
}
```
