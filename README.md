[![Build Status](https://travis-ci.org/biocore/songbird.svg?branch=master)](https://travis-ci.org/biocore/songbird)

# Songbird
### What does Songbird produce?
The primary output from Songbird is a file containing *differentials*.
These describe the log-fold change of features (microbes, metabolites, ...)
with respect to certain
field(s) in your sample metadata. The most important aspect of these differentials are *rankings*, which are obtained by sorting a column of differentials from lowest to highest. These rankings give information on the relative associations of features with a given covariate.

For more details, please see
[the paper introducing Songbird](https://www.nature.com/articles/s41467-019-10656-5).

### What do I need to run Songbird?

Songbird has a lot of parameters you can specify, but the three required
parameters are:

1. A [BIOM table](http://biom-format.org/index.html) containing feature
   counts for the samples in your dataset
2. A sample metadata file containing the covariates you're interested in
   studying (this should be a tab-separated file)
3. A "formula" specifying the covariates to be included in the model Songbird
   produces, and their interactions (see
   <a href="#specifying-a-formula">this section</a> for details)

### How do I run Songbird?
You can run Songbird as a standalone tool from the command-line or as a
[QIIME 2](https://qiime2.org) plugin. Either works!

### The "Red Sea" dataset
This README's tutorials use a subset of the [Red Sea metagenome dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315489/) as an example dataset. This data is stored in the `data/redsea/` directory within this repository. Throughout the rest of this README, we'll just refer to this dataset as "the Red Sea dataset", "Red Sea", or something like that.

Features in this dataset are KEGG orthologs (KOs) -- see the dataset's paper,
in particular
[this section](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315489/#__sec7title),
for more details.

### What will this README cover?

This README is divided into a few main sections:

1. **Using Songbird "standalone"**
2. **Using Songbird through QIIME 2**
3. **Specifying a formula**
4. **Interpreting model fitting information**
5. **Adjusting parameters to get reasonable fitting**
6. **Validating by comparing to null/baseline models**
7. **FAQs**
8. **Visualizing Songbird's differentials**
9. **More information**

# 1. Using Songbird "standalone"
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
	--training-column Testing \
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

When you open up Tensorboard in a web browser, it will show plots of the cross validation score and loss. See <a href="#interpreting-model-fitting">this section on interpreting model fitting</a> for details on how to understand these plots, and see <a href="#faqs-standalone">the section of the FAQs on running Songbird standalone</a> for details on how to use Tensorboard.

**Once you have a good fit for your model, you need to run a null model for comparison to make
sure that your model is more predictive than the null model.**
See <a href="#validating-null-model">this section on how to run and compare a null model</a>.

### A note about metadata and running Songbird standalone
If you have any "comment rows" in your metadata -- for example, a `#q2:types`
row -- you will need to delete these before running Songbird standalone, since
otherwise standalone Songbird will interpret these rows as actual samples.
([We're planning on addressing this in the future.](https://github.com/biocore/songbird/issues/78))

# 2. Using Songbird through [QIIME 2](https://qiime2.org)
### Installation
First, you'll need to make sure that QIIME 2 (**version 2019.7 or later**) is installed
before installing Songbird. In your QIIME 2 conda environment, you can install Songbird by
running:

```
conda install songbird -c conda-forge
```

Next, run the following command to get QIIME 2 to "recognize" Songbird:
```
qiime dev refresh-cache
```

### Importing data into QIIME 2
If your data is not already in QIIME2 format, you can import your BIOM tables
into QIIME 2 "Artifacts". Starting from the Songbird root folder, let's import
the Red Sea example data into QIIME 2 by running:

```
qiime tools import \
	--input-path data/redsea/redsea.biom \
	--output-path redsea.biom.qza \
	--type FeatureTable[Frequency]
```

### Running Songbird
With your QIIME 2-formatted feature table, you can run Songbird through QIIME 2 as follows:

```
qiime songbird multinomial \
	--i-table redsea.biom.qza \
	--m-metadata-file data/redsea/redsea_metadata.txt \
	--p-formula "Depth+Temperature+Salinity+Oxygen+Fluorescence+Nitrate" \
	--p-epochs 10000 \
	--p-differential-prior 0.5 \
	--p-training-column Testing \
	--p-summary-interval 1 \
	--o-differentials differentials.qza \
	--o-regression-stats regression-stats.qza \
	--o-regression-biplot regression-biplot.qza
```
You can add the `--verbose` option to see a progress bar while this is running.

### REQUIRED: Checking the fit of your model

**You need to diagnose the fit of the model Songbird creates (e.g. to
make sure it isn't "overfitting" to the data).** When you run Songbird
through QIIME 2, it will generate a `regression-stats.qza` Artifact. This can be
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
these plots is the same as with the Tensorboard plots: see <a href="#interpreting-model-fitting">this section on interpreting model fitting</a> for details on how to understand these plots.

**Once you have a good fit for your model, you need to run a null model for comparison to make
sure that your model is more predictive than the null model.**
See <a href="#validating-null-model">this section on how to run and compare a null model</a>.
# 3. Specifying a formula <span id="specifying-a-formula"></span>

### Hang on, what *is* a formula?
A **formula** specifies the statistical model to be built based on the columns in the metadata file.
For example, if a user wanted to build a statistical model testing for differences between disease states
while controlling for sex, the formula would look something like:
```bash
--formula "diseased+sex"
```
where `diseased` and `sex` are columns in the sample metadata file.
This is similar to the statistical formulas used in R, but the order in which you
specify the variables (a.k.a. column names) is not important. The backend we use here for processing formulas is
called [patsy](https://patsy.readthedocs.io/).

The metadata columns used in the `--formula` can be either numeric or categorical.  As you can imagine, there are many possible ways to specify metadata columns in
your formula -- in particular, encoding categorical variables can get tricky.
(For example, should a
[nominal](https://en.wikipedia.org/wiki/Level_of_measurement#Nominal_level)
variable like `sex` be encoded the
same way as an
[ordinal](https://en.wikipedia.org/wiki/Level_of_measurement#Ordinal_scale)
variable like `disease_progression`? Probably not!)
In this section we'll go over a few common cases, and link to some more general
documentation at the end in case these examples aren't sufficient.

### The implicit "reference": how categorical variables are handled <span id="implicit-reference"></span>
Let's say your formula just includes one categorical variable:
```bash
--formula "healthy_or_sick"
```
...where the only possible values of `healthy_or_sick` are `healthy` and
`sick`. The output differentials produced by Songbird will only have two
columns:
1. `Intercept`
2. `healthy_or_sick[T.healthy]` OR `healthy_or_sick[T.sick]`.

The second differential column indicates association with the given value:
`healthy_or_sick[T.healthy]` differentials indicate association with
`healthy`-valued samples *using `sick`-valued samples as a reference*,
and `healthy_or_sick[T.sick]` differentials indicate association with
`sick`-valued samples *using `healthy`-valued samples as a reference*. You'll
only get one of these columns; the choice of reference value, if left
unspecified, is arbitrary.

The reference value is used as the denominator in the log-fold change
computation of differentials. So, for `healthy_or_sick[T.healthy]` -- a
differential column where `sick`-valued samples are implicitly set as the
reference -- the features with the most negative differential ranking values
will be more associated with `sick`-valued samples, whereas the features with
the most positive differential ranking values will be more associated with
`healthy`-valued samples.

**It is possible to explicitly set the reference value** in your formula.
Going back to the `healthy_or_sick` example, if you know you want to use
`healthy`-valued samples as a reference, you can describe this in
the formula like so:
```bash
--formula "C(healthy_or_sick, Treatment('healthy'))"
```
This will ensure that your second column of differentials is
`C(healthy_or_sick, Treatment('healthy'))[T.sick]` -- that is, the association
with `sick`-valued samples using `healthy`-valued samples as a reference.

If you have more than two categories in a given column (e.g. `healthy`, `disease A`, or `disease B`) 
then Songbird will return `K-1` columns of differentials (where `K` is the number of categories in a given column)
with each category compared to the reference.
In this example it would return two columns; one with `disease A` compared to `healthy` and one with `disease B` 
compared to `healthy`.

### Do you have some more simple examples of using formulas?

Sure! In general, it's a good idea to read over the [patsy documentation](https://patsy.readthedocs.io/en/latest/formulas.html) --
there are a lot of ways you can specify formulas. However, here are a few small
examples of common formula uses.

#### Example 1: I have a categorical metadata field, and I want to explicitly set the reference

This was described <a href="#implicit-reference">above</a>.
Basically, you'll want to specify a formula of something like

```bash
--formula "C(your_metadata_field, Treatment('desired_reference_value'))"
```

See [this blog post](http://mortonjt.blogspot.com/2018/05/encoding-design-matrices-in-patsy.html) for more details.

#### Example 2: I have a categorical metadata field with *ordered* values, and I want to account for that

See [this blog post](https://mortonjt.github.io/probable-bug-bytes/probable-bug-bytes/ordinal-variables/) for details on how to do this.

Basically, you'll want to specify a formula of something like
```bash
--formula "C(your_metadata_field, Diff, levels=['first_level', 'second_level', 'third_level'])"
```

### How many variables can be passed into a formula?
That depends on the number of samples you have -- the rule of thumb is to only have about
10% of your samples.
So if you have 100 samples, you should not have a formula with more than 10 variables.  This measure needs to be used with caution, since the number of categories will also impact this.  A categorical variable with *k* categories counts as *k-1* variables, so a column with 3 categories will be represented as 2 variables in the model.  Continuous (or numerical) variables will only count as 1 variable.  **Beware of overfitting, though!** You can mitigate the risk of overfitting by adjusting the `--differential-prior` parameter.
For more information on `--differential-prior` and other Songbird parameters, please see
<a href="#faqs-parameters">this section of the FAQs on parameters</a>.

### Further documentation on patsy formulas
In case the above examples didn't cut it, here are some links to external documentation that may
be helpful, sorted roughly into "less" and "more" technical categories:

##### Less technical

- [Video explaining various types of categorical encodings](https://www.youtube.com/watch?v=WRxHfnl-Pcs)
- [Blog post explaining coding nominal variables](http://mortonjt.blogspot.com/2018/05/encoding-design-matrices-in-patsy.html)
- [Blog post explaining coding ordinal variables](https://mortonjt.github.io/probable-bug-bytes/probable-bug-bytes/ordinal-variables/)

##### More technical

- [patsy formula documentation](https://patsy.readthedocs.io/en/latest/formulas.html)
- [patsy API reference](https://patsy.readthedocs.io/en/latest/API-reference.html)

If you have any other resources you'd like to see added here, feel free to send a PR/issue.

### Hey, isn't it "formulae" instead of "formulas"?

That's debatable!

# 4. Interpreting model fitting information <span id="interpreting-model-fitting"></span>

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

Try setting the `--summary-interval`/`--p-summary-interval` to `1` to record the loss at every second. This should give you more detail in these plots. (If what you want to do is make Songbird run *longer*, i.e. over more iterations, then you can increase the `--epochs`/`--p-epochs` parameter -- more on this later.)

Also, as mentioned above: if you're using Tensorboard, you may also need to refresh the graph a few times to get stuff to show up.

## 4.1. Explaining Graph 1: `cv_error` or `Cross validation score`

This is a graph of the prediction accuracy of the model; the model will try to guess the count values for the training samples that were set aside in the script above, using only the metadata categories it has. Then it looks at the real values and sees how close it was.

The x-axis is the number of iterations (meaning times the model is training across the entire dataset). Every time you iterate across the training samples, you also run the test samples and the averaged results are being plotted on the y-axis.

The number of iterations is influenced by a number of parameters, most notably
by the `--epochs`/`--p-epochs` parameter. All else being equal, the greater this number is the more iterations the model will go through.

The y-axis is the average number of counts off for each feature. The model is predicting the sequence counts for each feature in the samples that were set aside for testing. So in the Cross Validation graphs shown above it means that, on average, the model is off by around 5 to 10 counts, which is low. However, this is ABSOLUTE error -- not relative error (unfortunately we can’t do relative errors because of the sparsity of metagenomic datasets).

#### How can I tell if this graph "looks good"?

The raw numbers will be variable, but the most important thing is the shape of the graph. You want to see exponential decay and a stable plateau. The cross validation graphs shown in the Tensorboard/QIIME 2 summaries above are good examples.
**Importantly, you need to compare your model to a null model** to ensure that the covariates entered into the formula are _improving_ the model fit. More details on this can be found in <a href="#validating-null-model">section 6: Validating by comparing to null/baseline models</a>.
	

## 4.2. Explaining Graph 2: `loss`

This graph is labelled "loss" because "loss" is the function being optimized. The goal here is to reduce the error of the training samples.

This graph represents how well the model fits your data.
Just like the prediction accuracy graph, the x-axis is the number of iterations (meaning times the model is training across the entire dataset).

The y-axis is MINUS log probability of the model actually fitting: so LOWER is better (maximizing the probability = minimizing the negative log probability).

#### How can I tell if *this* graph "looks good"?
Again, the numbers vary greatly by dataset, but you want to see the curve decaying and plateau as close to zero as possible (the loss graphs shown in the Tensorboard/QIIME 2 summaries above
are nice).


# 5. Adjusting parameters to get reasonable fitting <span id="adjusting-parameters"></span>

### An introductory note
It's worth noting that, ignoring stuff like `--output-dir`,
**the only required parameters to Songbird** are a feature table, metadata, and a formula.

In the examples using the Red Sea dataset above, the reason we
changed the `--epochs` and `--differential-prior` to different values from the default was
due to consulting Tensorboard to optimize parameter fit.

### Okay, so *how* should I adjust parameters to get my model to fit properly?

It's recommended to start with a small formula (with only a few variables in the model) and increase from there, because it makes debugging easier. (See <a href="#specifying-a-formula">this section on specifying formulas</a> for more information.) **If your graphs are going down but not exponentially and not plateauing**, you should consider increasing the number of iterations by, for example, increasing `--epochs`/`--p-epochs`.

**If your graphs are going down but then going back up**, this suggests overfitting; try reducing the number of variables in your formula, or reducing `--differential-prior`/`--p-differential-prior`. As a rule of thumb, you should try to keep the number of metadata categories less than 10% the number of samples (e.g. for 100 samples, no more than 10 metadata categories).

**If you have a lot of samples**, you may want to try increasing the
`--num-random-test-examples`/`--p-num-random-test-examples` and/or
`--batch-size`/`--p-batch-size` parameters. This parameter specifices how many samples will be held out of model Training and used for model Testing. As a rule of thumb, this should be ~10-20% of your samples.

**If your sample categories are not evenly distributed** (i.e. you have 100 healthy samples but only 20 sick samples), you can specify which samples will be used for training versus testing. By default, the train/test samples are randomly assigned. You can add a column to your metadata file that marks each sample as either `Train` or `Test` (these values are case sensitive!) and specifying this column in `--p-training-column`. Make sure that a relatively even number of each sample category are in the Train and Test groups (i.e. so you are not training only on healthy samples and testing on sick). More information <a href="#training-column">on specifying the training columns can be found here</a>.

### Is there anything else I can do?

There are other parameters besides the ones mentioned above that you can
tweak in Songbird in order to adjust your model's fitting. Depending on if
you're using Songbird standalone or through QIIME 2, you can run
`songbird multinomial --help` or `qiime songbird multinomial --help` to get
a list of all available parameters, respectively.


# 6. Validating by comparing to null/baseline models <span id="validating-null-model"></span>

## 6.1. Generating a null model
Now that we have generated a model and understand how to interpret the diagnostic plots, it is important to compare this model to a model made without any metadata input (a "null model"). In other words, this will allow us to see how strongly the covariates in the formula can be associated with the features of the model, as compared to random chance. We can do this by simply supplying `--formula`/`--p-formula` with `"1"` as shown below.

Please note that the standalone and QIIME 2 versions of Songbird will get you
different things. Both show you a plot with multiple model fitting curves, but:

  1. The standalone version of Songbird has support through Tensorboard for
     comparing model fitting (see the screenshot below).

  2. The QIIME 2 version of Songbird uses a much simpler interface.
     It also includes a <a href="#explaining-q2">_Q<sup>2</sup>_ value</a>.

We recommend trying this out with the QIIME 2 version at first.

### 6.1.1. Null models and "standalone" Songbird

If you're using Songbird standalone, Tensorboard makes it particularly easy to compare different parameters:
if you simply change a parameter and run Songbird again (under a different output file name) that graph will pop up on top of the first graphs in Tensorboard!
Let's generate a null model as follows:

```
songbird multinomial \
	--input-biom data/redsea/redsea.biom \
	--metadata-file data/redsea/redsea_metadata.txt \
	--formula "1" \
	--epochs 10000 \
	--differential-prior 0.5 \
	--training-column Testing \
	--summary-interval 1 \
	--summary-dir results
```

Notice how this is the same command that we used before (e.g. we have the same
`--differential-prior` and `--epochs`), with the exception that the `--formula`
is now just `"1"`.

Let's open up Tensorboard. You can click the graphs on and off in the lower left hand panel, and read just the axis for a given graph (or set of graphs) by clicking the blue expansion rectangle underneath the graph. Within Tensorboard, you can also view a summary of final results under the `HPARAMS` tab -- that will look something like this:

![HParams for the Red Sea Data](https://github.com/biocore/songbird/raw/master/images/redsea-tutorial-tensorboard-hparams.png)

Above, the model with the lowest cross-validation error is highlighted in green. You can see how its performance
compares to other models that were fit on this dataset with various parameters. Additionally, the model's diagnostic
plots are shown, which will have an exponential decay and a stable plateau if the model converged.

### 6.1.2. Null models and QIIME 2 + Songbird
If you're running Songbird through QIIME 2, the
`qiime songbird summarize-paired` command allows you to view two sets of
diagnostic plots at once as follows:

```bash
# Generate a null model
qiime songbird multinomial \
	--i-table redsea.biom.qza \
	--m-metadata-file data/redsea/redsea_metadata.txt \
	--p-formula "1" \
	--p-epochs 10000 \
	--p-differential-prior 0.5 \
	--p-training-column Testing \
	--p-summary-interval 1 \
	--o-differentials null-diff.qza \
	--o-regression-stats null-stats.qza \
	--o-regression-biplot null-biplot.qza

# Visualize the first model's regression stats *and* the null model's
# regression stats
qiime songbird summarize-paired \
	--i-regression-stats regression-stats.qza \
	--i-baseline-stats null-stats.qza \
	--o-visualization paired-summary.qzv
```

The summary generated will look something like as follows:

![Summary of the QIIME 2 Songbird run on the Red Sea data](https://github.com/biocore/songbird/raw/master/images/redsea-tutorial-summarize-paired-output.png "Summary of the QIIME 2 Songbird run on the Red Sea data")

This visualization isn't quite as fancy as the Tensorboard one we showed above,
but it does include a _Q<sup>2</sup>_ value.

#### 6.1.2.1. Sidenote: Interpreting _Q<sup>2</sup>_ values <span id="explaining-q2"></span>

The _Q<sup>2</sup>_ score is adapted from the Partial least squares literature.  Here it is given by `Q^2 = 1 - m1/m2`, where `m1` indicates the average absolute model error and `m2` indicates the average absolute null or baseline model error.  If _Q<sup>2</sup>_ is close to 1, that indicates a high predictive accuracy on the cross validation samples. If _Q<sup>2</sup>_ is low or below zero, that indicates poor predictive accuracy, suggesting possible overfitting. This statistic behaves similarly to the _R<sup>2</sup>_ classically used in a ordinary linear regression if `--p-formula` is `"1"` in the `m2` model.

If the _Q<sup>2</sup>_ score is extremely close to 0 (or negative), this indicates that the model is overfit or that the metadata supplied to the model are not predictive of microbial composition across samples. You can think about this in terms of "how does using the metadata columns in my formula *improve* a model?" If there isn't really an improvement, then you may want to reconsider your formula.

... [But as long as your _Q<sup>2</sup>_ score is above zero, your model is learning something useful](https://forum.qiime2.org/t/songbird-optimizing-the-loss-function/13479/8).

## 6.2. Generating baseline models
_(For the sake of simplicity, this subsection just uses Songbird through QIIME
2.)_

In addition to comparing your model to a null model, it may also be useful to compare it to 'baseline' models, which contain only a subset of the formula variables. The _Q<sup>2</sup>_ score will allow you to determine how much a given variable(s) improves the model, and therefore how important it is for determining differntially abundant microbes. 

For instance, in the Red Sea dataset we can fit a model on just Depth and compare it to our original model to gauge how informative the other variables are (i.e. Temperature, Salinity, Oxygen, Fluorescence, and Nitrate).  The _Q<sup>2</sup>_ value we see in the `paired-summary.qzv` is the predictive accuracy estimated from the variables left out of the regression fit.  

```bash
# Generate a baseline model
qiime songbird multinomial \
	--i-table redsea.biom.qza \
	--m-metadata-file data/redsea/redsea_metadata.txt \
	--p-formula "Depth" \
	--p-epochs 10000 \
	--p-differential-prior 0.5 \
	--p-training-column Testing \
	--p-summary-interval 1 \
	--o-differentials baseline-diff.qza \
	--o-regression-stats baseline-stats.qza \
	--o-regression-biplot baseline-biplot.qza

# Visualize the first model's regression stats *and* the baseline model's
# regression stats
qiime songbird summarize-paired \
	--i-regression-stats regression-stats.qza \
	--i-baseline-stats baseline-stats.qza \
	--o-visualization paired-summary-2.qzv
```
This plot provides evidence for how much the model fit improved by adding `Temperature`, `Salinity`, `Oxygen`, `Fluorescence`, and `Nitrate`. A positive _Q<sup>2</sup>_ score indicates an improvement over the baseline model.

### TL;DR
Basically, you'll want to futz around with the parameters until you see two
nice exponential decay graphs, where your model provides an improvement in the cross-validation graph
compared to the null model.
Once you have that, you can start to look at the `differentials` Songbird produced!


# 7. FAQs

## 7.1. FAQs: Running Songbird standalone <span id="faqs-standalone"></a>
**Q.** What am I looking at in the output directory?

**A.** There are 3 major types of files to note:

1. `differentials.tsv`: This contains the ranks of the features for certain metadata categories.  The higher the rank, the more associated it is with that category.  The lower the rank, the more negatively associated it is with a category.  The recommended way to view these files is to sort the features within a given column in this file and investigate the top/bottom features with the highest/lowest ranks. ([Qurro](https://github.com/biocore/qurro) makes this sort of analysis easier.)

   The first column in `differentials.tsv` contains the IDs of your features (if you plugged in a QIIME 2 feature table and your "features" are ASVs/sOTUs/..., then you can look up the sequence or bacterial name by merging with rep-seqs or taxonomy, respectively).  Once you have identified the features that change the most and least (have the highest and lowest differentials) you can plot the log ratio of these features across metadata categories or gradients!

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
See <a href="#interpreting-model-fitting">this section on interpreting model fitting</a> for details on how to understand these plots.

## 7.2. FAQs: Running Songbird through QIIME 2

**Q.** What are all of these QZA files, and what can I do with them?

**A.** When you run `qiime songbird multinomial`, you'll get three output QIIME 2 artifacts:

1. `differentials.qza`: This is analagous to the `differentials.tsv` file described above. This is represented as a QIIME 2 `FeatureData[Differential]` artifact, so you can directly load it into QIIME 2 plugins that accept differentials like [Qurro](https://github.com/biocore/qurro).

2. `regression-stats.qza`: This artifact contains information about how Songbird's model fitting went. You can visualize this using `qiime songbird summarize-single`, and if you have multiple Songbird runs on the same dataset you can visualize two artifacts of this type by using `qiime songbird summarize-paired`. See <a href="#interpreting-model-fitting">this section on interpreting model fitting</a> for details on how to understand the resulting visualization.

3. `regression-biplot.qza`: This is a biplot. It's a bit unconventionally structured in that points in the biplot correspond to features and arrows in the biplot correspond to covariates. We'll show how to visualize this later in this FAQ section.

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

## 7.3. FAQs: Parameters <span id="faqs-parameters"></span>

**Q.** The "specifying a formula" section mentioned that I can mitigate overfitting with `--differential-prior`. What does that mean?

**A.** By overfitting, we're referring to scenarios when the model attempts to memorize data points rather than
building predictive models to undercover biological patterns.  See https://xkcd.com/1725/

The `--differential-prior` command specifies the width of the prior distribution of the differentials. For `--differential-prior 1`, this means 99% of rankings (given in differentials.tsv) are within -3 and +3 (log fold change). The higher differential-prior is, the more parameters can have bigger changes, so you want to keep this relatively small.  If you see overfitting (accuracy and fit increasing over iterations in tensorboard) you may consider reducing the differential-prior in order to reduce the parameter space.

**Q.** What's up with the `--training-column` argument? <span id="training-column"></span>

**A.** That is used for cross-validation if you have a specific reproducibility question that you are interested in answering.  If this is specified, only samples labeled "Train" under this column will be used for building the model and samples labeled "Test" will be used for cross validation.  In other words the model will attempt to predict the feature abundances for the "Test" samples. Ideally the random samples held out for testing will be a relatively even mix of samples from different categories. This parameter can be especially useful if your samples are not evenly distributed across sample categories, so that you can ensure even distribution of sample categories across the "Train" and "Test" groups.

The resulting prediction accuracy is used to evaluate the generalizability of the model in order to determine if the model is overfitting or not.  If this argument is not specified, then 5 random samples will be chosen for the test dataset.  If you want to specify more random samples to allocate for cross-validation, the `--num-random-test-examples` argument can be specified. As a rule of thumb, the number of test samples held out for cross-validation is typically 10-20%.

**Note that the `Train` and `Test` values are case sensitive** -- if you
specify `train` and `test` instead, [things may go wrong](https://github.com/biocore/songbird/issues/102).

**Q.** How long should I expect this program to run?

**A.** There are many factors that influence this. The numbers of samples and
features in your dataset play a large role, as does the
`--epochs`/`--p-epochs` parameter.

The larger the batch size, the more samples you average per iteration, but the less iterations you have - which can sometimes buy you less time to reach convergence (so you may have to compensate by increasing the epochs).  On the other hand, if you decrease the batch size, you can have more iterations, but the variability between each iteration is higher. This also depends on if your program will converge.  This may also depend on the `--learning-rate` which specifies the resolution (smaller step size = smaller resolution, but may take longer to converge). **You will need to <a href="#interpreting-model-fitting">consult with Tensorboard (or your regression-stats.qza output) to make sure that your model fit is sane</a>.**  See this paper for more details on gradient descent: https://arxiv.org/abs/1609.04747

**Q.** How exactly does filtering work?!?

**A.** There are two filtering parameters `--min-feature-count` and `--min-sample-count`.  Be __careful__, these two parameters have different behaviors.
`--min-sample-count` will filter out samples according to the total read count.  For instance, if `--min-sample-count 1000` is specified, then
all samples that have less than 1000 reads will be filtered out.

`--min-feature-count` will filter out features according to how many __samples__ they appear in.  For instance, if `--min-feature-count 10` is specified,
then all features than appear in less than 10 samples will be thrown out.  It is important to note that we are filtering according to number of samples rather than number of reads.  The reason why this behavior is chosen relates to a rule of thumb commonly used in linear regression - if a microbe appears in less than 10 samples, it is difficult to fit a meaningful line for that microbe.  In other words, there is not even resolution in the study to say anything meaningful about that microbe in the context of differential abundance analysis.  The `--min-feature-count` filter is applied _after_ the `--min-sample-count` is applied.

## 7.4. FAQs: Output files

**Q.** Why do I have so many columns in my `differentials` even when I'm only using one continuous variable?

**A.** If Songbird sees that a sample metadata column contains **any** non-numeric values -- even things like `Not Applicable`, `NA`, `nan`, etc. -- then Songbird will assume that that sample metadata column is categorical (and not quantitative). This is likely why you're seeing multiple columns for a single sample metadata column.

**If you're having this problem and you're running Songbird standalone, you will need to delete any `#q2:` headers at the start of the sample metadata file -- otherwise, Songbird will interpret these lines as describing actual samples.**

## 7.5. FAQs: Installation

**Q.** I've installed Songbird, but when I try to run it I get an error that
says something like ``ImportError: /lib64/libc.so.6: version `GLIBC_2.15' not found``. What's going on here?

**A.** In our experience, this is a problem with installing TensorFlow on
conda. You should be able to get around this problem by uninstalling
TensorFlow, then re-installing it from the `defaults` conda channel:

```bash
conda uninstall tensorflow
conda install -c defaults tensorflow
```

# 8. Visualizing Songbird's differentials

[Qurro](https://github.com/biocore/qurro) generates interactive visualizations of the differentials produced by Songbird! A Qurro demo using the Red Sea dataset is available [here](https://biocore.github.io/qurro/demos/red_sea/index.html).

You can also view the output differentials as a nice, sortable table using
`qiime metadata tabulate`:

```bash
qiime metadata tabulate \
	--m-input-file differentials.qza \
	--o-visualization differentials-viz.qzv
```

(The output `.qzv` file can be viewed using `qiime tools view` or at
[view.qiime2.org](https://view.qiime2.org).)

![Tabulated differentials](https://github.com/biocore/songbird/raw/master/images/redsea-differentials-tabulated.png "Tabulated differentials, viewed at view.qiime2.org.")

# 9. More information

## 9.1. Further Documentation

For a more complete tutorial, see the following url that includes real datasets:

https://github.com/knightlab-analyses/reference-frames

## 9.2. Acknowledgements

Credits to Lisa Marotz ([@lisa55asil](https://github.com/lisa55asil)) and Marcus Fedarko ([@fedarko](https://github.com/fedarko)) for the FAQs and a ton of the documentation in this README.

## 9.3. Citations

If you use this tool and you like it, please cite it at

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
