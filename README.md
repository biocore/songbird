[![Build Status](https://travis-ci.org/biocore/songbird.svg?branch=master)](https://travis-ci.org/biocore/songbird)

# Installation
Songbird can be installed on conda-forge as follows
```
conda create -n songbird_env songbird -c conda-forge
source activate songbird_env
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
All of the coefficients are stored under `<results>/differentials.tsv`.

The most important aspect of the coefficients are the rankings, or the ordering of the coefficients within a covariate.

Diagnostics can be run via Tensorboard

```
tensorboard --logdir <your-result-directory>
```

This will give cross validation results and information about the loss.

See more information about the multinomial regression through

```
songbird multinomial --help
```

For a more complete tutorial see the following url that includes real datasets

https://github.com/knightlab-analyses/reference-frames

# FAQs
**Q** What am I looking at in the output directory?

**A** There are 3 major types of files to note

`differentials.tsv`: This contains the ranks of the microbes for a given metadata categories.  The higher the rank, the more associated it is with that category.  The lower the rank, the more negatively associated it is with a category.  The recommended way to view these files is to sort the microbes within a given column in this file and investigate the top/bottom microbes with the highest/lowest ranks.

The first column is the features (if you plugged in a q2 table, then you can look up the sequence or bacterial name by merging with rep-seqs or taxonomy, respectively).  Once you have identified the microbes that change the most and least (have the highest and lowest coefficients) you can plot the log ratio of these microbes across metadata categories or gradients!

note: continuous variables should only produce ONE column in the differentials.tsv file, if not, something is wrong with the metadata (maybe not all numbers or something)

`checkpoint` : this points to checkpoint files -- this can be used for saving intermediate results.  This is more important for jobs that will take days to run, where the models parameter can be investigated while the program is running, rather than waiting for `differentials.tsv` to be written.

`events.out.tfevents.*` : These files are what is being read into Tensorboard - more discussion on this later.

**Q**. Why do I have so many columns in my differentials.tsv even when I'm only using one continuous variable?

**A**. A couple things could be happening.  First, the standalone songbird script assumes that the mapping files only have 1 line for the header, so you need to reformat.  In addition, it could be that there are other values in that column (i.e. Not Applicable, NA, nan, ...), and these sorts of values need to be removed in order to properly perform songbird regression on continuous variables. For continuous variables, only numeric characters are accepted otherwise songbird assumes this is a categorical value.

**Q**. What is a formula?  What should be passed into here?

**A**. A formula specifies the statistical model to be built based on the columns in the metadata file.
For example, if a user wanted to build a statistical model testing for differences between disease states
while controlling for gender, the formula would look something as follows

--formula "diseased+gender"

where "diseased" and "gender" are the columns of the sample metadata file.
This is similar to the statistical formulas used in R, but the order of the variables is not important. The backend we use here is called patsy.
More details can be found here: https://patsy.readthedocs.io/en/latest/formulas.html

**Q**. That's cool!  How many variables can be pass into the formula?

**A**. That depends on the number of samples you have -- the rule of thumb is to only have about 10% of your samples.
So if you have 100 samples, you should not have a formula with more than 10 variables.  This measure needs to be used with caution, since the number of categories will also impact this.  A categorical variable with *k* categories counts as *k-1* variables, so a column with 3 categories will be represented as 2 variables in the model.  Continuous variables will only count as 1 variable.  You can sometime migitate this risk with the `--differential-prior` parameter.

**Q**. Wait a minute, what do you mean that I can migitate overfitting with the `--differential-prior`?

**A**. When I mean overfitting, I'm referring to scenarios when the models attempts to memorize data points rather than
building predictive models to undercover biological patterns.  See https://xkcd.com/1725/

The `--differential-prior` command specifies the width of the prior distribution of the coefficients. For `--differential-prior 1`, this means 99% of rankings (given in differentials.tsv) are within -3 and +3 (log fold change). The higher differential-prior is, the more parameters can have bigger changes, so you want to keep this relatively small.  If you see overfitting (accuracy and fit increasing over iterations in tensorboard) you may consider reducing the differential-prior in order to reduce the parameter space.

**Q**. What's up with the `--training-column` argument?

**A**. That is used for cross-validation if you have a specific reproducibility question that you are interested in answering.  If this is specified, only samples labeled "Train" under this column will be used for building the model and samples labeled "Test" will be used for cross validation.  In other words the model will attempt to predict the microbe abundances for the "Test" samples.  The resulting prediction accuracy is used to evaluate the generalizability of the model in order to determine if the model is overfitting or not.  If this argument is not specified, then 10 random samples will be chosen for the test dataset.  If you want to specify more random samples to allocate for cross-validation, the `--num-random-test-examples` argument can be specified.

**Q**. How long should I expect this program to run?

**A**. That primarily depends on a few things, namely how many samples and microbes are in your dataset, and the number of `--epoch` and the `--batch-size`.  The `--batch-size` specifies the number of samples to analyze for each iteration, and the `--epoch` specifies the number of total passes through the dataset.  For example, if you have a 100 samples in your dataset and you specify `--batch-size 5` and `--epochs 200`, then you will have `(100/5)*200=4000` iterations total. The larger the batch size, the more samples you average per iteration, but the less iterations you have - which can sometimes buy you less time to reach convergence (so you may have to compensate by increasing the epoch).  On the other hand, if you decrease the batch size, you can have more iterations, but the variability between each iteration is higher. This also depends on if your program will converge.  This may also depend on the `--learning-rate` which specifies the resolution (smaller step size = smaller resolution, but may take longer to converge). You will need to consult with Tensorboard to make sure that your model fit is sane.  See this paper for more details on gradient descent: https://arxiv.org/abs/1609.04747

**Q**. I'm confused, what is Tensorboard?

**A**. Tensorboard is a diagnostic tool that runs in a web browser. To open tensorboard, make sure you’re in the songbird environment (`regression`) and `cd` into the folder you are running the script above from. Then run:

```bash
tensorboard --logdir .

Returning line will look something like:
TensorBoard 1.9.0 at http://Lisas-MacBook-Pro-2.local:6006 (Press CTRL+C to quit)
```

Open the website (highlighted in red) in a browser. (Hint; if that doesn’t work try putting only the port number (here it is 6006), adding `localhost`, localhost:6006). Leave this tab alone. Now any songbird output directories that you add to the folder that tensorflow is running in will be added to the webpage.

This should produce a website with 2 graphs, which tensorflow actively updates as songbird is running.
![tensorboard](https://github.com/biocore/songbird/raw/master/images/tensorboard-output.png "Tensorboard")
A description of these two graphs is outlined below.

FIRST graph in Tensorflow; 'Prediction accuracy'. Labelled  ‘accuracy/mean_absolute_error’

This is a graph of the prediction accuracy of the model; the model will try to guess the count values for the training samples that were set aside in the script above, using only the metadata categories it has. Then it looks at the real values and sees how close it was.

The x-axis is the number of iterations (meaning times the model is training across the entire dataset). Every time you iterate across the training samples, you also run the test samples and the averaged results are being plotted on the y-axis.

**Number of iterations = `--epoch #` multiplied by the `--batch-size` parameter**

The y-axis is the average number of counts off for each feature. The model is predicting the sequence counts for each feature in the samples that were set aside for testing. So in the graph above it means that, on average, the model is off by ~16 counts, which is low. However, this is ABSOLUTE error not relative error (unfortunately we can’t do relative errors because of the sparsity of metagenomic datasets)


**Q**. So how can you tell if this graph ‘looks good’??

**A**. The raw numbers will be variable, so it is difficult to make a blanket statement, but the most important thing is the shape of the graph. You want to see exponential decay and a stable plateau (further discussion below)

SECOND graph in Tensorflow; 'Model fit'

This graph is labelled ‘loss’ because ‘loss’ is the function being optimized. The goal here is to reduce the error of the training samples.

This graph represents how well the model fits your data.
Just like the prediction accuracy graph, the x-axis is the number of iterations (meaning times the model is training across the entire dataset).

The y-axis is MINUS log probability of the model actually fitting - so LOWER is better (maximizing the probability = minimizing the negative log probability).

**Q**. What does a good model fit look like??

**A**. Again the numbers vary greatly by dataset. But you want to see the curve decaying, and plateau as close to zero as possible (above example is a nice one).

**Q**. So how can you adjust your model to get nice exponential decays in both prediction accuracy and model fit??

**A**. If you simply change a parameter and run again (under a different output file name) that graph will pop up on top of the first graphs in tensorflow! You can click the graphs on and off in the lower left hand panel, and read just the axis for a given graph (or set of graphs) by clicking the blue expansion rectangle underneath the graph:

It's recommended to start with a small formula (few variables in the model) and increase from there, because it makes debuggin easier. If your graphs are going down but not exponentially and not plateauing, you should consider increasing the number of iterations by increasing `--epoch`

If your graphs are going down but then going back up, this suggests overfitting; try reducing the number of variables in your formula, or reducing `--differential-prior`. As a rule of thumb, you should try to keep the number of metadata categories less than 10% the number of samples (e.g. for 100 samples, no more than 10 metadata categories).

So basically we want to futz around with the parameters until we see two nice exponential decay graphs.
Once you have that, we can view the differentials.tsv output to look at the ranks.




Credits to Lisa Marotz ([@lisa55asil](https://github.com/lisa55asil)) for the FAQs notes.




# QIIME2 tutorial

First make sure that qiime2 is installed before installing songbird. In your qiime2 environment install songbird:
```
conda install songbird -c conda-forge
```
Then run"
```
qiime dev refresh-cache
```

Once qiime2 is properly interfaced with songbird, you can import your biom tables
into Artifacts.  Here we will be using a subset of the [Redsea metagenome dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315489/)
as an example.  Starting from the songbird root folder, you can import this dataset as follows

```
qiime tools import \
	--input-path data/redsea/redsea.biom \
	--output-path redsea.biom.qza \
	--type FeatureTable[Frequency]
```

You can then run the qiime2 songbird multinomial commmand as follows.

```
qiime songbird multinomial \
	--i-table redsea.biom.qza \
	--m-metadata-file data/redsea/redsea_metadata.txt \
	--p-formula "Depth+Temperature+Salinity+Oxygen+Fluorescence+Nitrate" \
	--o-differentials differentials.qza \
	--o-regression-stats regression-stats.qza \
	--o-regression-biplot regression-biplot.qza
```
Don't forget to try out the `--verbose` option.

Diagnostic plots can also be drawn from the qiime2 interface as follows
```
qiime songbird summarize-single \
    --i-feature-table redsea.biom.qza \
    --i-regression-stats regression-stats.qza \
    --o-visualization regression-summary
```
Are you able to see anything in your summary?  It is possible that your summary interval is not set correctly!
By default, it is set to 1 minute, so that it can take a measure every minute - but here you ran for just a few seconds!
Try setting `--p-summary-interval 1` to record the loss at every second and set `--p-epochs 5000` to make it run longer in the `multinomial` command.  Now try to summarize these new results.

One can also generate Qsquared values by comparing it to a baseline model as follows
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
    --o-visualization paired-summary
```

The baseline model above just looks at the means (i.e. intercept), to determine how much better the first model can perform compared to the baseline model.
But one can imagine using other baseline models to contrast - for instance, fitting a model on just Temperature to gauge how informative other variables such as Salinity and Oxygen are.  The Qsquared value is the predictive accuracy estimated from the samples left out of the regression fit.

The resulting differentials learned from the regression model can be visualized as a biplot, given from `regression-biplot.qza`.  You can directly visualize this in emperor

```
qiime emperor biplot \
	--i-biplot regression-biplot.qza \
	--m-sample-metadata-file data/redsea/feature_metadata.txt \
	--p-ignore-missing-samples \
	--p-number-of-features 7 \
	--o-visualization emperor-biplot
```

You can view the resulting visualization at https://view.qiime2.org

These biplots have a different interpretation - the points correspond to microbes and the arrow correspond to covariates of interest. Running these models on the full dataset can yield something similar to as follows
![biplot](https://github.com/biocore/songbird/raw/master/images/redsea-biplot.png "Regression biplot")

# Related packages

For interactively visualizing the differentials coming from songbird, definitely check out [rankratioviz](https://github.com/fedarko/rankratioviz)

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
