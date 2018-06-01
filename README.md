# Installation
First make sure to install Tensorflow, namely
```
conda create -n regression tensorflow -c conda-forge
source activate regression
```
Then install this package from source
```
pip install git+https://github.com/mortonjt/regression.git
```
# Getting started

First split your dataset into train/test datasets

``` 
process.py split_dataset \ 
    --input_biom <your-biom>.biom \
    --input_metadata <your-metadata>.txt \
    --split_ratio 0.1 \
    --output_dir <output_results>
```

Then you can run the multinomial regression
```
multinomial.py \
    --formula "your_formula"
    --train_biom <your-training-biom>.biom\
    --train_metadata <your-training-metadata>.txt \
    --test_biom <your-testing-biom>.biom \
    --test_metadata <your-testing-metadata>.txt \
    --batch_size 10 \
    --save_path <results>
```
All of the coefficients are stored under `<results>/beta.csv` and intercepts are stored under `<results>/gamma.csv`.
The reported results, namely `MSE` and `MRC` report cross validation results.  `MSE` is the mean squared errors on the counts, the square root of which corresponds to the
average number of counts that is expected to be wrong in a cell.  `MRC` is the average rank correlation between the predicted rank abundances and the observed rank abundances in the test dataset.  `MRC` values above 0.2 are generally acceptable, as it indicates roughly 20% of the top species are current.

The most important aspect of the coefficients are the rankings, or the ordering of the coefficients within a covariate.