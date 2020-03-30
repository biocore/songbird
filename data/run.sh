# songbird multinomial \
#     --input-biom feature-table.biom \
#     --metadata-file Workbook4.txt \
#     --formula Cat \
#     --min-feature-count 10

formula="Depth+Temperature+Salinity+Oxygen+Fluorescence+Turbidity+Nitrate+Phosphate"
mdfile=redsea/redsea_metadata.txt
biom=redsea/redsea.biom
songbird multinomial \
    --input-biom $biom \
    --metadata-file $mdfile \
    --formula $formula \
    --min-feature-count 10 \
    --batch-size 3 \
    --epochs 10000 \
    --num-random-test-examples 5 \
    --summary-dir summary1 \
    --summary-interval 1


formula="Depth+Temperature+Salinity+Oxygen"
songbird multinomial \
    --input-biom $biom \
    --metadata-file $mdfile \
    --formula $formula \
    --min-feature-count 10 \
    --batch-size 3 \
    --epochs 10000 \
    --num-random-test-examples 5 \
    --summary-dir summary2 \
    --summary-interval 1


formula="Depth+Temperature"
songbird multinomial \
    --input-biom $biom \
    --metadata-file $mdfile \
    --formula $formula \
    --min-feature-count 10 \
    --batch-size 3 \
    --epochs 10000 \
    --num-random-test-examples 5 \
    --summary-dir summary3 \
    --summary-interval 1

#Usage: songbird multinomial [OPTIONS]
#
#Options:
#  --input-biom TEXT               Input abundances  [required]
#  --metadata-file TEXT            Sample metadata table with covariates of
#                                  interest.  [required]
#  --formula TEXT                  Statistical formula specifying the
#                                  covariates to test for.  [required]
#  --training-column TEXT          The column in the metadata file used to
#                                  specify training and testing. These columns
#                                  should be specifically labeled (Train) and
#                                  (Test)
#  --num-random-test-examples INTEGER
#                                  Number of random training examples if
#                                  --training-column is not specified
#                                  [default: 5]
#  --epochs INTEGER                The number of total number of iterations
#                                  over the entire dataset  [default: 1000]
#  --batch-size INTEGER            Size of mini-batch  [default: 5]
#  --differential-prior FLOAT      Width of normal prior for the
#                                  `differentials`, or the coefficients of the
#                                  multinomial regression. Smaller values will
#                                  force the coefficients towards zero. Values
#                                  must be greater than 0.  [default: 1.0]
#  --learning-rate FLOAT           Gradient descent decay rate.  [default:
#                                  0.001]
#  --clipnorm FLOAT                Gradient clipping size.  [default: 10.0]
#  --min-sample-count INTEGER      The minimum number of counts a sample needs
#                                  for it to be included in the analysis
#                                  [default: 1000]
#  --min-feature-count INTEGER     The minimum number of counts a feature needs
#                                  for it to be included in the analysis
#                                  [default: 5]
#  --checkpoint-interval INTEGER   Number of seconds before a saving a
#                                  checkpoint  [default: 3600]
#  --summary-interval INTEGER      Number of seconds before a storing a
#                                  summary.  [default: 60]
#  --summary-dir TEXT              Summary directory to save regression
#                                  results. This will include a table of
#                                  differentials under `differentials.tsv` that
#                                  can be ranked, in addition to summaries that
#                                  can be loaded into Tensorboard and
#                                  checkpoints for recovering parameters during
#                                  runtime.  [default: summarydir]
#  --help                          Show this message and exit.
