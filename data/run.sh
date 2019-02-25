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
    --epoch 10000 \
    --num-random-test-examples 5 \
    --summary-interval 1



# Options:
#   --input-biom TEXT               Input abundances
#   --metadata-file TEXT            Input microbial abundances for testing
#   --formula TEXT                  statistical formula specifying the
#                                   covariates to test for.
#   --training-column TEXT          The column in the metadata file used to
#                                   specify training and testing. These columns
#                                   should be specifically labeled (Train) and
#                                   (Test)
#   --num-random-test-examples INTEGER
#                                   Number of random training examples if
#                                   --training-column is not specified
#   --epoch INTEGER                 Number of epochs to train
#   --batch-size INTEGER            Size of mini-batch
#   --beta-prior FLOAT              Width of normal prior for the coefficients
#                                   Smaller values will regularize parameters
#                                   towards zero. Values must be greater than 0.
#   --learning-rate FLOAT           Gradient descent decay rate.
#   --clipnorm FLOAT                Gradient clipping size.
#   --min-sample-count INTEGER      The minimum number of counts a sample needs
#                                   for it to be included in the analysis
#   --min-feature-count INTEGER     The minimum number of counts a feature needs
#                                   for it to be included in the analysis
#   --checkpoint-interval INTEGER   Number of seconds before a saving a
#                                   checkpoint
#   --summary-interval INTEGER      Number of seconds before a storing a
#                                   summary.
#   --summary-dir TEXT              Summary directory to save cross validation
#                                   results.
#   --help                          Show this message and exit.
