# Parameter Descriptions
DESCS = {
    "table": "Input table of counts.",
    "metadata": "Sample metadata file with covariates of interest.",
    "formula": (
        'The statistical formula specifying covariates to be included in the '
        'model and their interactions.'
    ),
    "training-column": (
        'The column in the metadata file used to '
        'specify training and testing. These columns '
        'should be specifically labeled (Train) and (Test).'
    ),
    "num-random-test-examples": (
        'Number of random samples to hold out for cross-validation if '
        'a training column is not specified.'
    ),
    "epochs": 'The total number of iterations over the entire dataset.',
    "batch-size": (
        "The number of samples to be evaluated per training iteration."
    ),
    "differential-prior": (
        'Width of normal prior for the `differentials`, or '
        'the coefficients of the multinomial regression. '
        'Smaller values will force the coefficients towards zero. '
        'Values must be greater than 0.'
    ),
    'learning-rate': 'Gradient descent decay rate.',
    'clipnorm': "Gradient clipping size.",
    'min-sample-count': (
        "The minimum number of counts a sample needs for it to be included in "
        "the analysis."
    ),
    'min-feature-count': (
        "The minimum number of samples a feature needs to be observed in "
        "for it to be included in the analysis."
    ),
    "summary-interval": "Number of seconds before storing a summary.",
    "silent": (
        'Flag denoting whether to hide progress bar and TensorFlow warnings. '
        'If False, progress bar and warnings are displayed. If True, neither '
        'progress bar nor warnings are displayed.'
    ),
    # FYI: The following parameters are exclusive to the non-Q2 Songbird script
    "checkpoint-interval": 'Number of seconds before a saving a checkpoint.',
    "summary-dir": (
        'Summary directory to save regression results to. '
        'This will include a table of differentials under '
        '`differentials.tsv` that can be ranked, in addition '
        'to summaries that can be loaded into Tensorboard and '
        'checkpoints for recovering parameters during runtime.'
    ),
    "random-seed": (
        'The number to used to receive consistent results for the random  '
        'processes in the fitting procedure.'
    ),
}

DEFAULTS = {
    "training-column": None,
    "num-random-test-examples": 5,
    "epochs": 1000,
    "batch-size": 5,
    "differential-prior": 1.,
    'learning-rate': 1e-3,
    'clipnorm': 10.,
    'min-sample-count': 1000,
    'min-feature-count': 10,
    "checkpoint-interval": 3600,
    "summary-interval": 10,
    "summary-dir": "summarydir",
    "random-seed": 0,
    "silent": False,
}
