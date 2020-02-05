# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import importlib
import qiime2.plugin
import qiime2.sdk
from songbird import __version__

from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.ordination import PCoAResults
from q2_types.sample_data import SampleData
from q2_types.feature_data import (FeatureData, Differential)
from songbird.q2 import (
    SongbirdStats, SongbirdStatsFormat, SongbirdStatsDirFmt,
    multinomial, summarize_single, summarize_paired
)
from songbird.parameter_info import DESCS

citations = qiime2.plugin.Citations.load('citations.bib', package='songbird')

plugin = qiime2.plugin.Plugin(
    name='songbird',
    version=__version__,
    website="https://github.com/mortonjt/songbird",
    citations=[citations['MortonMarotz2019']],
    short_description=('Plugin for differential abundance analysis '
                       'via count-based models.'),
    description=('This is a QIIME 2 plugin supporting statistical models on '
                 'feature tables and metadata.'),
    package='songbird')

plugin.methods.register_function(
    function=multinomial,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'metadata': Metadata,
        'formula': Str,
        'training_column': Str,
        'num_random_test_examples': Int,
        'epochs': Int,
        'batch_size': Int,
        'differential_prior': Float,
        'learning_rate': Float,
        'clipnorm': Float,
        'min_sample_count': Int,
        'min_feature_count': Int,
        'summary_interval': Float,
        'random_seed': Int,
        'silent': Bool,
    },
    outputs=[
        ('differentials', FeatureData[Differential]),
        ('regression_stats', SampleData[SongbirdStats]),
        ('regression_biplot', PCoAResults % Properties('biplot'))
    ],
    input_descriptions={
        'table': DESCS["table"],
    },
    output_descriptions={
        'differentials': ('Output differentials learned from the '
                          'multinomial regression.'),
        'regression_stats': ('Summary information about the loss '
                             'and cross validation error over iterations.'),
        'regression_biplot': ('A biplot of the regression coefficients')
    },
    parameter_descriptions={
        'metadata': DESCS["metadata"],
        'formula': DESCS["formula"],
        "training_column": DESCS["training-column"],
        'num_random_test_examples': DESCS["num-random-test-examples"],
        'epochs': DESCS["epochs"],
        'batch_size': DESCS["batch-size"],
        'differential_prior': DESCS["differential-prior"],
        'learning_rate': DESCS["learning-rate"],
        "clipnorm": DESCS["clipnorm"],
        "min_sample_count": DESCS["min-sample-count"],
        "min_feature_count": DESCS["min-feature-count"],
        "summary_interval": DESCS["summary-interval"],
        "random_seed": DESCS["random-seed"],
        "silent": (DESCS["silent"] + " (Only has an impact when using this "
                   "command with the --verbose option or through the Qiime2 "
                   "Artifact API)"),
    },
    name='Multinomial regression',
    description=("Performs multinomial regression and calculates "
                 "rank differentials for organisms with respect to the "
                 "covariates of interest."),
    citations=[]
)

plugin.visualizers.register_function(
    function=summarize_single,
    inputs={
        'regression_stats': SampleData[SongbirdStats]
    },
    parameters={},
    input_descriptions={
        'regression_stats': (
            "Summary information produced by running "
            "`qiime songbird multinomial`."
        )
    },
    parameter_descriptions={
    },
    name='Regression summary statistics',
    description=(
        "Visualize the convergence statistics from running multinomial "
        "regression, giving insight into how the model fit to your data."
    )
)

plugin.visualizers.register_function(
    function=summarize_paired,
    inputs={
        'regression_stats': SampleData[SongbirdStats],
        'baseline_stats': SampleData[SongbirdStats]
    },
    parameters={},
    input_descriptions={

        'regression_stats': (
            "Summary information for the reference model, produced by running "
            "`qiime songbird multinomial`."
        ),
        'baseline_stats': (
            "Summary information for the baseline model, produced by running "
            "`qiime songbird multinomial`."
        )

    },
    parameter_descriptions={
    },
    name='Paired regression summary statistics',
    description=(
        "Visualize the convergence statistics from two runs of multinomial "
        "regression, giving insight into how the models fit to your data. "
        "The produced visualization includes a 'pseudo-Q-squared' value."
    )
)

# Register types
plugin.register_formats(SongbirdStatsFormat, SongbirdStatsDirFmt)
plugin.register_semantic_types(SongbirdStats)
plugin.register_semantic_type_to_format(
    SampleData[SongbirdStats], SongbirdStatsDirFmt)

importlib.import_module('songbird.q2._transformer')
