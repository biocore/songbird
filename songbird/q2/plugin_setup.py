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

from qiime2.plugin import (Str, Properties, Int, Float,  Metadata)
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.ordination import PCoAResults
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from songbird.q2 import (
    SongbirdStats, SongbirdStatsFormat, SongbirdStatsDirFmt,
    Differential, DifferentialFormat, DifferentialDirFmt,
    multinomial, summarize_single, summarize_paired
)


# citations = qiime2.plugin.Citations.load(
#             'citations.bib', package='songbird')

plugin = qiime2.plugin.Plugin(
    name='songbird',
    version=__version__,
    website="https://github.com/mortonjt/songbird",
    # citations=[citations['morton2017balance']],
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
        'summary_interval': Int
    },
    outputs=[
        ('differentials', FeatureData[Differential]),
        ('regression_stats', SampleData[SongbirdStats]),
        ('regression_biplot', PCoAResults % Properties('biplot'))
    ],
    input_descriptions={
        'table': 'Input table of counts.',
    },
    output_descriptions={
        'differentials': ('Output differentials learned from the '
                          'multinomial regression.'),
        'regression_stats': ('Summary information about the loss '
                             'and cross validation error over iterations.'),
        'regression_biplot': ('A biplot of the regression coefficients')
    },
    parameter_descriptions={
        'metadata': 'Sample metadata table with covariates of interest.',
        'formula': ('The statistical formula specifying covariates to be '
                    'included in the model and their interactions'),
        'num_random_test_examples': (
            'Number of random samples to hold out for cross-validation '
            'if `training_column` is not specified'),
        'epochs': ('The number of total number of iterations '
                   'over the entire dataset'),
        'batch_size': ('The number of samples to be evaluated per '
                       'training iteration'),
        'differential_prior': (
            'Width of normal prior for the `differentials`, or '
            'the coefficients of the multinomial regression. '
            'Smaller values will force the coefficients towards zero. '
            'Values must be greater than 0.'),
        'learning_rate': ('Gradient descent decay rate.'),
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
        'feature_table': FeatureTable[Frequency],
        'regression_stats': SampleData[SongbirdStats]
    },
    parameters={},
    input_descriptions={
        'feature_table': ('Input biom table that was used for the '
                          'regression analysis.'),
        'regression_stats': ('results from multinomial regression '
                             'for reference model')
    },
    parameter_descriptions={
    },
    name='Regression summary statistics',
    description=("Visualize the convergence statistics of regression fit "
                 "including cross validation accuracy and the loglikehood "
                 "over the iterations")
)

plugin.visualizers.register_function(
    function=summarize_paired,
    inputs={
        'feature_table': FeatureTable[Frequency],
        'regression_stats': SampleData[SongbirdStats],
        'baseline_stats': SampleData[SongbirdStats]
    },
    parameters={},
    input_descriptions={
        'feature_table': ('Input biom table that was used for the '
                          'regression analysis.'),
        'regression_stats': ('results from multinomial regression '
                             'for reference model'),
        'baseline_stats': ('results from multinomial regression '
                           'for baseline model')
    },
    parameter_descriptions={
    },
    name='Paired regression summary statistics',
    description=("Visualize the convergence statistics of regression fit "
                 "including cross validation accuracy, loglikehood over the "
                 "iterations and the R2.")
)

# Register types
plugin.register_formats(SongbirdStatsFormat, SongbirdStatsDirFmt)
plugin.register_semantic_types(SongbirdStats)
plugin.register_semantic_type_to_format(
    SampleData[SongbirdStats], SongbirdStatsDirFmt)

plugin.register_formats(DifferentialFormat, DifferentialDirFmt)
plugin.register_semantic_types(Differential)
plugin.register_semantic_type_to_format(
    FeatureData[Differential], DifferentialDirFmt)

importlib.import_module('songbird.q2._transformer')
