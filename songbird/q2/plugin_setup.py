# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import qiime2.plugin
import qiime2.sdk
from songbird import __version__
from ._method import multinomial, regression_biplot
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata)
from q2_types.feature_table import FeatureTable, Composition, Frequency
from q2_types.ordination import PCoAResults


# citations = qiime2.plugin.Citations.load(
#             'citations.bib', package='songbird')

plugin = qiime2.plugin.Plugin(
    name='songbird',
    version=__version__,
    website="https://github.com/mortonjt/songbird",
    # citations=[citations['morton2017balance']],
    short_description=('Plugin for building count regression models.'),
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
        'epoch': Int,
        'batch_size': Int,
        'beta_prior': Float,
        'learning_rate': Float,
        'clipnorm': Float,
        'min_sample_count': Int,
        'min_feature_count': Int,
        'summary_interval': Int
    },
    outputs=[
        ('coefficients',
         FeatureTable[Composition % Properties('coefficients')])
    ],
    input_descriptions={
        'table': 'Input table of counts.',
    },
    parameter_descriptions={
        'metadata': 'Sample metadata table with covariates of interest.',
        'formula': ('The statistical formula specifying covariates to be '
                    'included in the model and their interactions'),
        'num_random_test_examples': ("The number of random examples to select "
                                     "if `training_column` isn't specified"),
        'epoch': ('The number of total number of iterations '
                  'over the entire dataset'),
        'batch_size': ('The number of samples to be evaluated per '
                       'training iteration'),
        'beta_prior': ('Width of normal prior for the coefficients '
                       'Smaller values will regularize parameters towards '
                       'zero. Values must be greater than 0.'),
        'learning_rate': ('Gradient descent decay rate.'),

    },
    name='Multinomial regression',
    description=("Performs multinomial regression and calculates "
                 "rank differentials for organisms with respect to the "
                 "covariates of interest."),
    citations=[]
)



plugin.methods.register_function(
    function=regression_biplot,
    inputs={
        'coefficients': FeatureTable[Composition % Properties('coefficients')]
    },
    parameters={},
    outputs=[
        ('biplot', PCoAResults % Properties("biplot"))
    ],
    input_descriptions={
        'coefficients': 'Input table of coefficients',
    },
    parameter_descriptions={},
    output_descriptions={
        'biplot': ('A biplot of the regression coefficients')
    },
    name='Builds Multinomial regression biplot',
    description=("Performs multinomial regression and calculates "
                 "rank differentials for organisms with respect to the "
                 "covariates of interest."),
    citations=[]
)
