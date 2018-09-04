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
from q2_gneiss import __version__


#citations = qiime2.plugin.Citations.load('citations.bib', package='q2_gneiss')

plugin = qiime2.plugin.Plugin(
    name='gneiss',
    version=__version__,
    website="https://github.com/mortonjt/songbird",
    #citations=[citations['morton2017balance']],
    short_description=('Plugin for building regression models.'),
    description=('This is a QIIME 2 plugin supporting statistical models on '
                 'feature tables and metadata.'),
    package='songbird')
