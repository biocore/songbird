import biom
import skbio
import pandas as pd
import numpy as np
import tensorflow as tf
from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv, centralize
from songbird.multinomial import MultRegression
from songbird.util import match_and_filter, split_training
import matplotlib.pyplot as plt
from qiime2.plugin import Metadata


def _convergence_plot(regression, baseline, ax0, ax1):
        ax0.plot(iterations, np.array(regression['loglikehood']))
        ax0.ylabel('Loglikehood', fontsize=14)
        ax0.xlabel('# Iterations', fontsize=14)

        ax1.plot(iterations, np.array(regression['cross-validation'].values))
        ax1.ylabel('Cross validation score', fontsize=14)
        ax1.xlabel('# Iterations', fontsize=14)


def _summarize(n, regression, baseline=None):
    """ Helper method for generating summary pages

    Parameters
    ----------
    n : int
       Number of samples.
    regression : pd.DataFrame
       Regression summary with column names
       ['loglikehood', 'cross-validation']
    baseline : pd.DataFrame
       Baseline regression summary with column names
       ['loglikehood', 'cross-validation']
    """
    iterations = np.arange(regression)

    if baseline is None:
        fig, ax = plt.figure(1, 2, figsize=(10, 10))
        _convergence_plot(regression, baseline, ax[0], ax[1])
    else:
        # this provides a pseudo-r2 commonly provided in the context
        # of logistic / multinomail regression (proposed by Cox & Snell)
        # http://www3.stat.sinica.edu.tw/statistica/oldpdf/a16n39.pdf
        D = np.array(regression['loglikehood'] - regression['loglikehood'])
        # need to normalize so that the max is 1.
        rmax = 1 - np.exp(regression['loglikehood'] * 2 / n)
        r2 = np.exp(-2 * D / n) / rmax

        fig, ax = plt.figure(1, 3, figsize=(10, 10))
        _convergence_plot(regression, baseline, ax[0], ax[1])

        ax[2].plot(iterations, r2)
        ax[2].ylabel('Pseudo R-squared', fontsize=14)
        ax[2].xlabel('# Iterations', fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence-plot.svg'))
    fig.savefig(os.path.join(output_dir, 'convergence-plot.pdf'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Convergence summary</h1>\n')
        index_f.write('<img src="convergence-stats.svg" alt="heatmap">')
        index_f.write('<a href="heatmap.pdf">')
        index_f.write('Download as PDF</a><br>\n')
        index_f.write('<style>%s</style>' % css)
        index_f.write('<div class="square numerator">'
                      'Numerator<br/></div>')
        index_f.write('<div class="square denominator">'
                      'Denominator<br/></div>')
        index_f.write('</body></html>\n')


def single_summary(table, regression):
    n = table.shape[1]
    _summarize(n, regression)


def paired_summary(table, regression, baseline):
    n = table.shape[1]
    _summarize(n, regression, baseline)
