import os
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
    iterations = np.arange(len(regression.index))
    ax0.plot(iterations[1:], np.array(regression['loglikehood'])[1:])
    ax0.set_ylabel('Loglikehood', fontsize=14)
    ax0.set_xlabel('# Iterations', fontsize=14)

    ax1.plot(iterations[1:], np.array(regression['cross-validation'].values)[1:])
    ax1.set_ylabel('Cross validation score', fontsize=14)
    ax1.set_xlabel('# Iterations', fontsize=14)


def _summarize(output_dir: str, n: int,
               regression: pd.DataFrame, baseline : pd.DataFrame=None):
    """ Helper method for generating summary pages

    Parameters
    ----------
    output_dir : str
       Name of output directory
    n : int
       Number of samples.
    regression : pd.DataFrame
       Regression summary with column names
       ['loglikehood', 'cross-validation']
    baseline : pd.DataFrame
       Baseline regression summary with column names
       ['loglikehood', 'cross-validation']

    Note
    ----
    This assumes that the same summary interval was used
    for both analyses.
    """

    if baseline is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        _convergence_plot(regression, baseline, ax[0], ax[1])
    else:
        # this provides a pseudo-r2 commonly provided in the context
        # of logistic / multinomail regression (proposed by Cox & Snell)
        # http://www3.stat.sinica.edu.tw/statistica/oldpdf/a16n39.pdf
        bound = min(len(baseline.index), len(regression.index))
        D = np.array(regression['loglikehood'][:bound] -
                     baseline['loglikehood'][:bound])
        # need to normalize so that the max is 1.
        r2 = np.exp(2 * D / n)

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        _convergence_plot(regression, baseline, ax[0], ax[1])
        iterations = np.arange(bound)
        ax[2].plot(iterations[1:], r2[1:])
        ax[2].set_ylabel('Pseudo R-squared', fontsize=14)
        ax[2].set_xlabel('# Iterations', fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence-plot.svg'))
    fig.savefig(os.path.join(output_dir, 'convergence-plot.pdf'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Convergence summary</h1>\n')
        index_f.write('<img src="convergence-plot.svg" alt="convergence_plots">')
        index_f.write('<a href="convergence-plot.pdf">')
        index_f.write('Download as PDF</a><br>\n')


def single_summary(output_dir: str,  table : biom.Table,
                   regression: pd.DataFrame):
    n = table.shape[1]
    _summarize(output_dir, n, regression)


def paired_summary(output_dir: str, table : biom.Table,
                   regression: pd.DataFrame, baseline: pd.DataFrame):
    n = table.shape[1]
    _summarize(output_dir, n, regression, baseline)
