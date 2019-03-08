import os
import biom
import qiime2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _convergence_plot(regression, baseline, ax0, ax1):
    iterations = np.array(regression['iteration'])
    ax0.plot(iterations[1:],
             np.array(regression['loglikehood'])[1:], label='model')
    ax0.set_ylabel('Loglikehood', fontsize=14)
    ax0.set_xlabel('# Iterations', fontsize=14)

    ax1.plot(iterations[1:],
             np.array(regression['cross-validation'].values)[1:],
             label='model')
    ax1.set_ylabel('Cross validation score', fontsize=14)
    ax1.set_xlabel('# Iterations', fontsize=14)

    if baseline is not None:
        iterations = baseline['iteration']
        ax0.plot(iterations[1:],
                 np.array(baseline['loglikehood'])[1:], label='baseline')
        ax0.set_ylabel('Loglikehood', fontsize=14)
        ax0.set_xlabel('# Iterations', fontsize=14)
        ax0.legend()

        ax1.plot(iterations[1:],
                 np.array(baseline['cross-validation'].values)[1:],
                 label='baseline')
        ax1.set_ylabel('Cross validation score', fontsize=14)
        ax1.set_xlabel('# Iterations', fontsize=14)

        ax1.legend()


def _summarize(output_dir: str, n: int,
               regression: pd.DataFrame, baseline: pd.DataFrame = None):
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
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    if baseline is None:
        _convergence_plot(regression, None, ax[0], ax[1])
        q2 = None
    else:
        _convergence_plot(regression, baseline, ax[0], ax[1])

        # this provides a pseudo-r2 commonly provided in the context
        # of logistic / multinomail regression (proposed by Cox & Snell)
        # http://www3.stat.sinica.edu.tw/statistica/oldpdf/a16n39.pdf
        end = min(10, len(regression.index))
        # trim only the last 10 numbers

        # compute a q2 score, which is commonly used in
        # partial least squares for cross validation
        l0 = np.mean(baseline['cross-validation'][-end:])
        lm = np.mean(regression['cross-validation'][-end:])
        D = lm - l0
        q2 = np.exp(2 * D / n)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence-plot.svg'))
    fig.savefig(os.path.join(output_dir, 'convergence-plot.pdf'))

    index_fp = os.path.join(output_dir, 'index.html')
    with open(index_fp, 'w') as index_f:
        index_f.write('<html><body>\n')
        index_f.write('<h1>Convergence summary</h1>\n')
        if q2 is not None:
            index_f.write(
                'Pseudo Q-squared: %f\n' % q2
            )
        index_f.write(
            '<img src="convergence-plot.svg" alt="convergence_plots">'
        )
        index_f.write('<a href="convergence-plot.pdf">')
        index_f.write('Download as PDF</a><br>\n')


def summarize_single(output_dir: str,  feature_table: biom.Table,
                     regression_stats: qiime2.Metadata):
    n = feature_table.shape[1]
    _summarize(output_dir, n, regression_stats.to_dataframe())


def summarize_paired(output_dir: str, feature_table: biom.Table,
                     regression_stats: qiime2.Metadata,
                     baseline_stats: qiime2.Metadata):
    n = feature_table.shape[1]
    _summarize(output_dir, n,
               regression_stats.to_dataframe(),
               baseline_stats.to_dataframe())
