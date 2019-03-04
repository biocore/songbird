import biom
import skbio
import qiime2
import pandas as pd
import numpy as np
import tensorflow as tf
from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv
from songbird.multinomial import MultRegression
from songbird.util import match_and_filter, split_training
from qiime2.plugin import Metadata


def multinomial(table: biom.Table,
                metadata: Metadata,
                formula: str,
                training_column: str = None,
                num_random_test_examples: int = 5,
                epochs: int = 1000,
                batch_size: int = 5,
                differential_prior: float = 1,
                learning_rate: float = 1e-3,
                clipnorm: float = 10,
                min_sample_count: int = 1000,
                min_feature_count: int = 10,
                summary_interval: int = 60) -> (
                    pd.DataFrame, qiime2.Metadata, skbio.OrdinationResults
                ):

    # load metadata and tables
    metadata = metadata.to_dataframe()
    # match them
    table, metadata, design = match_and_filter(
        table, metadata,
        formula, min_sample_count, min_feature_count
    )

    # convert to dense representation
    dense_table = table.to_dataframe().to_dense().T

    # split up training and testing
    trainX, testX, trainY, testY = split_training(
        dense_table, metadata, design,
        training_column, num_random_test_examples
    )

    model = MultRegression(learning_rate=learning_rate, clipnorm=clipnorm,
                           beta_mean=differential_prior,
                           batch_size=batch_size,
                           save_path=None)
    with tf.Graph().as_default(), tf.Session() as session:
        model(session, trainX, trainY, testX, testY)

        loss, cv, its = model.fit(
            epochs=epochs,
            summary_interval=summary_interval,
            checkpoint_interval=None)

    md_ids = np.array(design.columns)
    obs_ids = table.ids(axis='observation')

    beta_ = clr(clr_inv(np.hstack((np.zeros((model.p, 1)), model.B))))

    differentials = pd.DataFrame(
        beta_.T, columns=md_ids, index=obs_ids,
    )
    convergence_stats = pd.DataFrame(
        {
            'loglikehood': loss,
            'cross-validation': cv,
            'iteration': its
        }
    )

    convergence_stats.index.name = 'id'
    convergence_stats.index = convergence_stats.index.astype(np.str)

    c = convergence_stats['loglikehood'].astype(np.float)
    convergence_stats['loglikehood'] = c

    c = convergence_stats['cross-validation'].astype(np.float)
    convergence_stats['cross-validation'] = c

    c = convergence_stats['iteration'].astype(np.int)
    convergence_stats['iteration'] = c

    # regression biplot
    if differentials.shape[-1] > 1:
        u, s, v = np.linalg.svd(differentials)
        pc_ids = ['PC%d' % i for i in range(len(s))]
        samples = pd.DataFrame(u[:, :len(s)] @ np.diag(s),
                               columns=pc_ids, index=differentials.index)
        features = pd.DataFrame(v.T[:, :len(s)],
                                columns=pc_ids, index=differentials.columns)
        short_method_name = 'regression_biplot'
        long_method_name = 'Multinomial regression biplot'
        eigvals = pd.Series(s, index=pc_ids)
        proportion_explained = eigvals**2 / (eigvals**2).sum()
        biplot = OrdinationResults(
            short_method_name, long_method_name, eigvals,
            samples=samples, features=features,
            proportion_explained=proportion_explained)
    else:
        # this is to handle the edge case with only intercepts
        biplot = OrdinationResults('', '', pd.Series(), pd.DataFrame())

    return differentials, qiime2.Metadata(convergence_stats), biplot
