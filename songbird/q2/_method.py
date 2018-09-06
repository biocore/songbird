import biom
import pandas as pd
import numpy as np
import tensorflow as tf
from skbio.stats.composition import clr, clr_inv
from songbird.multinomial import MultRegression
from songbird.util import match_and_filter, split_training
from qiime2.plugin import Metadata


def multinomial(table: biom.Table,
                metadata: Metadata,
                formula: str,
                training_column: str,
                num_random_test_examples: int=10,
                epoch: int=10,
                batch_size: int=5,
                beta_prior: float=1,
                learning_rate: float=0.1,
                clipnorm: float=10,
                min_sample_count: int=10,
                min_feature_count: int=10,
                summary_interval: int=60) -> (
                    pd.DataFrame
                ):

    # load metadata and tables
    metadata = metadata.to_dataframe()

    # match them
    table, metadata, design = match_and_filter(table, metadata)

    # convert to dense representation
    dense_table = table.to_dataframe().to_dense().T

    # split up training and testing
    trainX, trainY, testX, testY = split_training(
        dense_table, metadata, design,
        training_column, num_random_test_examples
    )

    model = MultRegression(learning_rate=learning_rate, clipnorm=clipnorm,
                           beta_mean=beta_prior,
                           batch_size=batch_size,
                           save_path=None)
    with tf.Graph().as_default(), tf.Session() as session:
        model(session, trainX, trainY, testX, testY)

        model.fit(
            epoch=epoch,
            summary_interval=summary_interval,
            checkpoint_interval=None)

    md_ids = np.array(design.columns)
    obs_ids = table.ids(axis='observation')

    beta_ = clr(clr_inv(np.hstack((np.zeros((model.p, 1)), model.B))))

    beta_ = pd.DataFrame(
        beta_.T, columns=md_ids, index=obs_ids,
    )
    return beta_
