import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from skbio.stats.composition import clr_inv as softmax
from biom import Table
from patsy import dmatrix


def random_multinomial_model(num_samples, num_features,
                             reps=1,
                             low=2, high=10,
                             beta_mean=0,
                             beta_scale=5,
                             mu=1,
                             sigma=1,
                             seed=0):
    """ Generates a table using a random poisson regression model.

    Here we will be simulating microbial counts given the model, and the
    corresponding model priors.

    Parameters
    ----------
    num_samples : int
        Number of samples
    num_features : int
        Number of features
    tree : np.array
        Tree specifying orthonormal contrast matrix.
    low : float
        Smallest gradient value.
    high : float
        Largest gradient value.
    beta_mean : float
        Mean of beta prior (for regression coefficients)
    beta_scale : float
        Scale of beta prior (for regression coefficients)
    mu : float
        Mean sequencing depth (in log units)
    sigma : float
        Variance for sequencing depth

    Returns
    -------
    table : biom.Table
        Biom representation of the count table.
    metadata : pd.DataFrame
        DataFrame containing relevant metadata.
    beta : np.array
        Regression parameter estimates.
    """
    N = num_samples

    # generate all of the coefficient using the random poisson model
    state = check_random_state(seed)
    beta = state.normal(beta_mean, beta_scale, size=(2, num_features-1))

    X = np.hstack([np.linspace(low, high, num_samples // reps)]
                  for _ in range(reps))
    X = np.vstack((np.ones(N), X)).T
    phi = np.hstack((np.zeros((N, 1)), X @ beta))
    probs = softmax(phi)
    n = [mu] * N

    table = np.vstack(
        state.multinomial(n[i], probs[i, :])
        for i in range(N)
    ).T

    samp_ids = pd.Index(['S%d' % i for i in range(num_samples)],
                        name='sampleid')
    feat_ids = ['F%d' % i for i in range(num_features)]
    balance_ids = ['L%d' % i for i in range(num_features-1)]

    table = Table(table, feat_ids, samp_ids)
    metadata = pd.DataFrame(X, columns=['Ones', 'X'], index=samp_ids)
    beta = pd.DataFrame(beta.T, columns=['Intercept', 'beta'],
                        index=balance_ids)

    return table, metadata, beta


def _type_cast_to_float(df):
    """ Attempt to cast all of the values in dataframe to float.

    This will try to type cast all of the series within the
    dataframe into floats.  If a column cannot be type casted,
    it will be kept as is.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # TODO: Will need to improve this, as this is a very hacky solution.
    for c in df.columns:
        s = df[c]
        try:
            df[c] = s.astype(np.float64)
        except Exception:
            continue
    return df


def read_metadata(filepath):
    """ Reads in a sample metadata file

    Parameters
    ----------
    filepath: str
       The file path location of the sample metadata file

    Returns
    -------
    pd.DataFrame :
       The metadata table with inferred types.
    """
    metadata = pd.read_table(
        filepath, dtype=object)
    cols = metadata.columns
    metadata = metadata.set_index(cols[0])
    metadata = _type_cast_to_float(metadata.copy())

    return metadata


def match_and_filter(table, metadata, formula,
                     min_sample_count, min_feature_count):
    """ Matches and aligns biom and metadata tables.

    This will also return the patsy representation.

    Parameters
    ----------
    table : biom.Table
        Table of abundances
    metadata : pd.DataFrame
        Sample metadata

    Returns
    -------
    table : biom.Table
        Filtered biom table
    metadata : pd.DataFrame
        Sample metadata
    """
    # match them

    metadata = metadata.loc[table.ids(axis='sample')]

    def sample_filter(val, id_, md):
        return id_ in metadata.index and np.sum(val) > min_sample_count

    def read_filter(val, id_, md):
        return np.sum(val > 0) > min_feature_count

    def metadata_filter(val, id_, md):
        return id_ in metadata.index

    table = table.filter(metadata_filter, axis='sample')
    table = table.filter(sample_filter, axis='sample')
    table = table.filter(read_filter, axis='observation')

    metadata = metadata.loc[table.ids(axis='sample')]

    def sort_f(xs):
        return [xs[metadata.index.get_loc(x)] for x in xs]

    table = table.sort(sort_f=sort_f, axis='sample')
    design = dmatrix(formula, metadata, return_type='dataframe')
    design = design.dropna()

    def design_filter(val, id_, md):
        return id_ in design.index

    table = table.filter(design_filter, axis='sample')
    return table, metadata, design


def split_training(dense_table, metadata, design, training_column=None,
                   num_random_test_examples=10):

    if training_column is None:
        idx = np.random.random(design.shape[0])
        i = np.argsort(idx)[num_random_test_examples]

        threshold = idx[i]
        train_idx = ~(idx < threshold)
    else:
        train_idx = metadata.loc[design.index, training_column] == "Train"

    trainX = design.loc[train_idx].values
    testX = design.loc[~train_idx].values

    trainY = dense_table.loc[train_idx].values
    testY = dense_table.loc[~train_idx].values

    return trainX, testX, trainY, testY
