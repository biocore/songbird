import click
import pandas as pd
import numpy as np
import os
from util import random_multinomial_model, random_poisson_model
from gneiss.cluster import rank_linkage
from biom.util import biom_open
from biom.table import Table

from biom import load_table
from scipy.sparse import coo_matrix


@click.group()
def process():
    pass

@process.command()
@click.option('--input_biom', help='Input biom table')
@click.option('--input_metadata', help='Input metadata')
@click.option('--split_ratio', default=0.75,
              help='Number of training vs test examples')
@click.option('--output_dir', help='output directory')
def split_dataset(input_biom, input_metadata, split_ratio, output_dir):
    table = load_table(input_biom)
    metadata = pd.read_table(input_metadata, index_col=0)
    metadata.columns = [x.replace('-', '_') for x in metadata.columns]

    metadata_filter = lambda val, id_, md: id_ in metadata.index
    table = table.filter(metadata_filter, axis='sample')
    metadata = metadata.loc[table.ids(axis='sample')]

    sample_ids = metadata.index
    D, N = table.shape
    samples = pd.Series(np.arange(N), index=sample_ids)
    train_size = int(N * split_ratio)
    test_size = N - train_size

    test_samples = set(np.random.choice(sample_ids, size=test_size))

    test_idx =  np.array([(x in test_samples) for x in metadata.index])
    train_idx = ~test_idx
    f = lambda id_, md: id_ in test_samples
    gen = table.partition(f)

    _, train_table = next(gen)
    _, test_table = next(gen)

    train_metadata = metadata.iloc[train_idx]
    test_metadata = metadata.iloc[test_idx]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    test_metadata_path = os.path.join(
        output_dir, 'test_' + os.path.basename(input_metadata))
    train_metadata_path = os.path.join(
        output_dir, 'train_' + os.path.basename(input_metadata))

    test_biom_path = os.path.join(output_dir, 'test_' + os.path.basename(input_biom))
    train_biom_path = os.path.join(output_dir, 'train_' + os.path.basename(input_biom))

    print(train_metadata_path)
    train_metadata.to_csv(train_metadata_path, sep='\t')
    test_metadata.to_csv(test_metadata_path, sep='\t')

    with biom_open(train_biom_path, 'w') as f:
        train_table.to_hdf5(f, "train")

    with biom_open(test_biom_path, 'w') as f:
        test_table.to_hdf5(f, "test")


if __name__ == "__main__":
    process()
