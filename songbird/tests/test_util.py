from io import StringIO
import numpy as np
import unittest
from songbird.util import read_metadata, match_and_filter, split_training
from biom import Table
import pandas as pd
import pandas.util.testing as pdt
import numpy.testing as npt


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.data1 = StringIO(
            '\n'.join(
                [
                    "sampleid\tcol1\tcol2\tcol3",
                    "0\t0\t2\t4",
                    "1\t1\t3\t5",
                    "3\t4\t8\t12"
                ]
            )
        )

        self.data2 = StringIO(
            '\n'.join(
                [
                    "sampleid\tcol1\tcol2\tcol3",
                    "a\t0\t1\t4",
                    "1\t1\ta\t5",
                    "3\t4\tc\t12"
                ]
            )
        )

    def test_read_index(self):
        res = read_metadata(self.data1)
        exp = pd.DataFrame(
            [
                ['0', 0.0, 2.0,  4.0],
                ['1', 1.0, 3.0,  5.0],
                ['3', 4.0, 8.0, 12.0]
            ],
            columns=['sampleid', 'col1', 'col2', 'col3'])

        exp = exp.set_index('sampleid')
        pdt.assert_frame_equal(exp, res)

    def test_read_categorical(self):
        res = read_metadata(self.data2)
        exp = pd.DataFrame(
            [
                ['a', 0.0, '1',  4.0],
                ['1', 1.0, 'a',  5.0],
                ['3', 4.0, 'c', 12.0]
            ],
            columns=['sampleid', 'col1', 'col2', 'col3'])

        exp = exp.set_index('sampleid')
        pdt.assert_frame_equal(exp, res)


class TestFilters(unittest.TestCase):

    def setUp(self):
        X = np.array(
            [[10, 1, 4, 1, 4, 0],
             [0, 0, 2, 0, 2, 8],
             [0, 1, 2, 1, 2, 4],
             [0, 1, 0, 1, 0, 0],
             [2, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0],
             [7, 1, 0, 1, 0, 0]]
        )
        oids = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7']
        sids = ['s1', 's2', 's3', 's4', 's5', 's6']

        bigX = np.array(
            [[10, 1, 4, 1, 4, 1, 0],
             [0, 0, 2, 0, 2, 1, 8],
             [0, 1, 2, 1, 2, 1, 4],
             [0, 1, 0, 1, 0, 1, 0],
             [2, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 1, 0],
             [4, 0, 0, 0, 0, 1, 0]]
        )

        self.big_table = Table(
            bigX, oids, sids + ['s9'],
        )

        self.metadata = pd.DataFrame(
            np.vstack(
                (
                    np.ones(8),
                    np.array(['a', 'a', 'b', 'b', 'a', 'a', 'b', 'a']),
                    np.arange(8).astype(np.float64),
                    np.array(['Test', 'Test', 'Train', 'Train',
                              'Train', 'Train', 'Test', 'Train'])
                )
            ).T,
            columns=['intercept', 'categorical', 'continuous', 'train'],
            index=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
        )
        self.metadata['continuous'] = self.metadata[
            'continuous'].astype(np.float64)
        self.trimmed_metadata = self.metadata.loc[
            ['s1', 's2', 's3', 's4', 's5', 's6']
        ]
        df = pd.DataFrame(
            [
                {'intercept': 1, 'categorical': 'b',
                 'continuous': 1., 'train': 'Train'},
                {'intercept': 1, 'categorical': 'b',
                 'continuous': 1., 'train': 'Train'}
            ], index=['s2', 's4']
        )
        df = df.reindex(columns=['intercept', 'categorical',
                                 'continuous', 'train'])
        self.metadata_dup = self.metadata.append(df)
        self.table = Table(X, oids, sids)

    def test_match_duplicate(self):
        formula = 'C(categorical) + continuous'
        res = match_and_filter(self.table, self.metadata_dup, formula,
                               min_sample_count=0, min_feature_count=0)
        res_table, res_metadata, res_design = res

        pdt.assert_frame_equal(res_table.to_dataframe(),
                               self.table.to_dataframe())

        exp_metadata = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array(['a', 'a', 'b', 'b', 'a', 'a']),
                    np.arange(6).astype(np.float64),
                    np.array(['Test', 'Test', 'Train', 'Train',
                              'Train', 'Train'])
                )
            ).T,
            columns=['intercept', 'categorical', 'continuous', 'train'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )
        exp_metadata['continuous'] = exp_metadata[
            'continuous'].astype(np.float64)
        pdt.assert_frame_equal(res_metadata, exp_metadata)
        exp_design = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array([0, 0, 1, 1, 0, 0]),
                    np.arange(6).astype(np.float64)
                )
            ).T,
            columns=['Intercept', 'C(categorical)[T.b]', 'continuous'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )

        pdt.assert_frame_equal(res_design, exp_design)

    def test_match_and_filter_no_filter(self):
        formula = 'C(categorical) + continuous'
        res = match_and_filter(self.table, self.metadata, formula,
                               min_sample_count=0, min_feature_count=0)
        res_table, res_metadata, res_design = res

        pdt.assert_frame_equal(res_table.to_dataframe(),
                               self.table.to_dataframe())

        exp_metadata = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array(['a', 'a', 'b', 'b', 'a', 'a']),
                    np.arange(6).astype(np.float64),
                    np.array(['Test', 'Test', 'Train', 'Train',
                              'Train', 'Train'])
                )
            ).T,
            columns=['intercept', 'categorical', 'continuous', 'train'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )
        exp_metadata['continuous'] = exp_metadata[
            'continuous'].astype(np.float64)
        pdt.assert_frame_equal(res_metadata, exp_metadata)
        exp_design = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array([0, 0, 1, 1, 0, 0]),
                    np.arange(6).astype(np.float64)
                )
            ).T,
            columns=['Intercept', 'C(categorical)[T.b]', 'continuous'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )

        pdt.assert_frame_equal(res_design, exp_design)

    def test_match_and_filter_big_table(self):
        formula = 'C(categorical) + continuous'
        res = match_and_filter(self.big_table, self.metadata, formula,
                               min_sample_count=0, min_feature_count=0)

        res_metadata = res[1]
        drop_metadata = res_metadata.dropna()
        res_design = res[2]
        drop_design = res_design.dropna()
        self.assertEqual(res_design.shape[0], drop_design.shape[0])
        self.assertEqual(res_metadata.shape[0], drop_metadata.shape[0])

    def test_split_training_random(self):
        np.random.seed(0)
        design = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array([0, 0, 1, 1, 0, 0]),
                    np.arange(6)
                )
            ).T,
            columns=['Intercept', 'C(categorical)[T.b]', 'continuous'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )
        res = split_training(self.table.to_dataframe().T,
                             self.trimmed_metadata, design,
                             training_column=None,
                             num_random_test_examples=2)

        trainX, testX, trainY, testY = res
        # print(trainX.shape, testX.shape, trainY.shape, testY.shape)
        npt.assert_allclose(trainX.shape, np.array([4, 3]))
        npt.assert_allclose(trainY.shape, np.array([4, 7]))

        npt.assert_allclose(testX.shape, np.array([2, 3]))
        npt.assert_allclose(testY.shape, np.array([2, 7]))

    def test_split_training_fixed(self):
        np.random.seed(0)
        design = pd.DataFrame(
            np.vstack(
                (
                    np.ones(6),
                    np.array([0, 0, 1, 1, 0, 0]),
                    np.arange(6)
                )
            ).T,
            columns=['Intercept', 'C(categorical)[T.b]', 'continuous'],
            index=['s1', 's2', 's3', 's4', 's5', 's6']
        )
        t = self.table.to_dataframe().T
        res = split_training(t,
                             self.metadata, design,
                             training_column='train',
                             num_random_test_examples=2)

        exp_trainX = design.iloc[2:].values
        exp_testX = design.iloc[:2].values
        exp_trainY = t.iloc[2:].values
        exp_testY = t.iloc[:2].values

        res_trainX, res_testX, res_trainY, res_testY = res

        npt.assert_allclose(exp_trainX, res_trainX)
        npt.assert_allclose(exp_trainY, res_trainY)
        npt.assert_allclose(exp_testX, res_testX)
        npt.assert_allclose(exp_testY, res_testY)


if __name__ == "__main__":
    unittest.main()
