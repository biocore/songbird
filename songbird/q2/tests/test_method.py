import qiime2
import unittest
import numpy as np
import tensorflow as tf
from songbird.q2._method import multinomial
from songbird.util import random_multinomial_model

from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv
import numpy.testing as npt
import pandas as pd
from songbird.q2.plugin_setup import plugin as songbird_plugin


class TestMultinomial(unittest.TestCase):

    def setUp(self):
        res = random_multinomial_model(
            num_samples=200, num_features=15,
            reps=1,
            low=-1, high=1,
            beta_mean=0,
            beta_scale=1,
            mu=1000,  # sequencing depth
            sigma=0.01,
            seed=0)

        self.table, self.md, self.beta = res

    def test_fit(self):
        tf.set_random_seed(0)
        md = self.md

        md.name = 'sampleid'
        md = qiime2.Metadata(md)

        exp_beta = clr(clr_inv(np.hstack((np.zeros((2, 1)), self.beta.T))))

        res_beta, res_stats, res_biplot = multinomial(
            table=self.table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            formula="X", epochs=1000)

        # test biplot
        self.assertIsInstance(res_biplot, OrdinationResults)
        u = res_biplot.samples.values
        v = res_biplot.features.values.T
        npt.assert_allclose(u @ v, res_beta.values,
                            atol=0.5, rtol=0.5)

        npt.assert_allclose(exp_beta, res_beta.T, atol=0.6, rtol=0.6)
        self.assertGreater(len(res_stats.to_dataframe().index), 1)

    def test_fit_consistency(self):
        md = self.md

        md.name = 'sampleid'
        md = qiime2.Metadata(md)

        res_beta1, res_stats1, res_biplot1 = multinomial(
            table=self.table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            formula="X", epochs=1000)

        res_beta2, res_stats2, res_biplot2 = multinomial(
            table=self.table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            formula="X", epochs=1000)

        npt.assert_array_equal(res_beta1, res_beta2)
        end_res_stats1 = res_stats1.to_dataframe().iloc[-1]
        end_res_stats2 = res_stats2.to_dataframe().iloc[-1]
        npt.assert_array_equal(end_res_stats1, end_res_stats2)
        npt.assert_array_equal(res_biplot1.eigvals, res_biplot2.eigvals)
        npt.assert_array_equal(res_biplot1.samples, res_biplot2.samples)
        npt.assert_array_equal(res_biplot1.features, res_biplot2.features)

    def test_fit_float_summary_interval(self):
        tf.set_random_seed(0)
        md = self.md

        multregression = songbird_plugin.actions['multinomial']

        md.name = 'sampleid'
        md = qiime2.Metadata(md)

        exp_beta = clr(clr_inv(np.hstack((np.zeros((2, 1)), self.beta.T))))

        q2_table = qiime2.Artifact.import_data('FeatureTable[Frequency]',
                                               self.table)

        q2_res_beta, q2_res_stats, q2_res_biplot = multregression(
            table=q2_table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            formula="X", epochs=1000,
            summary_interval=0.5,
        )

        try:
            res_biplot = q2_res_biplot.view(OrdinationResults)
        except Exception:
            raise AssertionError('res_biplot unable to be coerced to '
                                 'OrdinationResults')
        try:
            res_beta = q2_res_beta.view(pd.DataFrame)
        except Exception:
            raise AssertionError('res_beta unable to be coerced to '
                                 'pd.DataFrame')
        try:
            res_stats = q2_res_stats.view(qiime2.Metadata)
        except Exception:
            raise AssertionError('res_stats unable to be coerced to '
                                 'qiime2.Metadata')

        u = res_biplot.samples.values
        v = res_biplot.features.values.T
        npt.assert_allclose(u @ v, res_beta.values,
                            atol=0.5, rtol=0.5)

        npt.assert_allclose(exp_beta, res_beta.T, atol=0.6, rtol=0.6)
        self.assertGreater(len(res_stats.to_dataframe().index), 1)


if __name__ == "__main__":
    unittest.main()
