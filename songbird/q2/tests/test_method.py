import qiime2
import unittest
import numpy as np
import contextlib
import io
import tensorflow as tf
from songbird.q2._method import multinomial
from songbird.util import random_multinomial_model

from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv
import numpy.testing as npt


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

    def test_quiet(self):
        tf.set_random_seed(0)
        md = self.md

        md.name = 'sampleid'
        md = qiime2.Metadata(md)

        exp_beta = clr(clr_inv(np.hstack((np.zeros((2, 1)), self.beta.T))))

        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            res_beta, res_stats, res_biplot = multinomial(
                table=self.table, metadata=md,
                min_sample_count=0, min_feature_count=0,
                formula="X", epochs=1000, quiet=True)
        assert f.getvalue() == ""

    def test_not_quiet(self):
        tf.set_random_seed(0)
        md = self.md

        md.name = 'sampleid'
        md = qiime2.Metadata(md)

        exp_beta = clr(clr_inv(np.hstack((np.zeros((2, 1)), self.beta.T))))

        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            res_beta, res_stats, res_biplot = multinomial(
                table=self.table, metadata=md,
                min_sample_count=0, min_feature_count=0,
                formula="X", epochs=1000)
        assert f.getvalue() != ""

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


if __name__ == "__main__":
    unittest.main()
