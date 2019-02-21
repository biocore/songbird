import qiime2
import unittest
import numpy as np
import tensorflow as tf
from songbird.q2._method import multinomial, regression_biplot
from songbird.util import random_multinomial_model

from skbio import OrdinationResults
from skbio.stats.composition import clr, clr_inv, centralize
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

        res_beta, res_stats = multinomial(table=self.table, metadata=md,
                                          formula="X", epoch=50000)
        npt.assert_allclose(exp_beta, res_beta.T, atol=0.5, rtol=0.5)
        self.assertGreater(len(res_stats.to_dataframe().index), 1)


class TestRegressionBiplot(unittest.TestCase):

    def setUp(self):
        res = random_multinomial_model(
            num_samples=200, num_features=5,
            reps=1,
            low=-1, high=1,
            beta_mean=0,
            beta_scale=1,
            mu=1000,  # sequencing depth
            sigma=0.01,
            seed=0)

        self.table, self.md, self.beta = res

    def test_biplot(self):
        exp = clr(centralize(clr_inv(self.beta)))
        res = regression_biplot(self.beta)
        self.assertIsInstance(res, OrdinationResults)
        u = res.samples.values
        v = res.features.values.T
        npt.assert_allclose(u @ v, np.array(exp), atol=0.5, rtol=0.5)


if __name__ == "__main__":
    unittest.main()
