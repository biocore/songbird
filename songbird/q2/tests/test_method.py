import qiime2
import unittest
import numpy as np
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
            formula="X", epochs=100000)

        npt.assert_allclose(exp_beta, res_beta.T, atol=0.6, rtol=0.6)
        self.assertGreater(len(res_stats.to_dataframe().index), 1)

        # test biplot
        self.assertIsInstance(res_biplot, OrdinationResults)
        u = res_biplot.samples.values
        v = res_biplot.features.values.T
        npt.assert_allclose(u @ v, np.array(exp_beta), atol=0.5, rtol=0.5)


if __name__ == "__main__":
    unittest.main()
