import os
import shutil
import qiime2
import unittest
import tensorflow as tf
from songbird.q2._method import multinomial
from songbird.q2._summary import summarize_single, summarize_paired
from songbird.util import random_multinomial_model


class TestSummary(unittest.TestCase):

    def setUp(self):
        res = random_multinomial_model(
            num_samples=50, num_features=5,
            reps=1,
            low=-1, high=1,
            beta_mean=0,
            beta_scale=1,
            mu=1000,  # sequencing depth
            sigma=0.5,
            seed=0)
        self.table, self.md, self.beta = res

        tf.set_random_seed(0)
        md = self.md
        md.name = 'sampleid'
        md = qiime2.Metadata(md)
        self.ref_beta, self.ref_stats = multinomial(
            table=self.table, metadata=md,
            summary_interval=1,
            formula="X", epoch=50000)

        self.base_beta, self.base_stats = multinomial(
            table=self.table, metadata=md,
            summary_interval=1,
            formula="1", epoch=50000)
        self.results = "results"
        if not os.path.exists(self.results):
            os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_summarize_single(self):
        summarize_single(self.results, self.table, self.ref_stats)

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Convergence summary</h1>', html)
            self.assertIn(
                '<img src="convergence-plot.svg" alt="convergence_plots">',
                html
            )
            self.assertIn('<a href="convergence-plot.pdf">', html)

    def test_summarized_paired(self):
        summarized_paired(self.results, self.table,
                          self.ref_stats, self.base_stats)

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Convergence summary</h1>', html)
            self.assertIn(
                '<img src="convergence-plot.svg" alt="convergence_plots">',
                html
            )
            self.assertIn('<a href="convergence-plot.pdf">', html)


if __name__ == "__main__":
    unittest.main()
