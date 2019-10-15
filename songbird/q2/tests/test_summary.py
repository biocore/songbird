import os
import shutil
import qiime2
import unittest
import tensorflow as tf
from songbird.q2._method import multinomial
from songbird.q2._summary import summarize_single, summarize_paired, _summarize
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
        self.ref_beta, self.ref_stats, _ = multinomial(
            table=self.table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            summary_interval=1,
            formula="X", epochs=1000)

        self.base_beta, self.base_stats, _ = multinomial(
            table=self.table, metadata=md,
            min_sample_count=0, min_feature_count=0,
            summary_interval=1,
            formula="1", epochs=1000)
        self.results = "results"
        if not os.path.exists(self.results):
            os.mkdir(self.results)

    def tearDown(self):
        shutil.rmtree(self.results)

    def test_summarize_single(self):
        summarize_single(self.results, self.ref_stats)

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
            self.assertIn("how to interpret these plots", html)
            self.assertIn(
                "how to adjust Songbird's parameters to get the model to fit",
                html
            )

    def test_summarize_error(self):
        """Tests that a certain error in _summarize() is raised if needed.

        This error should be raised if "baseline" is not None (i.e. we're
        calling this from summarize_paired()), but n is None (i.e. we don't
        have data on the number of samples).

        This should never happen in practice, but we might as well test to make
        sure that an appropriate error is thrown if it ever *would* happen.
        """
        with self.assertRaisesRegex(
            ValueError, "n is None, but baseline is not None."
        ):
            _summarize(self.results, self.ref_stats, self.base_stats)

    def test_summarize_paired(self):
        summarize_paired(self.results,
                         self.ref_stats,
                         self.base_stats)

        index_fp = os.path.join(self.results, 'index.html')
        self.assertTrue(os.path.exists(index_fp))

        with open(index_fp, 'r') as fh:
            html = fh.read()
            self.assertIn('<h1>Convergence summary</h1>', html)
            self.assertIn('Pseudo Q-squared', html)

            self.assertIn(
                '<img src="convergence-plot.svg" alt="convergence_plots">',
                html
            )
            self.assertIn('<a href="convergence-plot.pdf">', html)
            self.assertIn("how to interpret these plots", html)
            self.assertIn(
                "how to adjust Songbird's parameters to get the model to fit",
                html
            )


if __name__ == "__main__":
    unittest.main()
