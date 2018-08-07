import tensorflow as tf
import unittest
import numpy as np
from songbird.multinomial import MultRegression
from songbird.util import random_multinomial_model


class TestMultRegression(unittest.TestCase):

    def setUp(self):
        res = random_multinomial_model(
            num_samples=20, num_features=10,
            reps=1,
            low=2, high=10,
            beta_mean=0,
            beta_scale=0.5,
            mu = 6,
            sigma = 0.1,
            seed=0)
        self.table, self.md, self.beta = res

    def test_init(self):
        model = MultRegression()

    def test_call(self):
        model = MultRegression()
        Y = np.array(self.table.matrix_data.todense()).T
        X = self.md.values
        trainX = X[:15]
        trainY = Y[:15]
        testX = X[15:]
        testY = Y[15:]
        with tf.Graph().as_default(), tf.Session() as session:
            model(session, trainX, trainY, testX, testY)


if __name__ == "__main__":
    unittest.main()
