import tensorflow as tf
import unittest
import numpy as np
import numpy.testing as npt
from songbird.multinomial import MultRegression
from songbird.util import random_multinomial_model


class TestMultRegression(unittest.TestCase):

    def setUp(self):
        res = random_multinomial_model(
            num_samples=150, num_features=50,
            reps=1,
            low=-1, high=1,
            beta_mean=0,
            beta_scale=1,
            mu=1000,
            sigma=0.5,
            seed=0)

        self.table, self.md, self.beta = res

    def test_fit(self):
        tf.set_random_seed(0)
        model = MultRegression(
            batch_size=100, learning_rate=1e-3, beta_scale=1)
        Y = np.array(self.table.matrix_data.todense()).T
        X = self.md.values
        trainX = X[:-5]
        trainY = Y[:-5]
        testX = X[-5:]
        testY = Y[-5:]
        print(trainY.shape)
        with tf.Graph().as_default(), tf.Session() as session:
            model(session, trainX, trainY, testX, testY)
            model.fit(epoch=int(10000))

        npt.assert_allclose(self.beta, model.B.T, atol=0.2, rtol=0.2)


if __name__ == "__main__":
    unittest.main()
