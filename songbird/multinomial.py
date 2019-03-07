import tensorflow as tf
import os
from tensorflow.contrib.distributions import Multinomial, Normal
from tqdm import tqdm
import time
import datetime
import numpy as np


class MultRegression(object):

    def __init__(self, beta_mean=0, beta_scale=1,
                 batch_size=5, learning_rate=0.001, beta_1=0.9, beta_2=0.99,
                 clipnorm=10., save_path=""):
        """ Build a tensorflow model

        Returns
        -------
        loss : tf.Tensor
           The log loss of the model.

        Notes
        -----
        If save_path is None, there won't be anything saved
        """
        if save_path == "":
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = "_".join([basename, suffix])

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.batch_size = batch_size
        self.clipnorm = clipnorm

        self.beta_mean = beta_mean
        self.beta_scale = beta_scale
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clipnorm = clipnorm
        self.save_path = save_path

    def __call__(self, session, trainX, trainY, testX, testY):
        """ Initialize the actual graph

        Parameters
        ----------
        session : tf.Session
            Tensorflow session
        trainX : np.array
            Input training design matrix.
        trainY : np.array
            Output training OTU table, where rows are samples and columns are
            observations.
        testX : np.array
            Input testing design matrix.
        testY : np.array
            Output testing OTU table, where rows are samples and columns are
            observations.
        """
        self.session = session
        self.N, self.p = trainX.shape
        self.D = trainY.shape[1]
        holdout_size = testX.shape[0]

        # Place holder variables to accept input data
        self.X_ph = tf.constant(trainX, dtype=tf.float32, name='G_ph')
        self.Y_ph = tf.constant(trainY, dtype=tf.float32, name='Y_ph')
        self.X_holdout = tf.constant(testX, dtype=tf.float32, name='G_holdout')
        self.Y_holdout = tf.constant(testY, dtype=tf.float32, name='Y_holdout')

        batch_ids = tf.multinomial(tf.ones([1, self.N]), self.batch_size)
        sample_ids = tf.squeeze(batch_ids)

        Y_batch = tf.gather(self.Y_ph, sample_ids, axis=0)
        X_batch = tf.gather(self.X_ph, sample_ids, axis=0)

        total_count = tf.reduce_sum(Y_batch, axis=1)
        holdout_count = tf.reduce_sum(self.Y_holdout, axis=1)

        # Define PointMass Variables first
        self.qbeta = tf.Variable(
            tf.random_normal([self.p, self.D-1]), name='qB')

        # regression coefficents distribution
        beta = Normal(loc=tf.zeros([self.p, self.D-1]) + self.beta_mean,
                      scale=tf.ones([self.p, self.D-1]) * self.beta_scale,
                      name='B')

        eta = tf.matmul(X_batch, self.qbeta, name='eta')

        phi = tf.nn.log_softmax(
            tf.concat(
                [tf.zeros([self.batch_size, 1]), eta], axis=1), name='phi'
        )

        Y = Multinomial(total_count=total_count, logits=phi, name='Y')

        # cross validation
        with tf.name_scope('accuracy'):
            pred = tf.reshape(
                holdout_count, [-1, 1]) * tf.nn.softmax(
                    tf.concat([
                        tf.zeros([holdout_size, 1]),
                        tf.matmul(self.X_holdout, self.qbeta)
                    ], axis=1), name='phi'
                )

            self.cv = tf.reduce_mean(
                tf.squeeze(tf.abs(pred - self.Y_holdout))
            )
            tf.summary.scalar('cv_error', self.cv)

        self.loss = -(tf.reduce_sum(beta.log_prob(self.qbeta)) +
                      tf.reduce_sum(Y.log_prob(Y_batch)) *
                      (self.N / self.batch_size))

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=self.beta_1, beta2=self.beta_2)

        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.clipnorm)
        self.train = optimizer.apply_gradients(zip(gradients, variables))

        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('qbeta', self.qbeta)
        self.merged = tf.summary.merge_all()
        if self.save_path is not None:
            self.writer = tf.summary.FileWriter(self.save_path,
                                                self.session.graph)
        else:
            self.writer = None
        tf.global_variables_initializer().run()

    def fit(self, epochs=10, summary_interval=100, checkpoint_interval=3600):
        """ Fits the model.

        Parameters
        ----------
        epochs : int
           Number of epochs to train
        summary_interval : int
           Number of seconds until a summary is recorded
        checkpoint_interval : int
           Number of seconds until a checkpoint is recorded

        Returns
        -------
        loss: np.array
            log likelihood loss.
        cv : np.array
            cross validation loss
        iter_n : np.array
            iterations
        """
        num_iter = (self.N // self.batch_size) * epochs
        cv = None
        last_checkpoint_time = 0
        last_summary_time = 0
        saver = tf.train.Saver()
        loss = []
        cv = []
        iter_n = []

        for i in tqdm(range(0, num_iter)):
            now = time.time()

            if now - last_summary_time > summary_interval:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, summary, train_loss, test_cv, grads = self.session.run(
                    [self.train, self.merged,
                     self.loss, self.cv, self.gradients],
                    options=run_options,
                    run_metadata=run_metadata
                )
                cv.append(test_cv)
                loss.append(train_loss)
                iter_n.append(i)

                if self.writer is not None:
                    self.writer.add_run_metadata(run_metadata, 'step%d' % i)
                    self.writer.add_summary(summary, i)
                last_summary_time = now
            else:
                _, summary, train_loss, grads = self.session.run(
                    [self.train, self.merged, self.loss, self.gradients],
                )
                if self.writer is not None:
                    self.writer.add_summary(summary, i)

            if checkpoint_interval is not None:
                if now - last_checkpoint_time > checkpoint_interval:
                    saver.save(self.session,
                               os.path.join(self.save_path, "model.ckpt"),
                               global_step=i)
                last_checkpoint_time = now

        train_loss, test_cv, B = self.session.run(
            [self.loss, self.cv, self.qbeta]
        )
        cv.append(test_cv)
        loss.append(train_loss)
        iter_n.append(i)

        self.B = B
        return np.array(loss), np.array(cv), np.array(iter_n)
