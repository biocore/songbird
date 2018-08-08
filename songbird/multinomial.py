import tensorflow as tf
import os
import numpy as np
import pandas as pd
from biom import load_table, Table

from gneiss.balances import _balance_basis
from gneiss.composition import ilr_transform
from gneiss.util import match, match_tips, rename_internal_nodes

from tensorflow.contrib.distributions import Multinomial, Normal
from patsy import dmatrix
from skbio import TreeNode
from skbio.stats.composition import closure, clr_inv, clr
from scipy.stats import spearmanr
from tqdm import tqdm
import time
import datetime


class MultRegression(object):

    def __init__(self, beta_mean=0, beta_scale=1,
                 batch_size=50, learning_rate=0.1, beta_1=0.9, beta_2=0.99,
                 clipnorm=10., save_path=None):
        """ Build a tensorflow model

        Returns
        -------
        loss : tf.Tensor
           The log loss of the model.

        """
        if save_path is None:
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

        # Place holder variables to accept input data
        self.X_ph = tf.constant(trainX, dtype=tf.float32)
        self.Y_ph = tf.constant(trainY, dtype=tf.float32)
        self.X_holdout = tf.constant(testX, dtype=tf.float32)
        self.Y_holdout = tf.constant(testY, dtype=tf.float32)
        holdout_count = tf.reduce_sum(self.Y_holdout, axis=1)

        # subsample datasets
        batch_ids = tf.multinomial(tf.zeros([1, self.N]),
                                   self.batch_size)
        sample_ids = tf.squeeze(batch_ids)

        Y_batch = tf.gather(self.Y_ph, sample_ids, axis=0)
        X_batch = tf.gather(self.X_ph, sample_ids, axis=0)
        total_count = tf.reduce_sum(Y_batch, axis=1)

        # Define PointMass Variables first
        self.qbeta = tf.Variable(tf.random_normal([self.p, self.D-1]), name='qB')

        # Distributions
        # regression coefficents distribution
        beta = Normal(loc=tf.zeros([self.p, self.D-1]) + self.beta_mean,
                      scale=tf.ones([self.p, self.D-1]) * self.beta_scale,
                      name='B')

        Bprime = tf.concat([tf.zeros([self.p, 1]), self.qbeta], axis=1)
        eta = tf.matmul(X_batch, Bprime)
        phi = tf.nn.log_softmax(eta)
        Y = Multinomial(total_count=total_count, logits=phi, name='Y')

        # cross validation
        with tf.name_scope('accuracy'):
            pred =  tf.reshape(holdout_count, [-1, 1]) * tf.nn.softmax(
                tf.matmul(self.X_holdout, Bprime))
            self.cv = tf.reduce_mean(tf.squeeze(tf.abs(pred - self.Y_holdout)))
            tf.summary.scalar('mean_absolute_error', self.cv)

        self.loss = -(tf.reduce_mean(beta.log_prob(self.qbeta)) + \
                tf.reduce_mean(Y.log_prob(Y_batch)) * (self.N / self.batch_size))
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=self.beta_1, beta2=self.beta_2)

        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(gradients, self.clipnorm)
        self.train = optimizer.apply_gradients(zip(gradients, variables))

        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('qbeta', self.qbeta)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.save_path, self.session.graph)
        tf.global_variables_initializer().run()


    def fit(self, epoch=10, summary_interval=1000, checkpoint_interval=3600):
        """ Fits the model.

        Parameters
        ----------
        epoch : int
           Number of epochs to train
        summary_interval : int
           Number of seconds until a summary is recorded
        checkpoint_interval : int
           Number of seconds until a checkpoint is recorded

        Returns
        -------
        loss: float
            log likelihood loss.
        cv : float
            cross validation loss
        """
        num_iter = (self.N // self.batch_size) * epoch
        idx = np.arange(self.N)
        cv = None
        last_checkpoint_time = 0
        last_summary_time = 0
        start_time = time.time()
        saver = tf.train.Saver()

        for i in tqdm(range(0, num_iter)):
            batch_idx = np.random.choice(idx, size=self.batch_size)

            now = time.time()

            if now - last_summary_time > summary_interval:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, cv, summary, train_loss, grads, B = self.session.run(
                    [self.train, self.cv, self.merged,
                     self.loss, self.gradients, self.qbeta],
                    options=run_options,
                    run_metadata=run_metadata
                )
                self.writer.add_run_metadata(run_metadata, 'step%d' % i)
                self.writer.add_summary(summary, i)
                last_summary_time = now
            else:
                _, summary, train_loss, grads = self.session.run(
                    [self.train, self.merged, self.loss, self.gradients],
                )
                self.writer.add_summary(summary, i)

            if now - last_checkpoint_time > checkpoint_interval:
                saver.save(self.session,
                           os.path.join(self.save_path, "model.ckpt"),
                           global_step=i)
                last_checkpoint_time = now

        self.B = B
        return train_loss, cv


    def predict(self, X):
        """ Performs a prediction

        Parameters
        ----------
        X : np.array
           Input table (likely OTUs).

        Returns
        -------
        np.array :
           Predicted abundances.
        """
        pass

    def cross_validate(self, X, Y):
        """
        Parameters
        ----------
        X : np.array
           Input table (likely OTUs).
        Y : np.array
           Output table (likely metabolites).

        Returns
        -------
        cv_loss: float
           Euclidean norm of the errors (i.e. the RMSE)

        """
        pass


flags = tf.app.flags
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_biom", None, "Input biom table. "
                    "i.e. input.biom")
flags.DEFINE_string("test_biom", None, "Input biom table. "
                    "i.e. input.biom")
flags.DEFINE_string("train_metadata", None, "Input sample metadata. "
                    "i.e. metadata.txt")
flags.DEFINE_string("test_metadata", None, "Input sample metadata. "
                    "i.e. metadata.txt")
flags.DEFINE_string("tree", None, "Input tree. "
                    "i.e. tree.nwk")
flags.DEFINE_string("formula", None, "Statistical formula for "
                    "specifying covariates.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_float("clipping_size", 10, "Gradient clipping size.")
flags.DEFINE_float("beta_mean", 0,
                   'Mean of prior distribution for covariates')
flags.DEFINE_float("beta_scale", 1.0,
                   'Scale of prior distribution for covariates')
flags.DEFINE_float("gamma_mean", 0,
                   'Mean of prior distribution for sample bias')
flags.DEFINE_float("gamma_scale", 1.0,
                   'Scale of prior distribution for sample bias')
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training sample.")
flags.DEFINE_integer("batch_size", 5,
                     "Number of training samples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("min_sample_count", 1000,
                     "The minimum number of counts a feature needs for it to be "
                     "included in the analysis")
flags.DEFINE_integer("min_feature_count", 5,
                     "The minimum number of counts a sample needs to be  "
                     "included in the analysis")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")
flags.DEFINE_boolean("verbose", False,
                     "Specifies if cross validation and summaries are "
                     "saved during training. ")
FLAGS = flags.FLAGS


class Options(object):
  """Options used by our Regression model."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    if isinstance(self.train_biom, str):
      self.train_table = load_table(self.train_biom)
    elif isinstance(self.train_biom, Table):
      self.train_table = self.train_biom
    if isinstance(self.test_biom, str):
      self.test_table = load_table(self.test_biom)
    elif isinstance(self.test_biom, Table):
      self.test_table = self.test_biom

    if isinstance(self.train_metadata, str):
      cols = pd.read_csv(self.train_metadata, nrows=1, sep='\t').columns.tolist()
      self.train_metadata = pd.read_table(
        self.train_metadata, dtype={cols[0]: object})
      self.train_metadata = self.train_metadata.set_index(cols[0])
    elif isinstance(self.train_metadata, pd.DataFrame):
      self.train_metadata = self.train_metadata

    if isinstance(self.test_metadata, str):
      cols = pd.read_csv(self.test_metadata, nrows=1, sep='\t').columns.tolist()
      self.test_metadata = pd.read_table(
        self.test_metadata, dtype={cols[0]: object})
      self.test_metadata = self.test_metadata.set_index(cols[0])
    elif isinstance(self.test_metadata, pd.DataFrame):
      self.test_metadata = self.test_metadata

    try:
      if isinstance(self.tree_file, str):
        self.tree = TreeNode.read(self.tree_file)
      elif isinstance(self.tree_file, TreeNode):
        self.tree = self.tree_file
    except:
      pass

    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    self.formula = self.formula


def main(_):

  opts = Options(
    save_path=FLAGS.save_path,
    train_biom=FLAGS.train_biom,
    test_biom=FLAGS.test_biom,
    train_metadata=FLAGS.train_metadata,
    test_metadata=FLAGS.test_metadata,
    formula=FLAGS.formula,
    learning_rate=FLAGS.learning_rate,
    clipping_size=FLAGS.clipping_size,
    beta_mean=FLAGS.beta_mean,
    beta_scale=FLAGS.beta_scale,
    gamma_mean=FLAGS.gamma_mean,
    gamma_scale=FLAGS.gamma_scale,
    epochs_to_train=FLAGS.epochs_to_train,
    num_neg_samples=FLAGS.num_neg_samples,
    batch_size=FLAGS.batch_size,
    min_sample_count=FLAGS.min_sample_count,
    min_feature_count=FLAGS.min_feature_count,
    statistics_interval=FLAGS.statistics_interval,
    summary_interval=FLAGS.summary_interval,
    checkpoint_interval=FLAGS.checkpoint_interval
  )
  # preprocessing
  train_table, train_metadata = opts.train_table, opts.train_metadata
  train_metadata = train_metadata.loc[train_table.ids(axis='sample')]

  sample_filter = lambda val, id_, md: (
    (id_ in train_metadata.index) and np.sum(val) > opts.min_sample_count)
  read_filter = lambda val, id_, md: np.sum(val) > opts.min_feature_count
  metadata_filter = lambda val, id_, md: id_ in train_metadata.index

  train_table = train_table.filter(metadata_filter, axis='sample')
  train_table = train_table.filter(sample_filter, axis='sample')
  train_table = train_table.filter(read_filter, axis='observation')
  train_metadata = train_metadata.loc[train_table.ids(axis='sample')]

  sort_f = lambda xs: [xs[train_metadata.index.get_loc(x)] for x in xs]
  train_table = train_table.sort(sort_f=sort_f, axis='sample')
  train_metadata = dmatrix(opts.formula, train_metadata, return_type='dataframe')

  metadata_filter = lambda val, id_, md: id_ in train_metadata.index
  train_table = train_table.filter(metadata_filter, axis='sample')

  # hold out data preprocessing
  test_table, test_metadata = opts.test_table, opts.test_metadata
  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  obs_lookup = set(train_table.ids(axis='observation'))
  feat_filter = lambda val, id_, md: id_ in obs_lookup

  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  test_table = test_table.filter(metadata_filter, axis='sample')
  test_table = test_table.filter(feat_filter, axis='observation')
  test_metadata = test_metadata.loc[test_table.ids(axis='sample')]
  sort_f = lambda xs: [xs[test_metadata.index.get_loc(x)] for x in xs]
  test_table = test_table.sort(sort_f=sort_f, axis='sample')
  test_metadata = dmatrix(opts.formula, test_metadata, return_type='dataframe')
  metadata_filter = lambda val, id_, md: id_ in test_metadata.index
  test_table = test_table.filter(metadata_filter, axis='sample')

  p = train_metadata.shape[1]   # number of covariates
  G_data = train_metadata.values
  y_data = np.array(train_table.matrix_data.todense()).T
  y_test = np.array(test_table.matrix_data.todense()).T
  N, D = y_data.shape

  save_path = opts.save_path
  learning_rate = opts.learning_rate
  batch_size = opts.batch_size
  beta_mean, beta_scale = opts.beta_mean, opts.beta_scale

  num_iter = (N // batch_size) * opts.epochs_to_train
  holdout_size = test_metadata.shape[0]
  checkpoint_interval = opts.checkpoint_interval

  # Model code
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      # Place holder variables to accept input data
      G_ph = tf.placeholder(tf.float32, [batch_size, p], name='G_ph')
      Y_ph = tf.placeholder(tf.float32, [batch_size, D], name='Y_ph')
      G_holdout = tf.placeholder(tf.float32, [holdout_size, p], name='G_holdout')
      Y_holdout = tf.placeholder(tf.float32, [holdout_size, D], name='Y_holdout')
      total_count = tf.placeholder(tf.float32, [batch_size], name='total_count')

      # Define PointMass Variables first
      qbeta = tf.Variable(tf.random_normal([p, D-1]), name='qB')

      # Distributions
      # regression coefficents distribution
      beta = Normal(loc=tf.zeros([p, D-1]) + beta_mean,
                    scale=tf.ones([p, D-1]) * beta_scale,
                    name='B')

      Bprime = tf.concat([tf.zeros([p, 1]), qbeta], axis=1)
      eta = tf.matmul(G_ph, Bprime)
      phi = tf.nn.log_softmax(eta)
      Y = Multinomial(total_count=total_count, logits=phi, name='Y')

      loss = -(tf.reduce_mean(beta.log_prob(qbeta)) + \
               tf.reduce_mean(Y.log_prob(Y_ph)) * (N / batch_size))
      optimizer = tf.train.AdamOptimizer(learning_rate)

      gradients, variables = zip(*optimizer.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, opts.clipping_size)
      train = optimizer.apply_gradients(zip(gradients, variables))

      with tf.name_scope('accuracy'):
        holdout_count = tf.reduce_sum(Y_holdout, axis=1)
        pred =  tf.reshape(holdout_count, [-1, 1]) * tf.nn.softmax(
          tf.matmul(G_holdout, Bprime))
        mse = tf.reduce_mean(tf.squeeze(tf.abs(pred - Y_holdout)))
        tf.summary.scalar('mean_absolute_error', mse)

      tf.summary.scalar('loss', loss)
      tf.summary.histogram('qbeta', qbeta)
      merged = tf.summary.merge_all()

      tf.global_variables_initializer().run()

      writer = tf.summary.FileWriter(save_path, session.graph)

      losses = np.array([0.] * num_iter)
      idx = np.arange(train_metadata.shape[0])
      log_handle = open(os.path.join(save_path, 'run.log'), 'w')

      last_checkpoint_time = 0
      start_time = time.time()
      saver = tf.train.Saver()

      for i in tqdm(range(0, num_iter)):
          batch_idx = np.random.choice(idx, size=batch_size)

          feed_dict={
              Y_ph: y_data[batch_idx].astype(np.float32),
              G_ph: train_metadata.values[batch_idx].astype(np.float32),
              Y_holdout: y_test.astype(np.float32),
              G_holdout: test_metadata.values.astype(np.float32),
              total_count: y_data[batch_idx].sum(axis=1).astype(np.float32)
          }

          if i % 1000 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary, train_loss, grads = session.run(
              [train, merged, loss, gradients],
              feed_dict=feed_dict,
              options=run_options,
              run_metadata=run_metadata
            )
            writer.add_run_metadata(run_metadata, 'step%d' % i)
            writer.add_summary(summary, i)
          elif i % 5000 == 0:
            _, summary, err, train_loss, grads = session.run(
              [train, mse, merged, loss, gradients],
              feed_dict=feed_dict
            )
            writer.add_summary(summary, i)
          else:
            _, summary, train_loss, grads = session.run(
              [train, merged, loss, gradients],
              feed_dict=feed_dict
            )
            writer.add_summary(summary, i)

          now = time.time()
          if now - last_checkpoint_time > checkpoint_interval:
            saver.save(session,
                       os.path.join(opts.save_path, "model.ckpt"),
                       global_step=i)
            last_checkpoint_time = now

          losses[i] = train_loss
      elapsed_time = time.time() - start_time
      print('Elapsed Time: %f seconds' % elapsed_time)

      # Cross validation
      pred_beta = Bprime.eval()
      mse, mrc = cross_validation(test_metadata.values,
                                  pred_beta,  y_test)
      print("MSE: %f, MRC: %f" % (mse, mrc))

      md_ids = np.array(train_metadata.columns)
      samp_ids = train_table.ids(axis='sample')
      obs_ids = train_table.ids(axis='observation')

      _, summary, train_loss, grads, beta_ = session.run(
        [train, merged, loss, gradients, qbeta],
        feed_dict=feed_dict
      )

      beta_ = clr(clr_inv(np.hstack((np.zeros((p, 1)), beta_))))
      pd.DataFrame(
        beta_, index=md_ids, columns=obs_ids,
      ).to_csv(os.path.join(save_path, 'beta.csv'))


if __name__ == "__main__":
  tf.app.run()
